"""
로컬 Whisper STT — 유튜브 자막이 없는 설교 영상을 음성인식으로 텍스트화한다.

유튜브 자동자막이 있는 영상은 publish_sermon.py가 그대로 자막을 쓰고,
자막이 없는 영상만 이 모듈로 넘어온다.

  pip install faster-whisper yt-dlp nvidia-cublas-cu12 nvidia-cudnn-cu12

GPU: GTX 1070 Ti (Pascal, CC 6.1) — float16 미지원이라 int8_float32로 돌린다.
"""

import glob
import os
import re
import site
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# CUDA DLL 경로 등록 (Windows) — ctranslate2 import 전에 반드시 실행
# ---------------------------------------------------------------------------


def _register_cuda_dlls():
    """
    pip로 설치한 cuBLAS/cuDNN DLL을 찾게 해준다.

    ctranslate2는 런타임에 LoadLibrary로 cublas64_12.dll을 부르는데,
    이 방식은 add_dll_directory를 참조하지 않는다. PATH에 직접 넣어야 한다.
    """
    dirs = []
    for p in site.getsitepackages():
        dirs.extend(glob.glob(os.path.join(p, "nvidia", "*", "bin")))

    if not dirs:
        return

    for d in dirs:
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass

    os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dlls()


BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "audio_cache"
# 전사 결과는 디스크에 남긴다. 96편 = GPU 10시간짜리 작업이라
# 요약 단계가 실패해도 다시 전사하는 일이 없어야 한다.
TRANSCRIPT_DIR = BASE_DIR / "transcripts"

# Pascal 세대는 fp16 가속이 없다. int8_float32가 이 GPU의 최적 조합.
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8_float32")

# 고유명사 보정용 프롬프트. 유튜브 자동자막에는 없는 이점으로,
# 설교에 반복 등장하는 성경 용어·교회 용어를 미리 물려준다.
INITIAL_PROMPT = (
    "생명샘명성교회 권영철 목사의 주일예배 설교입니다. "
    "하나님, 여호와, 예수 그리스도, 성령님, 아버지 하나님, "
    "에벤에셀, 임마누엘, 할렐루야, 아멘, 샬롬, 여호와 이레, "
    "아브라함, 이삭, 야곱, 요셉, 모세, 여호수아, 사무엘, 다윗, 솔로몬, "
    "엘리야, 엘리사, 히스기야, 느헤미야, 에스더, 다니엘, 바울, 베드로, "
    "창세기, 출애굽기, 신명기, 여호수아, 사무엘상, 사무엘하, 열왕기, "
    "시편, 잠언, 전도서, 이사야, 예레미야, 에스겔, 호세아, 말라기, "
    "마태복음, 마가복음, 누가복음, 요한복음, 사도행전, 로마서, "
    "고린도전서, 고린도후서, 갈라디아서, 에베소서, 빌립보서, 골로새서, "
    "데살로니가전서, 디모데전서, 히브리서, 야고보서, 베드로전서, 요한계시록, "
    "은혜, 축복, 구원, 회개, 믿음, 소망, 사랑, 십자가, 부활, 성찬, "
    "성도, 집사, 권사, 장로, 목사, 전도사, 예배, 찬양, 기도, 헌금, 간증, "
    # A/B 검증에서 Whisper가 실제로 틀렸던 단어들을 보강했다
    # (법궤→법괴, 블레셋→블랙셋, 시내산→신해산 오인식 발생)
    "법궤, 언약궤, 블레셋, 시내산, 미스바, 벧세메스, 실로, 가나안, "
    "이스라엘, 유다, 예루살렘, 갈릴리, 베들레헴, 나사렛, 애굽, 바벨론, "
    "십계명, 지성소, 제사장, 선지자, 사사, 장막, 성막, 안식일, 유월절."
)

# 위 프롬프트로도 남는 오인식을 확정 치환한다.
# 설교 맥락에서 다른 뜻으로 쓰일 여지가 없는 것만 넣는다.
TERM_FIXES = [
    (r"법괴", "법궤"),
    (r"법계(?=[를가는와의에])", "법궤"),
    (r"블랙셋", "블레셋"),
    (r"블랙세스", "블레셋"),
    (r"블레이셋", "블레셋"),
    (r"신해산", "시내산"),
    (r"애급", "애굽"),
    # '여하'는 '여하튼/여하한' 같은 일반어와 겹치므로 조사가 붙은 형태만 고친다
    (r"여하(?=께서|의|를|는|에게)", "여호와"),
]

# Whisper가 무음·찬양 구간에서 흔히 지어내는 문구들
HALLUCINATION_PATTERNS = [
    r"^시청해\s*주셔서\s*감사합니다\.?$",
    r"^구독과\s*좋아요",
    r"^감사합니다\.?$",
    r"^다음\s*영상에서\s*만나요",
    r"^한글\s*자막\s*by",
    r"^자막\s*제공",
    r"^MBC\s*뉴스",
    r"^\s*$",
]

_MODEL = None


def get_model():
    """Whisper 모델 싱글턴. 96편 배치에서 매번 로드하지 않도록 캐시한다."""
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel

        print(f"        Whisper 모델 로딩: {MODEL_SIZE} / {COMPUTE_TYPE} (cuda)")
        _MODEL = WhisperModel(MODEL_SIZE, device="cuda", compute_type=COMPUTE_TYPE)
    return _MODEL


def _find_js_runtime():
    """
    yt-dlp는 유튜브 서명 복호화에 JavaScript 런타임이 필요하다.
    없으면 일부 포맷을 못 받고 HTTP 403으로 실패한다. (실제로 겪음)
    """
    from shutil import which
    for name in ("deno", "node", "bun"):
        if which(name):
            return name
    return None


_JS_RUNTIME = _find_js_runtime()


def download_audio(video_id, attempts=3):
    """
    yt-dlp로 오디오만 받는다.

    ffmpeg 후처리를 일부러 쓰지 않는다. faster-whisper가 PyAV로 직접 디코딩하므로
    원본 m4a/webm 그대로 넘기면 되고, ffmpeg 설치 의존성을 없앨 수 있다.
    """
    AUDIO_DIR.mkdir(exist_ok=True)

    existing = list(AUDIO_DIR.glob(f"{video_id}.*"))
    if existing:
        return existing[0]

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio/best",
        "--no-playlist",
        "-o", str(AUDIO_DIR / f"{video_id}.%(ext)s"),
    ]
    if _JS_RUNTIME:
        cmd += ["--js-runtimes", _JS_RUNTIME]
    cmd.append(f"https://www.youtube.com/watch?v={video_id}")

    last_err = ""
    for i in range(attempts):
        result = subprocess.run(
            cmd, capture_output=True, encoding="utf-8", errors="replace"
        )
        files = list(AUDIO_DIR.glob(f"{video_id}.*"))
        if files:
            return files[0]
        last_err = (result.stderr or "")[-400:]
        if i < attempts - 1:
            # 일시적인 429/403은 잠깐 쉬면 풀리는 경우가 많다
            print(f"        다운로드 재시도 {i + 2}/{attempts} ...", flush=True)
            time.sleep(10 * (i + 1))

    raise RuntimeError(f"오디오 다운로드 실패: {video_id}\n{last_err}")


def _clean_segments(segments):
    """
    환각 제거.

    Whisper는 무음·찬양 구간에서 (1) 정형화된 지어낸 문구를 뱉거나
    (2) 같은 문장을 수십 번 반복한다. 둘 다 걸러낸다.
    """
    cleaned = []
    prev = None
    repeat = 0

    for text in segments:
        text = text.strip()
        if not text:
            continue

        if any(re.match(p, text) for p in HALLUCINATION_PATTERNS):
            continue

        # 동일 문장 3회 초과 반복은 환각으로 간주
        if text == prev:
            repeat += 1
            if repeat >= 2:
                continue
        else:
            repeat = 0
        prev = text

        cleaned.append(text)

    return cleaned


def get_cached(video_id):
    """이미 전사해둔 텍스트가 있으면 반환, 없으면 None."""
    path = TRANSCRIPT_DIR / f"{video_id}.txt"
    if path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8")
    return None


def transcribe(video_id, keep_audio=False, use_cache=True):
    """영상 ID를 받아 한국어 전사 텍스트를 반환한다."""
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    cache_path = TRANSCRIPT_DIR / f"{video_id}.txt"

    if use_cache and cache_path.exists():
        print("        전사 캐시 사용")
        return cache_path.read_text(encoding="utf-8")

    audio_path = download_audio(video_id)

    model = get_model()
    segments, info = model.transcribe(
        str(audio_path),
        language="ko",
        beam_size=5,
        initial_prompt=INITIAL_PROMPT,
        # 무음 구간을 잘라내 환각을 억제하고 처리 시간도 줄인다
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        # 이전 문맥을 물리면 반복 환각이 눈덩이처럼 커진다
        condition_on_previous_text=False,
    )

    texts = []
    duration = info.duration
    last_report = 0
    for seg in segments:
        texts.append(seg.text)
        # 스트리밍 제너레이터라 진행률을 여기서 찍는다
        pct = int(seg.end / duration * 100) if duration else 0
        if pct >= last_report + 10:
            last_report = pct - (pct % 10)
            print(f"        전사 {last_report}% ...", flush=True)

    cleaned = _clean_segments(texts)
    result = " ".join(cleaned)

    for pattern, correct in TERM_FIXES:
        result = re.sub(pattern, correct, result)

    cache_path.write_text(result, encoding="utf-8")

    if not keep_audio:
        try:
            audio_path.unlink()
        except OSError:
            pass

    return result


if __name__ == "__main__":
    vid = sys.argv[1]
    text = transcribe(vid, keep_audio="--keep" in sys.argv)
    print(f"\n{'='*50}")
    print(f"전사 완료: {len(text):,}자")
    print(f"{'='*50}")
    print(text[:2000])
