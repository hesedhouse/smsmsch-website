"""
생명샘명성교회 설교 자동 게시 스크립트
======================================
YouTube 채널에서 새 설교 영상을 감지하고,
자막을 추출한 뒤 AI로 정리하여 블로그 포스트를 생성합니다.

사용법:
  python scripts/publish_sermon.py              # 새 영상만 처리
  python scripts/publish_sermon.py --all        # 최근 5개 전부 처리
  python scripts/publish_sermon.py --video ID   # 특정 영상 처리
  python scripts/publish_sermon.py --backfill --stt   # 채널 전체 + 자막없으면 로컬 STT

자막이 없는 영상은 --stt를 줘야 로컬 Whisper로 음성인식한다.
GitHub Actions에는 GPU가 없으므로 주간 자동 게시는 --stt 없이 돌린다.

필요 패키지:
  pip install yt-dlp youtube-transcript-api anthropic
  pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12   # --stt 사용 시
"""

import atexit
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ===== 설정 =====
SITE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SITE_DIR / "data"
BLOG_DIR = SITE_DIR / "blog"
SERMONS_JSON = DATA_DIR / "sermons.json"

YOUTUBE_CHANNEL_URL = "https://www.youtube.com/@%EC%83%9D%EB%AA%85%EC%83%98%EB%AA%85%EC%84%B1%EA%B5%90%ED%9A%8C%EA%B6%8C%EC%98%81%EC%B2%A0/videos"


def atomic_write(path, text):
    """
    임시 파일에 쓰고 검증한 뒤 교체한다.

    작업 폴더가 NAS(RaiDrive) 위에 있어서, write_text처럼 원본을 먼저 0으로 자르고
    쓰는 방식은 쓰기가 유실되면 빈 파일만 남는다. 실제로 sermons.json이 이렇게 날아갔다.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    if tmp.read_text(encoding="utf-8") != text:
        tmp.unlink(missing_ok=True)
        raise IOError(f"쓰기 검증 실패: {path.name}")
    os.replace(tmp, path)


def load_env():
    """
    Z 드라이브의 공용 .env에서 API 키를 읽어 환경변수에 채운다.

    헤세드는 API 키를 `Z:\\hesedcorp\\.env` 한 곳에 모아두므로,
    거기 넣어두면 회사 PC·집 PC 양쪽에서 그대로 동작한다.
    이미 환경변수로 설정돼 있으면 그쪽을 우선한다.
    GitHub Actions에는 이 파일이 없으므로 조용히 넘어간다.
    """
    for candidate in (SITE_DIR / ".env", SITE_DIR.parent.parent / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if value and not os.environ.get(key):
                os.environ[key] = value


load_env()


def get_recent_videos(limit=5):
    """yt-dlp로 최근 영상 목록 가져오기"""
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--flat-playlist", "--dump-json",
         "--playlist-end", str(limit), YOUTUBE_CHANNEL_URL],
        capture_output=True, encoding="utf-8"
    )
    videos = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            d = json.loads(line)
            videos.append({"id": d["id"], "title": d["title"]})
    return videos


def get_published_ids():
    """이미 게시된 영상 ID 목록"""
    if not SERMONS_JSON.exists():
        return set()
    data = json.loads(SERMONS_JSON.read_text(encoding="utf-8"))
    return {s["videoId"] for s in data}


def extract_transcript(video_id, allow_stt=False):
    """
    설교 텍스트 확보. (텍스트, 출처) 를 반환한다.

    1순위는 YouTube 자동자막. 자막이 없는 영상은 allow_stt일 때만
    로컬 Whisper 음성인식으로 넘긴다.
    """
    # 전사본이 이미 있으면 자막 API를 건드리지 않는다.
    # 백필 96편처럼 자막이 없는 게 확실한 영상에 실패할 요청을 반복하면
    # 시간도 버리고 IP가 속도 제한에 걸릴 수 있다.
    if allow_stt:
        from transcribe_local import get_cached
        cached = get_cached(video_id)
        if cached:
            print("        전사 캐시 사용")
            return cached, "whisper"

    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        ytt = YouTubeTranscriptApi()
        transcript = ytt.fetch(video_id, languages=["ko"])
        return " ".join(s.text for s in transcript.snippets), "youtube"
    except Exception as e:
        if not allow_stt:
            raise
        print(f"        자막 없음 ({type(e).__name__}) → 로컬 STT 전환")
        from transcribe_local import transcribe
        return transcribe(video_id), "whisper"


def is_sermon_video(title):
    """
    게시 대상인 설교 영상인지 판별한다.

    이 채널의 설교 영상은 제목이 항상
      생명샘명성교회 주일예배 / "설교제목" [성경본문] / 2026년 7월 26일
    형식이다. 따옴표 설교제목이 없는 건 수요기도회·공지 같은 예외로,
    제목·본문이 비어 목록에서 어색하게 보이므로 제외한다.
    """
    return bool(re.search(r'"(.+?)"', title))


def parse_title(title):
    """영상 제목에서 설교 제목, 성경구절, 날짜 추출"""
    # 패턴: 생명샘명성교회 주일예배 / "설교제목" [성경구절] / 2026년 7월 26일
    sermon_title = title
    scripture = ""
    date_str = ""

    m = re.search(r'"(.+?)"', title)
    if m:
        sermon_title = m.group(1)

    m = re.search(r'\[(.+?)\]', title)
    if m:
        scripture = m.group(1)

    m = re.search(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', title)
    if m:
        date_str = f"{m.group(1)}년 {m.group(2)}월 {m.group(3)}일"

    return sermon_title, scripture, date_str


def make_slug(date_str, sermon_title):
    """블로그 포스트 slug 생성"""
    m = re.search(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', date_str)
    if m:
        date_part = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    else:
        date_part = datetime.now().strftime("%Y-%m-%d")
    # 제목에서 slug 생성 (한글 유지)
    clean = re.sub(r'[^\w가-힣]', '', sermon_title)
    return f"{date_part}-{clean[:20]}"


def summarize_with_ai(transcript, title, scripture, date_str, attempts=3):
    """
    Claude API로 설교 내용 정리.

    응답 JSON이 깨져 있는 경우가 드물게 있다(본문에 따옴표가 섞이는 등).
    다시 물으면 대개 정상으로 나오므로 파싱 실패 시 재요청한다.
    """
    last_err = None
    for i in range(attempts):
        try:
            return _summarize_once(transcript, title, scripture, date_str)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            if i < attempts - 1:
                print(f"        JSON 파싱 실패 → 재요청 {i + 2}/{attempts}", flush=True)
    raise last_err


def _summarize_once(transcript, title, scripture, date_str):
    import anthropic
    client = anthropic.Anthropic()

    prompt = f"""당신은 교회 설교를 정리하는 전문 편집자입니다.
아래는 유튜브 설교 영상에서 추출한 자막 텍스트입니다. 아래 JSON 형식으로 정리해 주세요.

## 정리 규칙
1. 구어체/반복/추임새 제거, 읽기 좋은 문어체로 정리
2. 핵심 메시지 3~5개를 간결하게
3. 설교 요약을 소제목이 있는 6개 이내 단락으로 구성
4. **성경 본문을 직접 옮겨 쓰지 말 것.** 기억에 의존한 인용은 조사가 어긋나 오인용이 된다.
   말씀을 인용해야 할 자리에는 장·절 표기만 쓰고(예: "사도행전 11:21"),
   본문은 시스템이 개역개정에서 가져와 넣는다.
5. 설교의 비유와 예화를 살려서 정리
6. 원문의 30~40% 수준으로 압축
7. 묵상 포인트 2~3개 (질문 형태)
8. 인용된 성경 구절 목록 정리
9. 자막은 음성인식으로 만들어져 고유명사가 틀리게 적혀 있을 수 있다.
   인명·지명·성경 용어(예: 법괴→법궤, 블랙셋→블레셋, 신해산→시내산, 여하→여호와)는
   문맥에 맞는 올바른 표기로 바로잡아 쓸 것.
10. 자막이 불분명해 확신이 서지 않는 내용은 지어내지 말고 생략할 것.

## 영상 정보
- 제목: {title}
- 본문: {scripture}
- 날짜: {date_str}

## 출력 형식 (JSON)
{{
  "summary_short": "1~2문장 요약 (블로그 카드용)",
  "key_messages": ["핵심1", "핵심2", ...],
  "sections": [
    {{"heading": "소제목", "content": "본문 (HTML 태그 가능: <p>, <blockquote>, <cite>, <strong>)"}},
    ...
  ],
  "scriptures": [
    {{"ref": "사도행전 11:21", "note": "이 구절이 설교에서 갖는 의미 한 문장"}},
    ...
  ],
  "reflections": ["묵상 질문1", "묵상 질문2", ...]
}}

## 자막 텍스트
{transcript}
"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    # JSON 부분만 추출
    m = re.search(r'\{[\s\S]+\}', text)
    if m:
        raw = m.group()
        # 제어 문자 제거 (줄바꿈/탭 제외)
        raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # strict=False로 재시도
            return json.loads(raw, strict=False)
    raise ValueError("AI 응답에서 JSON을 찾을 수 없습니다")


def generate_blog_html(slug, sermon_title, scripture, date_str, video_id, ai_data):
    """블로그 포스트 HTML 생성"""
    key_messages_html = "\n".join(
        f"          <li>{msg}</li>" for msg in ai_data["key_messages"]
    )

    sections_html = ""
    for sec in ai_data["sections"]:
        sections_html += f'\n        <h3>{sec["heading"]}</h3>\n'
        content = sec["content"]
        # content가 이미 HTML이 아니면 <p>로 감싸기
        if not content.strip().startswith("<"):
            content = f"<p>{content}</p>"
        sections_html += f"        {content}\n"

    # 구절 본문은 AI가 아니라 개역개정에서 가져와 넣는다.
    # 조회에 실패하면 본문 없이 표기와 설명만 남긴다(지어내지 않는다).
    from bible import lookup

    items = []
    for s in ai_data["scriptures"]:
        if isinstance(s, dict):
            ref, note = s.get("ref", "").strip(), s.get("note", "").strip()
        else:                                   # 구버전 "구절 — 설명" 문자열
            ref, _, note = str(s).partition("—")
            ref, note = ref.strip(), note.strip()

        found = lookup(ref)
        body = " ".join(t for _, t in found)
        block = f'          <li>\n            <strong>{ref}</strong>\n'
        if body:
            block += f'            <blockquote class="verse">{body}</blockquote>\n'
        if note:
            block += f'            <span class="verse-note">{note}</span>\n'
        block += "          </li>"
        items.append(block)

    scriptures_html = "\n".join(items)

    reflections_html = "\n".join(
        f"          <li>{r}</li>" for r in ai_data["reflections"]
    )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{sermon_title} - 생명샘명성교회</title>
  <meta name="description" content="{date_str} 주일설교 - {scripture}">
  <link rel="icon" href="../images/logo.jpg" type="image/jpeg">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="../css/style.css">
  <link rel="stylesheet" href="../css/blog-post.css">
</head>
<body>
  <nav class="navbar scrolled">
    <div class="nav-container">
      <a href="../" class="nav-logo" style="color:var(--primary-dark)">
        <img src="../images/logo.jpg" alt="생명샘명성교회 로고">
        <span>생명샘명성교회</span>
      </a>
      <button class="nav-toggle" id="navToggle" aria-label="메뉴 열기">
        <span></span><span></span><span></span>
      </button>
      <ul class="nav-menu" id="navMenu">
        <li><a href="../">홈</a></li>
        <li><a href="../#about">교회소개</a></li>
        <li><a href="../#worship">예배안내</a></li>
        <li><a href="../#sermons">설교영상</a></li>
        <li><a href="../#blog">설교말씀</a></li>
        <li><a href="../#location">오시는 길</a></li>
      </ul>
    </div>
  </nav>

  <main class="post">
    <article class="post-article">
      <header class="post-header">
        <a href="../#blog" class="post-back">&larr; 설교말씀 목록</a>
        <p class="post-date">{date_str} 주일예배</p>
        <h1>{sermon_title}</h1>
        <div class="post-meta">
          <span class="post-scripture">{scripture}</span>
          <span class="post-preacher">권영철 목사</span>
        </div>
      </header>

      <section class="post-key-messages">
        <h2>핵심 메시지</h2>
        <ol>
{key_messages_html}
        </ol>
      </section>

      <section class="post-body">
        <h2>설교 요약</h2>
{sections_html}
      </section>

      <section class="post-scriptures">
        <h2>인용 말씀</h2>
        <ul>
{scriptures_html}
        </ul>
      </section>

      <section class="post-reflection">
        <h2>묵상 포인트</h2>
        <ol>
{reflections_html}
        </ol>
      </section>

      <div class="post-video">
        <a href="https://www.youtube.com/watch?v={video_id}" target="_blank" rel="noopener">
          <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
            <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814z"/>
            <polygon points="9.545,15.568 15.818,12 9.545,8.432" fill="#fff"/>
          </svg>
          설교 영상 보기
        </a>
      </div>
    </article>
  </main>

  <footer class="footer">
    <div class="container">
      <div class="footer-content">
        <div class="footer-logo">
          <img src="../images/logo.jpg" alt="생명샘명성교회">
          <span>생명샘명성교회</span>
        </div>
        <p class="footer-address">경기도 고양시 일산동구 일산로 447 신풍빌딩 4, 5층</p>
        <p class="footer-copyright">&copy; 2026 생명샘명성교회. All rights reserved.</p>
      </div>
    </div>
  </footer>

  <script>
    document.getElementById('navToggle').addEventListener('click', () => {{
      document.getElementById('navMenu').classList.toggle('active');
    }});
  </script>
</body>
</html>"""

    return html


def update_sermons_json(slug, sermon_title, scripture, date_str, video_id, summary_short, source="youtube"):
    """sermons.json에 새 항목 추가 (최신순 정렬)"""
    data = []
    if SERMONS_JSON.exists():
        data = json.loads(SERMONS_JSON.read_text(encoding="utf-8"))

    # 날짜에서 ISO 형식 추출
    m = re.search(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', date_str)
    published_at = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else datetime.now().strftime("%Y-%m-%d")

    new_entry = {
        "slug": slug,
        "date": date_str,
        "title": sermon_title,
        "scripture": scripture,
        "summary": summary_short,
        "videoId": video_id,
        "publishedAt": published_at,
        "source": source
    }

    data.insert(0, new_entry)
    # 날짜순 정렬 (최신 먼저)
    data.sort(key=lambda x: x.get("publishedAt", ""), reverse=True)

    atomic_write(SERMONS_JSON, json.dumps(data, ensure_ascii=False, indent=2))


def process_video(video_id, title, allow_stt=False):
    """영상 하나를 처리하는 메인 파이프라인"""
    sermon_title, scripture, date_str = parse_title(title)
    slug = make_slug(date_str, sermon_title)

    print(f"\n{'='*50}")
    print(f"  처리 중: {sermon_title}")
    print(f"  본문: {scripture}")
    print(f"  날짜: {date_str}")
    print(f"{'='*50}")

    # 1. 자막 추출
    print("  [1/4] 자막 추출 중...")
    transcript, source = extract_transcript(video_id, allow_stt=allow_stt)
    label = "유튜브 자막" if source == "youtube" else "로컬 STT"
    print(f"        → {len(transcript):,}자 추출 완료 ({label})")

    # 2. AI 정리
    print("  [2/4] AI 정리 중...")
    ai_data = summarize_with_ai(transcript, title, scripture, date_str)
    print("        → 정리 완료")

    # 3. HTML 생성
    print("  [3/4] 블로그 포스트 생성 중...")
    html = generate_blog_html(slug, sermon_title, scripture, date_str, video_id, ai_data)
    post_path = BLOG_DIR / f"{slug}.html"
    atomic_write(post_path, html)
    print(f"        → {post_path.name} 저장")

    # 4. JSON 업데이트
    print("  [4/4] 목록 업데이트 중...")
    update_sermons_json(slug, sermon_title, scripture, date_str, video_id, ai_data["summary_short"], source)
    print("        → sermons.json 업데이트 완료")

    print(f"\n  ✓ 게시 완료: blog/{slug}.html")
    return slug


def _pid_alive(pid):
    """해당 PID가 아직 살아있는지 확인한다."""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, encoding="utf-8", errors="replace",
            ).stdout
            return str(pid) in out
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def acquire_lock():
    """
    배치 중복 실행 방지.

    셸 래퍼만 종료되고 파이썬 자식 프로세스가 살아남는 경우가 있는데,
    그 상태에서 배치를 다시 걸면 두 프로세스가 같은 audio_cache를 두고
    충돌해 다운로드가 무더기로 실패한다. (실제로 한 번 겪음)
    """
    # 작업 폴더가 Z 드라이브(회사 PC·집 PC 공유)에 있으므로 잠금에 호스트명을 함께 적는다.
    # 다른 PC의 PID를 이 PC 기준으로 검사하면 엉뚱하게 차단하거나 통과시킨다.
    host = platform.node()
    lock = SITE_DIR / ".batch.lock"

    if lock.exists():
        raw = lock.read_text(encoding="utf-8").strip()
        old_host, _, old_pid = raw.partition("\t")
        if not old_pid:  # 호스트명 없던 구버전 형식
            old_host, old_pid = host, raw

        if old_host != host:
            print(f"  다른 PC({old_host})의 잠금 파일이 남아 있어 무시합니다.")
        elif _pid_alive(old_pid):
            print(f"✗ 이미 배치가 실행 중입니다 ({old_host} PID {old_pid}).")
            print(f"  중단하려면: taskkill /F /PID {old_pid}   (또는 kill {old_pid})")
            sys.exit(1)
        else:
            print(f"  이전 실행(PID {old_pid})의 잠금 파일을 정리합니다.")

    lock.write_text(f"{host}\t{os.getpid()}", encoding="utf-8")
    atexit.register(lambda: lock.unlink(missing_ok=True))


def main():
    args = sys.argv[1:]
    allow_stt = "--stt" in args
    acquire_lock()

    if "--video" in args:
        idx = args.index("--video")
        video_id = args[idx + 1]
        videos = get_recent_videos(15)
        target = next((v for v in videos if v["id"] == video_id), None)
        if not target:
            # 개별 영상 정보 가져오기
            result = subprocess.run(
                [sys.executable, "-m", "yt_dlp", "--dump-json", "--no-playlist",
                 f"https://www.youtube.com/watch?v={video_id}"],
                capture_output=True, encoding="utf-8"
            )
            d = json.loads(result.stdout)
            target = {"id": d["id"], "title": d["title"]}
        process_video(target["id"], target["title"], allow_stt=allow_stt)
        return

    print("생명샘명성교회 설교 자동 게시")
    print("=" * 40)
    print("YouTube 채널에서 최근 영상을 확인합니다...\n")

    backfill = "--backfill" in args
    process_all = "--all" in args
    limit = 500 if backfill else (200 if process_all else 5)
    videos = get_recent_videos(limit)
    published = get_published_ids()

    new_videos = [v for v in videos if v["id"] not in published]

    skipped = [v for v in new_videos if not is_sermon_video(v["title"])]
    if skipped:
        print(f"설교 형식이 아니어서 제외: {len(skipped)}개")
        for v in skipped:
            print(f"  - {v['title'][:60]}")
        new_videos = [v for v in new_videos if is_sermon_video(v["title"])]

    if not new_videos:
        print("새로운 설교 영상이 없습니다.")
        return

    print(f"처리할 영상: {len(new_videos)}개")
    for v in new_videos:
        print(f"  - {v['title']}")

    # 1단계: 전사만 먼저 돌린다. GPU로 96편 ≈ 10시간이고 API 키가 필요없는 구간이라,
    # 요약(2단계)과 분리해두면 요약이 실패해도 다시 전사하지 않는다.
    if "--transcribe-only" in args:
        from transcribe_local import transcribe
        ok, bad = 0, []
        for i, v in enumerate(new_videos, 1):
            print(f"\n[{i}/{len(new_videos)}] {v['title'][:70]}", flush=True)
            try:
                text = transcribe(v["id"])
                print(f"        → {len(text):,}자 전사 완료", flush=True)
                ok += 1
            except Exception as e:
                print(f"        ✗ {type(e).__name__}: {e}", flush=True)
                bad.append((v["id"], v["title"], f"{type(e).__name__}: {e}"))
        print(f"\n{'='*40}")
        print(f"전사 완료 {ok}개 / 실패 {len(bad)}개")
        for vid, title, err in bad:
            print(f"  ✗ {vid} {title[:50]} — {err}")
        return

    done, failed = 0, []
    for i, v in enumerate(new_videos, 1):
        print(f"\n[{i}/{len(new_videos)}]", end=" ")
        try:
            process_video(v["id"], v["title"], allow_stt=allow_stt)
            done += 1
        except Exception as e:
            print(f"\n  ✗ 오류: {v['title']}")
            print(f"    {type(e).__name__}: {e}")
            failed.append((v["id"], v["title"], f"{type(e).__name__}: {e}"))

    print(f"\n{'='*40}")
    print(f"완료 {done}개 / 실패 {len(failed)}개")
    if failed:
        # 중단 후 재실행하면 이미 게시된 건 건너뛰므로 이어서 처리된다
        log = SITE_DIR / "backfill_failed.log"
        log.write_text(
            "\n".join(f"{vid}\t{title}\t{err}" for vid, title, err in failed),
            encoding="utf-8"
        )
        print(f"실패 목록: {log.name}")
    print("웹사이트를 배포하면 블로그에 반영됩니다.")


if __name__ == "__main__":
    main()
