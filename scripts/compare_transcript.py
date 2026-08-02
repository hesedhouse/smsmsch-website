"""
A/B 검증 — 같은 영상을 유튜브 자동자막 / 로컬 Whisper 두 방식으로 전사해 비교한다.

이미 게시된 설교(=유튜브 자막이 있는 영상)에 대고 돌려서
로컬 STT로 바꿔도 품질이 떨어지지 않는지 확인하는 용도.

  python scripts/compare_transcript.py ZlMfoOr4Weg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE_DIR = Path(__file__).resolve().parent.parent

# 설교에서 자주 틀리는 고유명사. 양쪽이 몇 번씩 맞췄는지 세어본다.
CHECK_TERMS = [
    "에벤에셀", "임마누엘", "여호와", "사무엘", "이사야", "할렐루야",
    "아브라함", "그리스도", "하나님", "성령", "십자가", "아멘",
]


def yt_caption(video_id):
    from youtube_transcript_api import YouTubeTranscriptApi
    ytt = YouTubeTranscriptApi()
    tr = ytt.fetch(video_id, languages=["ko"])
    return " ".join(s.text for s in tr.snippets)


def main():
    video_id = sys.argv[1]

    print("유튜브 자동자막 가져오는 중...")
    try:
        yt = yt_caption(video_id)
    except Exception as e:
        print(f"  자막 없음: {e}")
        yt = ""

    print("로컬 Whisper 전사 중...")
    from transcribe_local import transcribe
    wh = transcribe(video_id, keep_audio=True)

    (BASE_DIR / "_ab_youtube.txt").write_text(yt, encoding="utf-8")
    (BASE_DIR / "_ab_whisper.txt").write_text(wh, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"{'항목':<22}{'유튜브 자막':>16}{'로컬 Whisper':>18}")
    print(f"{'-'*60}")
    print(f"{'총 글자수':<22}{len(yt):>16,}{len(wh):>18,}")
    print(f"{'마침표 개수':<22}{yt.count('.'):>16,}{wh.count('.'):>18,}")
    print(f"{'쉼표 개수':<22}{yt.count(','):>16,}{wh.count(','):>18,}")
    print(f"{'물음표 개수':<22}{yt.count('?'):>16,}{wh.count('?'):>18,}")
    print(f"{'-'*60}")
    print("고유명사 인식 횟수")
    for t in CHECK_TERMS:
        print(f"  {t:<20}{yt.count(t):>16,}{wh.count(t):>18,}")
    print(f"{'='*60}")

    print("\n--- 유튜브 자막 앞부분 ---")
    print(yt[:700] if yt else "(없음)")
    print("\n--- 로컬 Whisper 앞부분 ---")
    print(wh[:700])
    print(f"\n전문 저장: _ab_youtube.txt / _ab_whisper.txt")


if __name__ == "__main__":
    main()
