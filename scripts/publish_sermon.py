"""
생명샘명성교회 설교 자동 게시 스크립트
======================================
YouTube 채널에서 새 설교 영상을 감지하고,
자막을 추출한 뒤 AI로 정리하여 블로그 포스트를 생성합니다.

사용법:
  python scripts/publish_sermon.py              # 새 영상만 처리
  python scripts/publish_sermon.py --all        # 최근 5개 전부 처리
  python scripts/publish_sermon.py --video ID   # 특정 영상 처리

필요 패키지:
  pip install yt-dlp youtube-transcript-api anthropic
"""

import json
import os
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


def extract_transcript(video_id):
    """YouTube 자막 추출"""
    from youtube_transcript_api import YouTubeTranscriptApi
    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(video_id, languages=["ko"])
    return " ".join(s.text for s in transcript.snippets)


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


def summarize_with_ai(transcript, title, scripture, date_str):
    """Claude API로 설교 내용 정리"""
    import anthropic
    client = anthropic.Anthropic()

    prompt = f"""당신은 교회 설교를 정리하는 전문 편집자입니다.
아래는 유튜브 설교 영상에서 추출한 자막 텍스트입니다. 아래 JSON 형식으로 정리해 주세요.

## 정리 규칙
1. 구어체/반복/추임새 제거, 읽기 좋은 문어체로 정리
2. 핵심 메시지 3~5개를 간결하게
3. 설교 요약을 소제목이 있는 6개 이내 단락으로 구성
4. 말씀 인용 시 성경 구절을 정확히 표기
5. 설교의 비유와 예화를 살려서 정리
6. 원문의 30~40% 수준으로 압축
7. 묵상 포인트 2~3개 (질문 형태)
8. 인용된 성경 구절 목록 정리

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
  "scriptures": ["구절1 — 설명", "구절2 — 설명", ...],
  "reflections": ["묵상 질문1", "묵상 질문2", ...]
}}

## 자막 텍스트
{transcript}
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    # JSON 부분만 추출
    m = re.search(r'\{[\s\S]+\}', text)
    if m:
        return json.loads(m.group())
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

    scriptures_html = "\n".join(
        f"          <li>{s}</li>" for s in ai_data["scriptures"]
    )

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


def update_sermons_json(slug, sermon_title, scripture, date_str, video_id, summary_short):
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
        "publishedAt": published_at
    }

    data.insert(0, new_entry)
    # 날짜순 정렬 (최신 먼저)
    data.sort(key=lambda x: x.get("publishedAt", ""), reverse=True)

    SERMONS_JSON.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def process_video(video_id, title):
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
    transcript = extract_transcript(video_id)
    print(f"        → {len(transcript):,}자 추출 완료")

    # 2. AI 정리
    print("  [2/4] AI 정리 중...")
    ai_data = summarize_with_ai(transcript, title, scripture, date_str)
    print("        → 정리 완료")

    # 3. HTML 생성
    print("  [3/4] 블로그 포스트 생성 중...")
    html = generate_blog_html(slug, sermon_title, scripture, date_str, video_id, ai_data)
    post_path = BLOG_DIR / f"{slug}.html"
    post_path.write_text(html, encoding="utf-8")
    print(f"        → {post_path.name} 저장")

    # 4. JSON 업데이트
    print("  [4/4] 목록 업데이트 중...")
    update_sermons_json(slug, sermon_title, scripture, date_str, video_id, ai_data["summary_short"])
    print("        → sermons.json 업데이트 완료")

    print(f"\n  ✓ 게시 완료: blog/{slug}.html")
    return slug


def main():
    args = sys.argv[1:]

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
        process_video(target["id"], target["title"])
        return

    print("생명샘명성교회 설교 자동 게시")
    print("=" * 40)
    print("YouTube 채널에서 최근 영상을 확인합니다...\n")

    videos = get_recent_videos(5)
    published = get_published_ids()

    process_all = "--all" in args

    new_videos = [v for v in videos if v["id"] not in published] if not process_all else videos

    if not new_videos:
        print("새로운 설교 영상이 없습니다.")
        return

    print(f"처리할 영상: {len(new_videos)}개")
    for v in new_videos:
        print(f"  - {v['title']}")

    for v in new_videos:
        try:
            process_video(v["id"], v["title"])
        except Exception as e:
            print(f"\n  ✗ 오류: {v['title']}")
            print(f"    {e}")

    print(f"\n{'='*40}")
    print("모든 처리가 완료되었습니다.")
    print(f"웹사이트를 배포하면 블로그에 반영됩니다.")


if __name__ == "__main__":
    main()
