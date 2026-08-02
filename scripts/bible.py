"""
성경 본문 조회 — 개역개정(GAE)

대한성서공회 성경읽기 페이지에서 장 단위로 받아 캐시하고, 절 범위를 뽑아 쓴다.
AI에게 구절을 외워 쓰게 하면 조사가 어긋난 오인용이 나오므로
(실측: "함께 하시매"를 "함께 하심에"로 씀) 본문은 반드시 여기서 가져온다.

  from bible import lookup
  lookup("사도행전 11:21")
  → [("사도행전 11:21", "주의 손이 그들과 함께 하시매 ...")]
"""

import html as _html
import json
import re
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "bible_cache"
READ_URL = "https://www.bskorea.or.kr/bible/korbibReadpage.php?version=GAE&book={code}&chap={chap}"

# 정경 순서(창세기~요한계시록). 대한성서공회 URL은 소문자 3자 코드를 쓴다.
# 요나만 jon이 아니라 jnh다 (실측 확인).
CODES = [
    "gen", "exo", "lev", "num", "deu", "jos", "jdg", "rut", "1sa", "2sa",
    "1ki", "2ki", "1ch", "2ch", "ezr", "neh", "est", "job", "psa", "pro",
    "ecc", "sng", "isa", "jer", "lam", "ezk", "dan", "hos", "jol", "amo",
    "oba", "jnh", "mic", "nam", "hab", "zep", "hag", "zec", "mal",
    "mat", "mrk", "luk", "jhn", "act", "rom", "1co", "2co", "gal", "eph",
    "php", "col", "1th", "2th", "1ti", "2ti", "tit", "phm", "heb", "jas",
    "1pe", "2pe", "1jn", "2jn", "3jn", "jud", "rev",
]

# 한글 책이름·약칭 → 코드. 설교 제목의 본문 표기가 제각각이라 둘 다 받는다.
NAMES = [
    ("창세기", "창"), ("출애굽기", "출"), ("레위기", "레"), ("민수기", "민"),
    ("신명기", "신"), ("여호수아", "수"), ("사사기", "삿"), ("룻기", "룻"),
    ("사무엘상", "삼상"), ("사무엘하", "삼하"), ("열왕기상", "왕상"), ("열왕기하", "왕하"),
    ("역대상", "대상"), ("역대하", "대하"), ("에스라", "스"), ("느헤미야", "느"),
    ("에스더", "에"), ("욥기", "욥"), ("시편", "시"), ("잠언", "잠"),
    ("전도서", "전"), ("아가", "아"), ("이사야", "사"), ("예레미야", "렘"),
    ("예레미야애가", "애"), ("에스겔", "겔"), ("다니엘", "단"), ("호세아", "호"),
    ("요엘", "욜"), ("아모스", "암"), ("오바댜", "옵"), ("요나", "욘"),
    ("미가", "미"), ("나훔", "나"), ("하박국", "합"), ("스바냐", "습"),
    ("학개", "학"), ("스가랴", "슥"), ("말라기", "말"),
    ("마태복음", "마"), ("마가복음", "막"), ("누가복음", "눅"), ("요한복음", "요"),
    ("사도행전", "행"), ("로마서", "롬"), ("고린도전서", "고전"), ("고린도후서", "고후"),
    ("갈라디아서", "갈"), ("에베소서", "엡"), ("빌립보서", "빌"), ("골로새서", "골"),
    ("데살로니가전서", "살전"), ("데살로니가후서", "살후"),
    ("디모데전서", "딤전"), ("디모데후서", "딤후"), ("디도서", "딛"), ("빌레몬서", "몬"),
    ("히브리서", "히"), ("야고보서", "약"), ("베드로전서", "벧전"), ("베드로후서", "벧후"),
    ("요한1서", "요일"), ("요한2서", "요이"), ("요한3서", "요삼"),
    ("유다서", "유"), ("요한계시록", "계"),
]

BOOK_MAP = {}
for _i, (_full, _abbr) in enumerate(NAMES):
    BOOK_MAP[_full] = CODES[_i]
    BOOK_MAP[_abbr] = CODES[_i]
# 설교 제목에서 흔히 쓰는 변형
BOOK_MAP.update({
    "요한일서": "1jn", "요한이서": "2jn", "요한삼서": "3jn",
    "아가서": "sng", "애가": "lam",
})

# 긴 이름이 짧은 이름에 먹히지 않도록 길이 내림차순으로 매칭한다
# (예: '예레미야애가'가 '예레미야'로 잘리면 안 된다)
_BOOK_RE = "|".join(sorted((re.escape(k) for k in BOOK_MAP), key=len, reverse=True))
REF_RE = re.compile(rf"({_BOOK_RE})\s*(\d+)\s*[:장]\s*([\d\s,~\-–]+)")


def _fetch_chapter(code, chap):
    """한 장을 받아 {절번호: 본문} 으로 파싱한다. 디스크에 캐시한다."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"{code}_{chap}.json"
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}

    req = urllib.request.Request(
        READ_URL.format(code=code, chap=chap),
        headers={"User-Agent": "Mozilla/5.0 (sermon-archive)"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        page = r.read().decode("utf-8", "replace")

    # 각주는 본문이 아니다: 마커(<a class=comment>)와 숨김 팝업(<div class=D2>)을 먼저 제거
    page = re.sub(r"<div[^>]*class=D2[^>]*>.*?</div>", "", page, flags=re.S)
    page = re.sub(r"<a class=comment.*?</a>", "", page, flags=re.S)

    verses = {}
    # 절은 보통 </span><br /> 로 끝나지만, 장의 마지막 절만은 <br /> 없이
    # </div> 로 닫힌다. 이걸 빼면 마지막 절이 페이지 끝까지(검색 UI·자바스크립트까지)
    # 삼켜서 3000자짜리 "절"이 만들어진다. 각주 div는 위에서 이미 제거했으므로
    # </div>를 종료 조건으로 써도 본문이 잘리지 않는다.
    for m in re.finditer(
        r'<span class="number">(\d+)(?:&nbsp;|\s)*</span>(.*?)'
        r'(?=<span><span class="number">|<br\s*/?>|</div>|\Z)',
        page, re.S,
    ):
        text = re.sub(r"<[^>]+>", "", m.group(2))
        text = _html.unescape(text).replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            verses.setdefault(int(m.group(1)), text)

    if verses:
        tmp = cache.with_suffix(".json.tmp")
        payload = json.dumps(verses, ensure_ascii=False)
        tmp.write_text(payload, encoding="utf-8")
        if tmp.read_text(encoding="utf-8") == payload:
            tmp.replace(cache)
        else:
            tmp.unlink(missing_ok=True)
        time.sleep(0.4)   # 상대 서버 배려
    return verses


def _expand(spec):
    """'19-26', '11-12, 27', '21' → [19,20,...,26] 형태로 편다."""
    out = []
    for part in re.split(r"[,\s]+", spec.strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*[~\-–]\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b >= a and b - a < 60:
                out.extend(range(a, b + 1))
        elif part.isdigit():
            out.append(int(part))
    return out


def lookup(reference, max_verses=12):
    """
    '사도행전 11:19-26' 같은 표기를 [(표기, 본문), ...] 로 돌려준다.
    여러 구절이 쉼표로 이어진 표기도 처리한다. 실패하면 빈 리스트.
    """
    results = []
    for m in REF_RE.finditer(reference or ""):
        book, chap, spec = m.group(1), int(m.group(2)), m.group(3)
        code = BOOK_MAP.get(book)
        if not code:
            continue
        try:
            chapter = _fetch_chapter(code, chap)
        except Exception:
            continue
        nums = _expand(spec)[:max_verses]
        text = " ".join(chapter[n] for n in nums if n in chapter)
        if text:
            label = f"{book} {chap}:{spec.strip().rstrip(',')}"
            results.append((label, text))
    return results


if __name__ == "__main__":
    import sys
    for ref in (sys.argv[1:] or ["사도행전 11:21", "시편 84:11-12, 누가복음 18:27"]):
        for label, text in lookup(ref):
            print(f"[{label}]\n  {text}\n")
