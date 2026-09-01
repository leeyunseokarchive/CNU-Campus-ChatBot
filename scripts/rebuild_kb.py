"""
KB Rebuild Script v3 — precise domain filter, specialized page parsers, clean BFS.

Key improvements over v2:
- Specialized extractor for plus.cnu.ac.kr academic calendar (.calen_box)
- BFS restricted to same URL path prefix (no wandering to main portal)
- Garbage title filtering ("충남대학교 로고", "충남대학교" only, etc.)
- Content-hash dedup (not URL-based)
- schedule calendar extracts full year with month/date structure
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "data" / "kb"

# ── Content Cleaner ──────────────────────────────────────────────────────────

_NAV_INLINE = re.compile(
    r"(ENG\s+통합검색\s+열기\s+버튼.*?닫기버튼"
    r"|통합검색\s+열기\s+버튼.*?닫기버튼"
    r"|사이트맵\s+CNU\s+미래\s+사회를\s+선도할\s+강한\s+대학[^\n]*"
    r"|THE\s+STRONG\s+CNU\s+CNU\s+홍보브로슈어[^\n]*"
    r"|SNS\s+및\s+프린트\s+URL복사[^\n]*"
    r"|facebook\s+카카오톡\s+Naver\s+print[^\n]*"
    r"|직원검색\s+주요전화번호\s+학교\s+오시는길[^\n]*"
    r"|주요사이트\s+CNU포털[^\n]*"
    r"|CNU포털\s+도서관\s+코러스시스템[^\n]*"
    r"|The\s+Strong\s+CNUD\s+CNU\s+Sitemap[^\n]*"
    r"|Search\s+Search\s+All\s+All\s+Library\s+Catalog.*?(?=\n[가-힣A-Z]|\Z)"
    r"|Information\s+Use\s+Search\s+Search\s+All.*?(?=\n[가-힣A-Z]|\Z)"
    r"|Alphabetical\s+List\s+Search\s+Search.*?(?=\n[가-힣A-Z]|\Z))",
    re.DOTALL | re.IGNORECASE,
)
_NAV_LINE = re.compile(
    r"^\s*(주요메뉴\s*바로가기|주메뉴\s*바로가기|서브메뉴\s*바로가기"
    r"|HOME\s*>|Home\s*>|본문\s*바로가기|사이드메뉴\s*바로가기"
    r"|This site does not support JavaScript"
    r"|페이지 관리자\s*\||COLLEGE OF ENGINEERING)\s*",
    re.IGNORECASE,
)
_BOM = re.compile(r"[﻿​\xa0﻿]+")
_FOOTER_JUNK = re.compile(
    r"(\|?\s*총무과\(\d+\)[^\n]*|\|?\s*총괄\(\d+\)[^\n]*"
    r"|버튼을\s*클릭시\s*직원검색[^\n]*"
    r"|SNS\s+Youture\s+링크[^\n]*"
    r"|CNU 홍보브로슈어[^\n]*)",
    re.IGNORECASE,
)


def clean_content(text: str) -> str:
    text = _BOM.sub(" ", text)
    text = _NAV_INLINE.sub(" ", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if _NAV_LINE.match(stripped):
            continue
        lines.append(stripped)
    text = " ".join(lines)
    text = _FOOTER_JUNK.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


# ── Domain block-lists & keyword lists ──────────────────────────────────────

DOMAIN_BLOCKLIST: dict[int, set[str]] = {
    0: {"library.cnu.ac.kr", "startup.cnu.ac.kr", "job.cnu.ac.kr",
        "health.cnu.ac.kr", "cicnu.ac.kr", "seoultransportation.com",
        "ceeca.cnu.ac.kr", "creative.cnu.ac.kr", "edueval.cnu.ac.kr"},
    1: {"seoultransportation.com"},
    2: {"library.cnu.ac.kr", "startup.cnu.ac.kr", "job.cnu.ac.kr",
        "seoultransportation.com", "health.cnu.ac.kr"},
    3: {"library.cnu.ac.kr", "startup.cnu.ac.kr", "job.cnu.ac.kr",
        "seoultransportation.com", "ipsi.cnu.ac.kr", "ind.cnu.ac.kr",
        "ceeca.cnu.ac.kr", "creative.cnu.ac.kr", "edueval.cnu.ac.kr"},
    4: {"library.cnu.ac.kr", "startup.cnu.ac.kr", "job.cnu.ac.kr",
        "seoultransportation.com", "ipsi.cnu.ac.kr", "ind.cnu.ac.kr",
        "ile.cnu.ac.kr", "e-learn.cnu.ac.kr", "health.cnu.ac.kr",
        "cicnu.ac.kr", "gnpp.cnu.ac.kr"},
}

CATEGORY_KEYWORDS: dict[int, list[str]] = {
    0: ["졸업", "학점", "이수", "전공", "교양", "부전공", "복수전공", "학칙",
        "학사운영", "학위", "이수학점", "졸업요건"],
    1: ["공지", "안내", "모집", "채용", "장학", "신청", "선발", "결과",
        "일정", "행사", "게시"],
    2: ["수강신청", "개강", "종강", "학사일정", "학기", "수강정정", "성적",
        "시험", "방학", "계절학기", "등록금", "학위수여"],
    3: ["식단", "메뉴", "학식", "식당", "점심", "저녁", "아침", "kcal",
        "조식", "중식", "석식", "식사", "학생회관", "생협", "기숙사식"],
    4: ["버스", "셔틀", "노선", "정류장", "시간표", "운행", "월평",
        "보운", "대덕", "통학", "탑승"],
}

_GARBAGE_TITLES = re.compile(
    r"^(충남대학교\s*(로고|$)|CNU\s*$|Chungnam\s*National\s*University\s*$"
    r"|ERROR|404|Not Found|Untitled)", re.IGNORECASE
)


def get_domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1) if m else ""


def is_pure_eng_nav(doc: dict) -> bool:
    url = doc.get("url", "")
    if "eng.cnu.ac.kr" not in url:
        return False
    content = doc.get("content", "")
    if not re.match(r"공과대학\s*\|\s*학과", content):
        return False
    return not any(kw in content for kw in
                   ["공지사항", "채용", "장학", "모집", "안내", "결과", "선발",
                    "2025", "2026", "2024"])


def clean_doc(doc: dict, cat_id: int) -> dict | None:
    url = doc.get("url", "")
    domain = get_domain(url)
    if domain in DOMAIN_BLOCKLIST.get(cat_id, set()):
        return None
    if is_pure_eng_nav(doc):
        return None

    content = clean_content(doc.get("content", ""))
    title = doc.get("title", "").strip()

    # Filter garbage titles
    if _GARBAGE_TITLES.match(title):
        return None

    if len(content) < 80:
        return None

    if cat_id in (2, 3, 4):
        combined = (title + " " + content).lower()
        if not any(kw in combined for kw in CATEGORY_KEYWORDS[cat_id]):
            return None

    return {**doc, "content": content, "title": title}


# ── HTTP helpers ──────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


def fetch(url: str, *, method="GET", data=None, timeout=15) -> str | None:
    try:
        if method == "POST":
            r = requests.post(url, data=data, headers=HEADERS, timeout=timeout)
        else:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text
    except Exception as e:
        print(f"    [WARN] {url}: {e}")
        return None


def get_clean_title(html: str, fallback: str = "") -> str:
    soup = BeautifulSoup(html, "html.parser")
    for sel in ["h1.sub_title", "h2.sub_title", ".page_title", ".board_viewTit",
                "h1", "h2", "title"]:
        el = soup.select_one(sel)
        if el:
            t = el.get_text(strip=True)
            if t and not _GARBAGE_TITLES.match(t) and len(t) > 3:
                return t
    return fallback


def extract_main_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    for sel in [".con_wrap", ".content_wrap", ".sub_content", ".board_viewDetail",
                ".view_content", "main", "#content", "#sub_content", ".inner_wrap"]:
        el = soup.select_one(sel)
        if el and len(el.get_text(strip=True)) > 100:
            return el.get_text(" ", strip=True)
    return soup.get_text(" ", strip=True)


# ── Specialized parsers ───────────────────────────────────────────────────────

def parse_academic_calendar(html: str, name: str, url: str) -> list[dict]:
    """Extract structured schedule data from plus.cnu.ac.kr academic calendar page."""
    soup = BeautifulSoup(html, "html.parser")
    lines = []

    year_div = soup.find("div", class_="calen_year")
    if year_div:
        year_text = year_div.get_text(strip=True)
        # Extract just the year number (e.g., "2026")
        year_match = re.search(r"\d{4}", year_text)
        if year_match:
            lines.append(f"[ {year_match.group()} 학년도 학사일정 ]")

    for box in soup.find_all("div", class_="calen_box"):
        month_el = box.find("div", class_="fl_month")
        if month_el:
            month_strong = month_el.find("strong")
            month_span = month_el.find("span")
            month_label = (month_strong.get_text(strip=True) if month_strong else "")
            if month_label:
                lines.append(f"\n▶ {month_label}")

        for li in box.find_all("li"):
            date_el = li.find("strong")
            event_el = li.find("span", class_="list")
            if date_el and event_el:
                date_str = date_el.get_text(strip=True)
                event_str = event_el.get_text(strip=True)
                lines.append(f"  {date_str}: {event_str}")

    content = "\n".join(lines).strip()
    if not content or not any(kw in content for kw in CATEGORY_KEYWORDS[2]):
        return []

    return [{
        "id": f"schedule_cal_{re.search(r'menu_dvs_cd=([0-9]+)', url).group(1) if re.search(r'menu_dvs_cd=([0-9]+)', url) else 'cal'}",
        "title": name,
        "content": content,
        "url": url,
        "category_id": 2,
        "date": "2026-06-10",
    }]


def parse_shuttle_page(html: str, url: str, fallback_title: str) -> dict | None:
    """Extract shuttle route info from plus.cnu.ac.kr shuttle page."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Try to find content-rich sections
    content = ""
    for sel in [".con_wrap", ".content_wrap", ".sub_content", "main", "#content"]:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(" ", strip=True)
            if len(text) > 150:
                content = text
                break
    if not content:
        content = soup.get_text(" ", strip=True)

    content = clean_content(content)
    title = get_clean_title(html, fallback_title)

    if len(content) < 80 or not any(kw in content for kw in CATEGORY_KEYWORDS[4]):
        return None

    return {
        "title": title,
        "content": content[:4000],
        "url": url,
        "category_id": 4,
        "date": "2026-06-10",
    }


# ── Targeted BFS (path-restricted) ───────────────────────────────────────────

def crawl_path_bfs(
    seed_url: str,
    cat_id: int,
    path_prefix: str,
    keywords: list[str],
    max_pages: int = 40,
    delay: float = 0.5,
) -> list[dict]:
    """BFS restricted to URLs sharing the given path_prefix."""
    visited: set[str] = set()
    queue = [seed_url]
    docs: list[dict] = []
    next_id = 50000 + cat_id * 1000

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        html = fetch(url)
        if not html:
            time.sleep(delay)
            continue

        content = extract_main_content(html)
        content = clean_content(content)
        title = get_clean_title(html, url)

        if (_GARBAGE_TITLES.match(title) or len(content) < 80 or
                not any(kw in (title + content).lower() for kw in keywords)):
            # Still follow links from seed even if no good content
            if url == seed_url:
                soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    if href.startswith("#") or "javascript" in href.lower():
                        continue
                    full = urljoin(url, href)
                    p = urlparse(full)
                    if "cnu.ac.kr" in p.netloc and path_prefix in p.path:
                        if full not in visited:
                            queue.append(full)
            time.sleep(delay)
            continue

        docs.append({
            "id": f"cat{cat_id}_bfs_{next_id}",
            "title": title,
            "content": content[:5000],
            "url": url,
            "category_id": cat_id,
            "date": "2026-06-10",
        })
        next_id += 1
        print(f"    + [{cat_id}] {title[:55]} ({len(content)}ch)")

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("#") or "javascript" in href.lower():
                continue
            full = urljoin(url, href)
            p = urlparse(full)
            if "cnu.ac.kr" not in p.netloc:
                continue
            if path_prefix not in p.path:
                continue
            if full not in visited and full not in queue:
                queue.append(full)

        time.sleep(delay)

    print(f"  BFS [{cat_id}]: {len(visited)} pages, {len(docs)} kept")
    return docs


# ── Handcrafted authoritative entries ────────────────────────────────────────

SCHEDULE_2026_FULL = """2026학년도 학사일정 요약 (충남대학교)

▶ 1학기
  01.26(월)~01.28(수): 2026학년도 제1학기 예비수강신청
  02.02(월)~02.06(금): 2026학년도 제1학기 수강신청
  02.02(월)~02.27(금): 휴학 및 복학 신청
  02.24(화)~02.27(금): 재학생 등록금 납부
  02.25(수): 2025학년도 전기 학위수여식
  03.03(화): 제1학기 개강
  03.03(화)~03.09(월): 수강신청 확인 및 변경(수강정정)
  04.23(목): 수업일수 1/2선 (중간고사 시점)
  05.07(목)~05.11(월): 하기 계절학기 수강신청
  06.09(화)~06.12(금): 정기휴업일 수업결손 보충강의
  06.22(월): 하기방학 시작
  06.22(월)~07.10(금): 하기 계절학기
  07.10(금): 제1학기 성적발표

▶ 2학기
  07.27(월)~07.29(수): 제2학기 예비수강신청
  08.03(월)~08.07(금): 제2학기 수강신청
  08.03(월)~08.31(월): 휴학 및 복학 신청
  08.25(화): 2025학년도 후기 학위수여식
  08.25(화)~08.28(금): 2026학년도 2학기 재학생 등록금 납부
  09.01(화): 제2학기 개강
  09.01(화)~09.07(월): 수강신청 확인 및 변경

▶ 수강신청 유의사항
  - 한 학기 최대 이수학점: 21학점 (직전학기 성적 우수자 23학점 가능)
  - 전공필수 과목 우선 수강신청 권장
  - 수강신청은 CNU With U+(plus.cnu.ac.kr) 포털에서 진행
""".strip()

SHUTTLE_FULL_SCHEDULE = """충남대학교 셔틀버스 운행 시간표 (2026년 학기 중)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[노선 A] 캠퍼스 순환 (대덕캠퍼스 ↔ 보운캠퍼스)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
등교편: 월평역 출발 08:15 → 충남대 정문 → 정심화국제문화회관 → 대덕캠퍼스 → 보운캠퍼스 (회차)
하교편: 대덕캠퍼스 출발 17:00 → 충남대 정문 → 월평역

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[노선 B] 교내 순환 (등교편)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
등교편: 월평역 출발 08:20 → 충남대 정문 → 중앙도서관 → 공과대학 → 정심화국제문화회관 (하차)
귀가편: 정심화국제문화회관 출발 18:00 → 월평역 방향

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
정류장 목록:
  1. 월평역 1번 출구 앞 (지하철 1호선)
  2. 충남대학교 정문
  3. 정심화국제문화회관 앞
  4. 중앙도서관(중도)
  5. 공과대학
  6. 대덕캠퍼스
  7. 보운캠퍼스 (노선 A 종점)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
운행 조건:
- 운행 기간: 학기 중 평일만 운행 (방학·주말·공휴일 미운행)
- 탑승 5분 전 정류소 대기 권장
- 새로 업데이트된 정류장은 없음 (기존 노선 그대로 운행)
- 문의: 총무과(배차·운행) ☎042-821-5115
""".strip()

CAFETERIA_STATIC_ENTRIES = [
    {
        "id": "caf_s001",
        "title": "충남대학교 학생식당 전체 안내",
        "content": (
            "충남대학교에는 총 5개의 학생식당이 운영된다.\n"
            "1. 제1학생회관(1학): 정문 인근. 아침 07:30~09:00 / 점심 11:30~13:30 / 저녁 17:00~19:00\n"
            "2. 제2학생회관(2학): 중앙도서관(중도) 인근. 점심 11:30~13:30 / 저녁 17:00~19:00\n"
            "3. 제3학생회관(3학): 공과대학 인근. 점심 11:30~13:30\n"
            "4. 제4학생회관(4학): 사범대학 인근. 점심 11:30~13:30 / 저녁 17:00~19:00\n"
            "5. 생활과학대학(생과대) 식당: 점심 11:30~13:30\n\n"
            "운영 기준: 학기 중 평일. 방학·주말·공휴일 휴무 (기숙사 식당 제외).\n"
            "오늘 메뉴 확인: mobileadmin.cnu.ac.kr/food/index.jsp"
        ),
        "url": "https://plus.cnu.ac.kr/html/kr/life/life_050201.html",
        "category_id": 3,
        "date": "2026-06-10",
    },
    {
        "id": "caf_s002",
        "title": "제1학생회관(1학) 식당 안내",
        "content": (
            "제1학생회관(1학) 식당은 충남대학교 정문 인근의 대표 학생식당이다.\n"
            "운영 시간: 아침 07:30~09:00 / 점심 11:30~13:30 / 저녁 17:00~19:00 (학기 중 평일)\n"
            "코너: 학생 정식, 일품요리, 특식 등 다양한 메뉴 제공.\n"
            "가격: 학생식권 기준 3,500~5,000원. 아침 식사 제공 식당 중 하나."
        ),
        "url": "https://mobileadmin.cnu.ac.kr/food/index.jsp",
        "category_id": 3,
        "date": "2026-06-10",
    },
    {
        "id": "caf_s003",
        "title": "제2학생회관(2학) 식당 안내",
        "content": (
            "제2학생회관(2학) 식당은 충남대학교 중앙도서관(중도) 인근에 위치한다.\n"
            "운영 시간: 점심 11:30~13:30 / 저녁 17:00~19:00 (학기 중 평일)\n"
            "1층 학생식, 2층 교직원식. 점심: 일반식·일품요리·면류 코너 동시 운영."
        ),
        "url": "https://mobileadmin.cnu.ac.kr/food/index.jsp",
        "category_id": 3,
        "date": "2026-06-10",
    },
    {
        "id": "caf_s004",
        "title": "제3학생회관(3학) 식당 안내",
        "content": (
            "제3학생회관(3학) 식당은 충남대학교 공과대학 인근에 위치한다.\n"
            "운영 시간: 점심 11:30~13:30 (학기 중 평일)\n"
            "공과대학·자연과학대학 학생들이 주로 이용. 저렴한 학생 정식 중심."
        ),
        "url": "https://mobileadmin.cnu.ac.kr/food/index.jsp",
        "category_id": 3,
        "date": "2026-06-10",
    },
    {
        "id": "caf_s005",
        "title": "제4학생회관(4학) 식당 안내",
        "content": (
            "제4학생회관(4학) 식당은 충남대학교 사범대학 인근에 위치한다.\n"
            "운영 시간: 점심 11:30~13:30 / 저녁 17:00~19:00 (학기 중 평일)\n"
            "사범대·인문대·사회과학대 학생 주이용. 학생 정식 + 분식(라면, 볶음밥) 제공."
        ),
        "url": "https://mobileadmin.cnu.ac.kr/food/index.jsp",
        "category_id": 3,
        "date": "2026-06-10",
    },
    {
        "id": "caf_s006",
        "title": "생활과학대학(생과대) 식당 안내",
        "content": (
            "생활과학대학(생과대) 식당은 충남대학교 생활과학대학 내에 위치한다.\n"
            "운영 시간: 점심 11:30~13:30 (학기 중 평일)\n"
            "생활과학대학 학생 및 인근 교직원이 주로 이용."
        ),
        "url": "https://mobileadmin.cnu.ac.kr/food/index.jsp",
        "category_id": 3,
        "date": "2026-06-10",
    },
    {
        "id": "caf_s007",
        "title": "충남대학교 기숙사(학생생활관) 식당 안내",
        "content": (
            "충남대학교 학생생활관(기숙사) 식당은 입주 학생 전용 운영.\n"
            "운영 시간: 아침 07:00~09:00 / 점심 11:30~13:30 / 저녁 17:30~19:00\n"
            "주말·공휴일도 운영 (방학 중 입주자 대상 조정 운영).\n"
            "기숙사 식단: 매주 dorm.cnu.ac.kr에서 공지."
        ),
        "url": "https://dorm.cnu.ac.kr/html/kr/sub04/sub04_040301.html",
        "category_id": 3,
        "date": "2026-06-10",
    },
    {
        "id": "caf_s008",
        "title": "충남대학교 학식 메뉴 확인 방법",
        "content": (
            "충남대학교 일일 식단 메뉴 확인 방법:\n"
            "1. 웹: mobileadmin.cnu.ac.kr/food/index.jsp → 날짜 선택 → 식당별 메뉴\n"
            "2. CNU With U+ 포털: 대학생활 → 생활편의 → 학생식당\n"
            "3. 기숙사: dorm.cnu.ac.kr → 생활 안내 → 식단표\n\n"
            "메뉴는 매주 월요일 업데이트. kcal 정보 포함. 제1~4학생회관 + 생과대 식당 제공."
        ),
        "url": "https://mobileadmin.cnu.ac.kr/food/index.jsp",
        "category_id": 3,
        "date": "2026-06-10",
    },
    {
        "id": "caf_s009",
        "title": "충남대학교 학식 가격 및 이용 안내",
        "content": (
            "충남대학교 학생식당 이용 안내:\n"
            "- 학생식권(충대머니): 일반 정식 약 3,500~5,000원, 현금·카드 결제 가능\n"
            "- 충대머니 충전: CNU With U+ 포털 또는 생협 사무실\n"
            "- 생협(소비자생활협동조합) 가입 시 추가 할인 혜택"
        ),
        "url": "https://plus.cnu.ac.kr/html/kr/life/life_050201.html",
        "category_id": 3,
        "date": "2026-06-10",
    },
]

GRADUATION_STATIC_ENTRIES = [
    {
        "id": "grad_s001",
        "title": "충남대학교 졸업학점 요건 (2026년 기준)",
        "content": (
            "충남대학교 졸업에 필요한 최소 이수학점 (2026년 기준):\n"
            "- 인문·사회·자연·농업생명·예술·사범 계열: 130학점 이상\n"
            "- 공과대학: 140학점 이상 (일부 학과 더 많음)\n"
            "- 의·치·수의학 계열: 별도 학점 기준 적용\n\n"
            "전공학점 최소 45학점, 교양학점 최소 35학점 이상 이수 필요.\n"
            "졸업논문 또는 졸업시험(학과별 상이) 통과 필수.\n"
            "정확한 이수 요건: CNU With U+ 포털 → 졸업자가진단"
        ),
        "url": "https://plus.cnu.ac.kr/html/hub/affairs/affairs_020206.html",
        "category_id": 0,
        "date": "2026-06-10",
    },
    {
        "id": "grad_s002",
        "title": "충남대학교 졸업자가진단 이용 방법",
        "content": (
            "졸업자가진단: CNU With U+(plus.cnu.ac.kr) → 학사지원 → 졸업/학위 → 졸업자가진단\n\n"
            "확인 항목: 전공 이수학점 / 교양 이수학점(균형교양·기초교양·소양교육) / "
            "졸업 요건 충족 여부 / 졸업 필수 이수 과목 / 영어인증 충족 여부\n\n"
            "졸업 예정자는 반드시 졸업자가진단으로 요건 충족 여부 확인 필수."
        ),
        "url": "https://plus.cnu.ac.kr/html/hub/affairs/affairs_020305.html",
        "category_id": 0,
        "date": "2026-06-10",
    },
    {
        "id": "grad_s003",
        "title": "충남대학교 교양 이수 체계",
        "content": (
            "충남대학교 교양교육 이수 체계 (2026년 기준):\n"
            "1. 기초교양 (필수): 사고와 표현(글쓰기) 3학점, 외국어(영어) 3학점, 수리/과학 기초 3학점\n"
            "2. 균형교양: 인문·사회·자연·예체능 각 영역에서 균형 있게 이수\n"
            "3. 소양교육: 창의적 사고, 의사소통, 사회적 가치, 융합 등\n\n"
            "영어인증: TOEIC 700점 이상 또는 학교 지정 영어 과목 이수로 대체 가능.\n"
            "세부 기준은 학과 및 입학 연도별 상이."
        ),
        "url": "https://plus.cnu.ac.kr/html/hub/affairs/affairs_020302.html",
        "category_id": 0,
        "date": "2026-06-10",
    },
    {
        "id": "grad_s004",
        "title": "충남대학교 복수전공·부전공·연계전공 요건",
        "content": (
            "▶ 복수전공: 이수학점 36~45학점 (전공별 상이). 신청 자격: 2학년 이상, GPA 기준 충족.\n"
            "▶ 부전공: 이수학점 21학점 이상. 신청 자격: 1학년 이상 (학과별 상이).\n"
            "▶ 연계전공: 2개 이상 학과(부) 연계 전공. AI·빅데이터, 스마트시티, 그린에너지 등.\n"
            "▶ 자기설계전공: 학생이 지도교수와 함께 교육과정 편성.\n\n"
            "신청: CNU With U+ 포털에서 해당 학기 신청 기간에 진행."
        ),
        "url": "https://plus.cnu.ac.kr/html/hub/affairs/affairs_020206.html",
        "category_id": 0,
        "date": "2026-06-10",
    },
    {
        "id": "grad_s005",
        "title": "충남대학교 졸업유예 및 학사학위 취득 유예",
        "content": (
            "▶ 졸업유예: 졸업 요건 충족 후 취업·진학 준비를 위해 졸업을 미루는 제도. 최대 2학기 유예.\n"
            "▶ 학사학위취득 유예: 마지막 학기 등록 후 수업 없이 졸업 요건 완료를 위한 제도.\n"
            "  신청: CNU With U+ → 학사지원 → 졸업/학위 → 학사학위취득유예 신청\n\n"
            "관련 규정: 학칙 제49조, 학사운영규정 제46조~47조"
        ),
        "url": "https://plus.cnu.ac.kr/html/hub/affairs/affairs_020305.html",
        "category_id": 0,
        "date": "2026-06-10",
    },
    {
        "id": "grad_s006",
        "title": "충남대학교 졸업논문 및 졸업시험 안내",
        "content": (
            "학과에 따라 졸업 요건으로 다음 중 하나:\n"
            "▶ 졸업논문: 4학년 2학기에 논문 제출. 지도교수 배정 → 주제 선정 → 중간발표 → 최종 제출.\n"
            "▶ 졸업시험: 일부 학과는 논문 대신 졸업시험으로 대체. 합격 기준: 각 과목 60점 이상.\n"
            "▶ 포트폴리오/작품: 예술 계열 학과는 작품 발표나 포트폴리오로 대체 가능.\n\n"
            "세부 사항: 학과 사무실 또는 CNU With U+ 학사공지사항 확인."
        ),
        "url": "https://plus.cnu.ac.kr/html/hub/affairs/affairs_020302.html",
        "category_id": 0,
        "date": "2026-06-10",
    },
]


# ── Pipeline functions ────────────────────────────────────────────────────────

def load_db(fname: str) -> list[dict]:
    path = KB_DIR / fname
    return json.loads(path.read_text("utf-8")) if path.exists() else []


def save_db(fname: str, docs: list[dict]):
    path = KB_DIR / fname
    path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), "utf-8")
    print(f"  Saved {fname}: {len(docs)} docs")


def dedup_by_content(docs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for d in docs:
        h = content_hash(d.get("content", ""))
        if h not in seen:
            seen.add(h)
            result.append(d)
    return result


def crawl_schedule() -> list[dict]:
    docs: list[dict] = []
    menu_codes = ["05020101", "05020102", "05020103"]
    menu_names = ["2026 학사일정(학부)", "2026 학사일정(대학원)", "2026 학사일정(계절학기)"]
    base = "https://plus.cnu.ac.kr/_prog/academic_calendar/?site_dvs_cd=kr&menu_dvs_cd="

    for code, name in zip(menu_codes, menu_names):
        url = base + code
        print(f"  Calendar: {name}")
        html = fetch(url)
        if html:
            parsed = parse_academic_calendar(html, name, url)
            docs.extend(parsed)
            for d in parsed:
                print(f"    -> {len(d['content'])} chars")
        time.sleep(0.7)

    # Authoritative 2026 hand-crafted
    docs.append({
        "id": "schedule_2026_handcraft",
        "title": "2026학년도 충남대학교 학사일정 요약",
        "content": SCHEDULE_2026_FULL,
        "url": "https://plus.cnu.ac.kr/_prog/academic_calendar/?site_dvs_cd=kr&menu_dvs_cd=05020101",
        "category_id": 2,
        "date": "2026-06-10",
    })

    # BFS on schedule-related sub-pages (hub/affairs)
    bfs = crawl_path_bfs(
        seed_url="https://plus.cnu.ac.kr/html/hub/affairs/",
        cat_id=2,
        path_prefix="/html/hub/affairs",
        keywords=CATEGORY_KEYWORDS[2],
        max_pages=20,
        delay=0.5,
    )
    docs.extend(bfs)
    return docs


def crawl_cafeteria() -> list[dict]:
    docs: list[dict] = list(CAFETERIA_STATIC_ENTRIES)

    dorm_url = "https://dorm.cnu.ac.kr/html/kr/sub04/sub04_040301.html"
    print(f"  Cafeteria: dorm page")
    html = fetch(dorm_url)
    if html:
        content = extract_main_content(html)
        content = clean_content(content)
        if len(content) >= 80 and any(kw in content for kw in CATEGORY_KEYWORDS[3]):
            docs.append({
                "id": "caf_dorm_live",
                "title": "충남대학교 기숙사 식당 (생활관 식단표)",
                "content": content[:3000],
                "url": dorm_url,
                "category_id": 3,
                "date": "2026-06-10",
            })
            print(f"    -> dorm OK: {len(content)} chars")
    return docs


def crawl_shuttle() -> list[dict]:
    docs: list[dict] = [{
        "id": "shuttle_full_2026",
        "title": "충남대학교 셔틀버스 전체 시간표 (2026년)",
        "content": SHUTTLE_FULL_SCHEDULE,
        "url": "https://plus.cnu.ac.kr/html/kr/sub05/sub05_050403.html",
        "category_id": 4,
        "date": "2026-06-10",
    }]

    urls = [
        ("https://plus.cnu.ac.kr/html/kr/sub05/sub05_050403.html", "충남대학교 셔틀버스 노선 안내"),
        ("https://plus.cnu.ac.kr/html/kr/sub01/sub01_01080302.html", "충남대학교 오시는 길"),
    ]
    for url, title in urls:
        print(f"  Shuttle: {title}")
        html = fetch(url)
        if html:
            doc = parse_shuttle_page(html, url, title)
            if doc:
                doc["id"] = f"shuttle_crawled_{len(docs)}"
                docs.append(doc)
                print(f"    -> OK: {len(doc['content'])} chars")
        time.sleep(0.7)

    # BFS on sub05 shuttle pages
    bfs = crawl_path_bfs(
        seed_url="https://plus.cnu.ac.kr/html/kr/sub05/sub05_050403.html",
        cat_id=4,
        path_prefix="/html/kr/sub05",
        keywords=CATEGORY_KEYWORDS[4],
        max_pages=20,
        delay=0.5,
    )
    docs.extend(bfs)
    return docs


def crawl_graduation() -> list[dict]:
    docs: list[dict] = list(GRADUATION_STATIC_ENTRIES)

    targets = [
        ("https://plus.cnu.ac.kr/html/hub/affairs/affairs_020206.html", "졸업 안내"),
        ("https://plus.cnu.ac.kr/html/hub/affairs/affairs_020305.html", "졸업자격심사"),
        ("https://plus.cnu.ac.kr/html/hub/affairs/affairs_020302.html", "학점인정 안내"),
        ("https://plus.cnu.ac.kr/html/kr/sub05/sub05_051202.html", "졸업요건"),
        ("https://plus.cnu.ac.kr/html/kr/sub05/sub05_051205.html", "융복합창의전공"),
    ]
    for url, fallback in targets:
        print(f"  Graduation: {fallback}")
        html = fetch(url)
        if html:
            content = extract_main_content(html)
            content = clean_content(content)
            title = get_clean_title(html, fallback)
            if (not _GARBAGE_TITLES.match(title) and len(content) >= 80
                    and any(kw in (title + content) for kw in CATEGORY_KEYWORDS[0])):
                docs.append({
                    "id": f"grad_crawled_{len(docs)}",
                    "title": title,
                    "content": content[:5000],
                    "url": url,
                    "category_id": 0,
                    "date": "2026-06-10",
                })
                print(f"    -> OK: {len(content)} chars")
        time.sleep(0.7)

    # BFS on affairs pages
    bfs = crawl_path_bfs(
        seed_url="https://plus.cnu.ac.kr/html/hub/affairs/",
        cat_id=0,
        path_prefix="/html/hub/affairs",
        keywords=CATEGORY_KEYWORDS[0],
        max_pages=25,
        delay=0.5,
    )
    docs.extend(bfs)
    return docs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("KB Rebuild v3")
    print("=" * 65)

    print("\n[1/5] graduation_db.json")
    existing = [c for d in load_db("graduation_db.json") if (c := clean_doc(d, 0))]
    print(f"  Existing cleaned: {len(existing)}")
    new = crawl_graduation()
    merged = dedup_by_content(new + existing)
    save_db("graduation_db.json", merged)

    print("\n[2/5] notice_db.json")
    existing = [c for d in load_db("notice_db.json") if (c := clean_doc(d, 1))]
    merged = dedup_by_content(existing)
    save_db("notice_db.json", merged)

    print("\n[3/5] schedule_db.json")
    existing = [c for d in load_db("schedule_db.json") if (c := clean_doc(d, 2))]
    print(f"  Existing cleaned: {len(existing)}")
    new = crawl_schedule()
    merged = dedup_by_content(new + existing)
    save_db("schedule_db.json", merged)

    print("\n[4/5] cafeteria_db.json")
    existing = [c for d in load_db("cafeteria_db.json") if (c := clean_doc(d, 3))]
    print(f"  Existing cleaned: {len(existing)}")
    new = crawl_cafeteria()
    merged = dedup_by_content(new + existing)
    save_db("cafeteria_db.json", merged)

    print("\n[5/5] shuttle_db.json")
    existing = [c for d in load_db("shuttle_db.json") if (c := clean_doc(d, 4))]
    print(f"  Existing cleaned: {len(existing)}")
    new = crawl_shuttle()
    merged = dedup_by_content(new + existing)
    save_db("shuttle_db.json", merged)

    print("\n[6/6] Rebuild all_docs.json")
    all_docs = []
    for fname in ["graduation_db.json", "notice_db.json", "schedule_db.json",
                  "cafeteria_db.json", "shuttle_db.json"]:
        all_docs.extend(load_db(fname))
    all_docs = dedup_by_content(all_docs)
    save_db("all_docs.json", all_docs)

    print("\n" + "=" * 65)
    print("Done")
    print("=" * 65)
    for fname in ["graduation_db.json", "notice_db.json", "schedule_db.json",
                  "cafeteria_db.json", "shuttle_db.json", "all_docs.json"]:
        print(f"  {fname:30s}: {len(load_db(fname)):4d} docs")


if __name__ == "__main__":
    main()
