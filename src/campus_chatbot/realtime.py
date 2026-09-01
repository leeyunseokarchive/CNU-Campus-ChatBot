from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .io_utils import PROJECT_ROOT

# ─── Constants ─────────────────────────────────────────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
}

_MEAL_URL = "https://mobileadmin.cnu.ac.kr/food/index.jsp"
_DORM_MEAL_URL = "https://dorm.cnu.ac.kr/html/kr/sub04/sub04_040301.html"

# 셔틀버스는 고정값이므로 실시간 크롤링 대상에서 제외
NOTICE_BOARDS = [
    ("sub07_0701", "0701", "일반공지"),
    ("sub07_0702", "0702", "학사공지"),
    ("sub07_0713", "0713", "장학공지"),
    ("sub07_0703", "0703", "취업공지"),
    ("sub07_0704", "0704", "학생공지"),
    ("sub07_0709", "0709", "국제교류공지"),
    ("sub07_0705", "0705", "행사문화공지"),
    ("sub010714",  "0712", "채용초빙공고"),
]

_BOARD_BASE = "https://plus.cnu.ac.kr/_prog/_board/"

# 게시판별 개별 글 상위 N개 크롤링
_DEEP_CRAWL_N = 5


class MultiSourceCrawler:
    def __init__(self):
        self.kb_realtime_dir = PROJECT_ROOT / "data" / "kb" / "realtime"
        self.kb_realtime_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self.session.verify = False

    # ─── Meal ─────────────────────────────────────────────────────────────────

    def _parse_mobileadmin_meal(self, html: str, date_str: str) -> list[dict]:
        """Parse per-restaurant meal table from mobileadmin."""
        soup = BeautifulSoup(html, "html.parser")
        table = soup.select_one("table.menu-tbl")
        if not table:
            return []

        # Column headers → restaurant names
        headers = []
        for th in table.select("thead th"):
            t = th.get_text(strip=True)
            headers.append(t)
        # headers: ['구분', '(blank/직원구분)', '제1학생회관', '제2학생회관', ...]
        # But actual header might differ; find restaurant names (제N학생회관, 생활과학대학)
        restaurant_cols: list[str] = []
        for h in headers:
            if any(kw in h for kw in ["학생회관", "생활과학", "식당", "생협"]):
                restaurant_cols.append(h)

        # Parse rows: meal_type × (직원|학생) × restaurant
        results: dict[str, dict[str, dict[str, list[str]]]] = {}
        # structure: results[restaurant][meal_type][구분] = [menu items]

        current_meal_type = ""
        for tr in table.select("tbody tr"):
            cells = tr.select("td, th")
            if not cells:
                continue

            texts = [c.get_text(separator="\n", strip=True) for c in cells]

            # Detect meal type cell (조식/중식/석식) — usually rowspan td
            first_td = cells[0]
            first_text = first_td.get_text(strip=True)
            if first_text in ("조식", "중식", "석식"):
                current_meal_type = first_text
                # row: [meal_type, 직원/학생, rest1, rest2, ...]
                row_cells = texts[1:]
            else:
                # row: [직원/학생, rest1, rest2, ...]
                row_cells = texts[:]

            if not current_meal_type:
                continue

            # Determine 직원/학생
            if not row_cells:
                continue
            staff_student = row_cells[0] if row_cells[0] in ("직원", "학생") else "학생"
            menu_cells = row_cells[1:]

            # Align menu_cells with restaurant_cols
            for idx, restaurant in enumerate(restaurant_cols):
                if idx >= len(menu_cells):
                    break
                menu_text = menu_cells[idx]
                if "운영안함" in menu_text or "메뉴운영내역" in menu_text:
                    continue

                # Parse menu items (split by newline / li tags)
                items = [
                    line.strip()
                    for line in menu_text.split("\n")
                    if line.strip() and len(line.strip()) > 1
                ]

                if restaurant not in results:
                    results[restaurant] = {}
                if current_meal_type not in results[restaurant]:
                    results[restaurant][current_meal_type] = {}
                results[restaurant][current_meal_type][staff_student] = items

        # Convert to flat item list
        output = []
        for restaurant, meal_map in results.items():
            for meal_type, staff_map in meal_map.items():
                for staff_student, items in staff_map.items():
                    if not items:
                        continue
                    content = "\n".join(items)
                    title = f"{date_str} {restaurant} {meal_type} ({staff_student})"
                    output.append({
                        "restaurant": restaurant,
                        "meal_type": meal_type,
                        "staff_student": staff_student,
                        "date": date_str,
                        "title": title,
                        "menu_items": items,
                        "content": content,
                        "source": "mobileadmin",
                    })
        return output

    def _parse_dorm_meal(self, html: str, date_str: str) -> list[dict]:
        """Parse today's meal from the dorm page.

        Table structure: header row = 아침 | 점심 | 저녁 (column headers)
        Body rows = menu items for each meal type.
        """
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.select("table")
        if not tables:
            return []

        table = tables[0]
        rows = table.select("tr")
        if not rows:
            return []

        # First row: detect meal-type column mapping
        header_cells = [td.get_text(strip=True) for td in rows[0].select("th, td")]
        meal_type_map = {"아침": "조식", "점심": "중식", "저녁": "석식"}

        # Build col_index → meal_type mapping
        col_to_meal: dict[int, str] = {}
        for i, h in enumerate(header_cells):
            for kr, mapped in meal_type_map.items():
                if kr in h:
                    col_to_meal[i] = mapped
                    break

        if not col_to_meal:
            return []

        output = []
        # Collect all body rows and merge into per-meal-type buckets
        meal_items: dict[str, list[str]] = {m: [] for m in meal_type_map.values()}

        for tr in rows[1:]:
            cells = tr.select("th, td")
            for col_idx, meal_type in col_to_meal.items():
                if col_idx >= len(cells):
                    continue
                cell_text = cells[col_idx].get_text(separator="\n", strip=True)
                lines = [
                    ln.strip()
                    for ln in cell_text.split("\n")
                    if ln.strip() and len(ln.strip()) > 2
                ]
                for ln in lines:
                    # Strip allergen codes like "5,6,9,15,16" at end
                    cleaned = re.sub(r'\s+\d+(?:,\d+)+\s*$', '', ln).strip()
                    # Strip bracket annotations like [쌀:국내산]
                    cleaned = re.sub(r'\[.*?\]', '', cleaned).strip()
                    if cleaned and cleaned not in meal_items[meal_type]:
                        meal_items[meal_type].append(cleaned)

        for meal_type, items in meal_items.items():
            if not items:
                continue
            title = f"{date_str} 학생생활관 {meal_type}"
            output.append({
                "restaurant": "학생생활관",
                "meal_type": meal_type,
                "staff_student": "학생",
                "date": date_str,
                "title": title,
                "menu_items": items,
                "content": "\n".join(items),
                "source": "dorm",
            })

        return output

    def crawl_meal(self) -> dict[str, Any]:
        today = datetime.now()
        date_str = today.strftime("%Y-%m-%d")
        ymd = today.strftime("%Y.%m.%d")

        items = []

        # 1. mobileadmin (제1~4학생회관, 생활과학대학)
        try:
            r = self.session.post(
                _MEAL_URL,
                data={
                    "searchYmd": ymd,
                    "searchLang": "OCL04.10",
                    "searchView": "cafeteria",
                    "searchCafeteria": "OCL03.02",
                },
                timeout=15,
            )
            r.raise_for_status()
            r.encoding = "utf-8"
            parsed = self._parse_mobileadmin_meal(r.text, date_str)
            items.extend(parsed)
            print(f"  [meal] mobileadmin: {len(parsed)} menu entries")
        except Exception as e:
            print(f"  [meal] mobileadmin ERROR: {e}")

        # 2. dorm (학생생활관)
        try:
            r = self.session.get(_DORM_MEAL_URL, timeout=15)
            r.raise_for_status()
            r.encoding = "utf-8"
            parsed = self._parse_dorm_meal(r.text, date_str)
            items.extend(parsed)
            print(f"  [meal] dorm: {len(parsed)} menu entries")
        except Exception as e:
            print(f"  [meal] dorm ERROR: {e}")

        result = {
            "category": "meal",
            "date": date_str,
            "crawled_at": datetime.now().isoformat(),
            "items": items,
        }

        cat_dir = self.kb_realtime_dir / "meal"
        cat_dir.mkdir(exist_ok=True)
        with open(cat_dir / f"{date_str}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    # ─── Notice ───────────────────────────────────────────────────────────────

    def _fetch_post_content(self, board_code: str, menu_dvs_cd: str, post_no: str) -> str:
        """Fetch full content of a single notice post."""
        url = (
            f"{_BOARD_BASE}?mode=V&no={post_no}"
            f"&code={board_code}&site_dvs_cd=kr&menu_dvs_cd={menu_dvs_cd}"
        )
        try:
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            r.encoding = "utf-8"
            soup = BeautifulSoup(r.text, "html.parser")

            # Extract post metadata: title, date, author
            title_el = soup.select_one(".board_viewTit")
            title = title_el.get_text(strip=True) if title_el else ""

            # Extract post detail (actual body)
            detail_el = soup.select_one(".board_viewDetail")
            body = detail_el.get_text(separator="\n", strip=True) if detail_el else ""

            # Get file attachments if any
            file_els = soup.select(".board_viewFile a, .file_list a, .attach a")
            files = [a.get_text(strip=True) for a in file_els if a.get_text(strip=True)]

            # Compose structured content
            parts = []
            if title:
                parts.append(f"[제목] {title}")
            if body:
                parts.append(body)
            if files:
                parts.append("[첨부파일] " + ", ".join(files[:5]))

            text = "\n".join(parts)

            # Clean up
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            if len(text) > 2000:
                text = text[:2000] + "..."
            return text.strip()
        except Exception as e:
            return f"(내용 조회 실패: {e})"

    def _crawl_notice_board(self, board_code: str, menu_dvs_cd: str, board_name: str) -> list[dict]:
        """Crawl notice board listing and top-N individual posts."""
        url = (
            f"{_BOARD_BASE}?code={board_code}"
            f"&site_dvs_cd=kr&menu_dvs_cd={menu_dvs_cd}"
        )
        try:
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            r.encoding = "utf-8"
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            print(f"  [notice] {board_name} list ERROR: {e}")
            return []

        # Extract individual post links and titles
        posts = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            title = a.get_text(strip=True)
            # Individual post links contain mode=V and no=NUMBER
            if "mode=V" in href and "no=" in href and len(title) > 5:
                no_match = re.search(r'no=(\d+)', href)
                if no_match:
                    posts.append({
                        "title": title,
                        "no": no_match.group(1),
                        "href": href,
                    })

        # Deduplicate by post number
        seen = set()
        unique_posts = []
        for p in posts:
            if p["no"] not in seen:
                seen.add(p["no"])
                unique_posts.append(p)

        # Also extract dates from the listing table
        date_map: dict[str, str] = {}
        table = soup.select_one(
            "table.board_list, table.bbs_list, .board_wrap table, table"
        )
        if table:
            for tr in table.select("tr"):
                tds = tr.select("td")
                title_text = ""
                date_text = ""
                post_no = ""
                for td in tds:
                    a_tag = td.find("a", href=True)
                    if a_tag:
                        t = a_tag.get_text(strip=True)
                        href_val = a_tag.get("href", "")
                        if len(t) > 5 and "no=" in href_val:
                            title_text = t
                            no_m = re.search(r'no=(\d+)', href_val)
                            if no_m:
                                post_no = no_m.group(1)
                    raw = td.get_text(strip=True)
                    if re.match(r'\d{4}[-./]\d{2}[-./]\d{2}', raw):
                        date_text = raw
                if post_no and date_text:
                    date_map[post_no] = date_text

        # Deep-crawl top N posts
        results = []
        for post in unique_posts[:_DEEP_CRAWL_N]:
            post_no = post["no"]
            post_title = post["title"]
            post_date = date_map.get(post_no, "")
            post_url = urljoin(_BOARD_BASE, post["href"])

            content = self._fetch_post_content(board_code, menu_dvs_cd, post_no)

            results.append({
                "board": board_name,
                "post_no": post_no,
                "title": post_title,
                "date": post_date,
                "url": post_url,
                "content": content,
            })
            print(f"    [notice] {board_name} post {post_no}: {post_title[:40]}")
            time.sleep(0.3)  # polite crawling

        return results

    def crawl_notice(self) -> dict[str, Any]:
        date_str = datetime.now().strftime("%Y-%m-%d")
        all_posts = []

        for board_code, menu_dvs_cd, board_name in NOTICE_BOARDS:
            posts = self._crawl_notice_board(board_code, menu_dvs_cd, board_name)
            all_posts.extend(posts)

        # Build summary content string for RAG
        summary_lines = [f"[{date_str} 기준 충남대학교 공지사항 요약]\n"]
        for post in all_posts:
            summary_lines.append(
                f"[{post['board']}] {post['date']} - {post['title']}"
            )

        # Structured items: each post is a separate item
        items = []
        for post in all_posts:
            items.append({
                "url": post["url"],
                "board": post["board"],
                "title": post["title"],
                "date": post["date"],
                "content": post["content"],
                "crawled_at": datetime.now().isoformat(),
                "status": "success",
            })

        # Also add a summary item at the front for quick listing
        items.insert(0, {
            "url": _BOARD_BASE,
            "board": "전체요약",
            "title": f"충남대학교 공지사항 목록 ({date_str})",
            "date": date_str,
            "content": "\n".join(summary_lines),
            "crawled_at": datetime.now().isoformat(),
            "status": "success",
        })

        result = {
            "category": "notice",
            "date": date_str,
            "crawled_at": datetime.now().isoformat(),
            "items": items,
        }

        cat_dir = self.kb_realtime_dir / "notice"
        cat_dir.mkdir(exist_ok=True)
        with open(cat_dir / f"{date_str}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"  [notice] saved {len(items)} items")
        return result

    # ─── Shuttle (static, no crawl) ───────────────────────────────────────────

    def get_shuttle_static(self) -> dict[str, Any]:
        """Shuttle is static — return fixed knowledge base content."""
        content = (
            "충남대학교 2026학년도 셔틀버스 운행 안내\n\n"
            "운영기준: 학기 중 평일 주간 운영 (평일 야간·주말·공휴일·방학 미운영)\n\n"
            "【교내 순환 버스 (대덕캠퍼스 내)】\n"
            "운행시간(오전): 08:20(월평역 등교편 출발), 08:30, 09:30, 09:40\n"
            "운행시간(오후): 10:30, 11:30, 13:30, 14:30, 15:30, 16:30, 17:30\n"
            "1일 총 10회 운행\n"
            "주요 경유 정류장: 정심화 국제문화회관 → 사회과학대학 입구 → 서문(공동실험실습관 앞) "
            "→ 예술대학 앞 → 도서관 앞 → 학생생활관 → 농업생명과학대학 앞 → 동문주차장 → (회차)\n\n"
            "【캠퍼스 순환 버스 (대덕 ↔ 보운)】\n"
            "첫차: 08:10 골프연습장 출발 → 중앙도서관(08:11) → 월평역(08:15) → 보운캠퍼스(08:50 회차)\n"
            "1일 1회 운행\n\n"
            "【월평역 이용 안내】\n"
            "- 캠퍼스 순환(대덕↔보운): 월평역 출발 08:15\n"
            "- 교내 순환(등교편): 월평역 출발 08:20 → 정심화 국제문화회관 하차\n"
        )
        return {
            "category": "shuttle",
            "note": "static_data",
            "content": content,
        }

    # ─── Orchestration ────────────────────────────────────────────────────────

    def crawl_all(self) -> dict[str, Any]:
        print("[crawl] Starting meal crawl...")
        meal_result = self.crawl_meal()

        print("[crawl] Starting notice crawl...")
        notice_result = self.crawl_notice()

        return {
            "meal": meal_result,
            "notice": notice_result,
            "shuttle": self.get_shuttle_static(),
        }

    def run_background(self, interval_hours: int = 1, duration_days: int = 3):
        total_steps = (duration_days * 24) // interval_hours
        print(f"Starting background crawler for {duration_days} days...")
        for step in range(total_steps):
            print(f"[{datetime.now().isoformat()}] Step {step+1}/{total_steps}")
            try:
                self.crawl_all()
                from .indexer import HybridIndexer
                HybridIndexer().build_index()
            except Exception as e:
                print(f"Error: {e}")
            if step < total_steps - 1:
                time.sleep(interval_hours * 3600)


# ─── Realtime meal answer helper (used by chatbot.py, run_chatbot.py, run_realtime.py) ──
def load_today_meal_answer(question: str = "") -> str | None:
    """Return a formatted meal response from today's crawled file, or None if unavailable."""
    from collections import defaultdict
    today = datetime.now().strftime("%Y-%m-%d")
    meal_dir = PROJECT_ROOT / "data" / "kb" / "realtime" / "meal"
    meal_file = meal_dir / f"{today}.json"

    if not meal_file.exists():
        recent = sorted(meal_dir.glob("*.json"), reverse=True)
        if not recent:
            return None
        meal_file = recent[0]
        today = meal_file.stem

    try:
        import json as _json
        data = _json.loads(meal_file.read_text(encoding="utf-8"))
        items = data.get("items", [])
    except Exception:
        return None

    if not items:
        return None

    # Prefer student menus
    student_items = [i for i in items if i.get("staff_student") == "학생"]
    if not student_items:
        student_items = items

    # Optional restaurant filter (기숙사/생활관 → 학생생활관 alias)
    rest_filter = None
    if any(kw in question for kw in ("기숙사", "생활관")):
        rest_filter = "학생생활관"
    else:
        for r in ["제1학생회관", "제2학생회관", "제3학생회관", "제4학생회관", "생활과학대학", "학생생활관"]:
            if r in question:
                rest_filter = r
                break
    if rest_filter:
        filtered = [i for i in student_items if i.get("restaurant") == rest_filter]
        if filtered:
            student_items = filtered

    # Optional meal-type filter
    meal_filter = None
    if "아침" in question or "조식" in question:
        meal_filter = "조식"
    elif "점심" in question or "중식" in question:
        meal_filter = "중식"
    elif "저녁" in question or "석식" in question:
        meal_filter = "석식"

    restaurants: dict = defaultdict(dict)
    for item in student_items:
        rest = item.get("restaurant", "")
        mt = item.get("meal_type", "")
        if not rest or not mt:
            continue
        if meal_filter and mt != meal_filter:
            continue
        restaurants[rest][mt] = item.get("menu_items", [])

    if not restaurants:
        return None

    import re as _re

    def _clean_item(name: str) -> str:
        """알레르기 코드 및 영문 전용 항목 정리."""
        # 끝에 붙은 모든 알레르기 숫자 코드 제거 (예: " 9,15 9", " 1,2,5")
        name = _re.sub(r'(\s+\d+(?:,\d+)*)+\s*$', '', name).strip()
        # 영문 알레르기 태그 제거 (예: "(beef included)", "(chicken included)")
        name = _re.sub(r'\([a-z\s]+included\)', '', name, flags=_re.IGNORECASE).strip()
        return name

    def _is_korean_item(name: str) -> bool:
        """한글 포함 항목만 표시 (영문 번역 라인 제외)."""
        return bool(_re.search(r'[가-힣]', name))

    date_display = datetime.strptime(today, "%Y-%m-%d").strftime("%m월 %d일")
    _weekly_kw = ("이번 주", "이번주", "주간", "weekly", "한 주", "한주")
    if any(kw in question for kw in _weekly_kw):
        header = f"{date_display} 충남대학교 오늘(금일) 학식 안내입니다. (주간 메뉴는 생협 홈페이지에서 확인 가능)\n"
    else:
        header = f"{date_display} 충남대학교 오늘의 학식 안내입니다.\n"
    lines = [header]
    for rest, meals in list(restaurants.items())[:6]:
        lines.append(f"【{rest}】")
        for mt in (["조식", "중식", "석식"] if not meal_filter else [meal_filter]):
            if mt in meals and meals[mt]:
                cleaned = [_clean_item(it) for it in meals[mt] if _is_korean_item(it)]
                if cleaned:
                    lines.append(f"  {mt}: {' / '.join(cleaned[:7])}")
        lines.append("")
    lines.append("운영시간: 아침 07:30~09:00 / 점심 11:30~13:30 / 저녁 17:00~19:00")
    return "\n".join(lines).strip()


# ─── Realtime notice answer helper ────────────────────────────────────────────
def load_today_notice_answer(question: str = "") -> str | None:
    """Return a formatted notice response from today's crawled file, or None if unavailable."""
    today = datetime.now().strftime("%Y-%m-%d")
    notice_dir = PROJECT_ROOT / "data" / "kb" / "realtime" / "notice"
    notice_file = notice_dir / f"{today}.json"
    if not notice_file.exists():
        recent = sorted(notice_dir.glob("*.json"), reverse=True)
        if not recent:
            return None
        notice_file = recent[0]
    try:
        import json as _json
        data = _json.loads(notice_file.read_text(encoding="utf-8"))
        items = data.get("items", [])
        individual = [it for it in items if it.get("title") and "공지사항 목록" not in it.get("title", "")]
        if not individual:
            return None

        today_display = datetime.now().strftime("%Y년 %m월 %d일")
        query_lower = question.lower()
        filter_kw = None
        for kw, label in [("채용", "채용초빙공고"), ("취업", "취업공지"), ("취업공지", "취업공지"), ("장학", "장학공지"), ("학사", "학사공지"), ("행사", "행사문화공지"), ("일반", "일반공지")]:
            if kw in query_lower:
                filter_kw = kw
                break

        lines = [f"{today_display} 기준 충남대학교 최신 공지사항입니다.\n"]
        count = 0
        if filter_kw:
            # 필터 키워드 매칭 항목을 앞으로 정렬
            matched = [it for it in individual if filter_kw in it.get("board", "") or filter_kw in it.get("title", "").lower()]
            unmatched = [it for it in individual if it not in matched]
            sorted_items = matched + unmatched
        else:
            sorted_items = individual
        for it in sorted_items:
            title = it.get("title", "").strip()
            if not title or len(title) < 5:
                continue
            date_str = it.get("date", "")
            board = it.get("board", "")
            entry = f"• [{date_str}] [{board}] {title}" if date_str else f"• [{board}] {title}"
            lines.append(entry)
            count += 1
            if count >= 10:
                break

        if count == 0:
            return None
        lines.append("\n더 많은 공지사항은 충남대학교 홈페이지(www.cnu.ac.kr) > 백마광장 > 공지사항에서 확인하세요.")
        return "\n".join(lines)
    except Exception:
        return None


# ─── Department info crawler ───────────────────────────────────────────────────

def _extract_dept_name(question: str) -> str | None:
    """질문에서 학과명을 추출한다."""
    import re
    # 긴 suffix를 먼저 매칭해야 "학과"가 "학" 앞에 잡힘
    match = re.search(r'([가-힣A-Za-z]+(?:공학부|공학과|학전공|학과|학부|공학|과학|전공|대학))', question)
    if match:
        name = match.group(1)
        # 너무 짧으면 무시
        if len(name) >= 3:
            return name
    return None


def crawl_dept_info(dept_name: str, timeout: int = 8) -> str | None:
    """네이버 검색 + 학과 홈페이지 크롤링으로 충남대 학과 정보를 가져온다."""
    try:
        import re as _re
        naver_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Accept-Language": "ko-KR,ko;q=0.9",
        }

        # 1. 네이버에서 충남대 학과 URL + AI 스니펫 추출
        resp = requests.get(
            "https://search.naver.com/search.naver",
            params={"query": f"충남대학교 {dept_name}"},
            headers=naver_headers,
            timeout=timeout,
        )
        soup = BeautifulSoup(resp.text, "html.parser")

        dept_url: str | None = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if _re.match(r"https://(?!www\.)[a-z]+\.cnu\.ac\.kr", href):
                dept_url = "https://" + href.replace("https://", "").split("/")[0]
                break

        naver_snippet = ""
        full_text = soup.get_text(separator=" ")
        m = _re.search(r"AI 출처 정보(.{50,300}?)네이버가 AI를", full_text)
        if m:
            naver_snippet = m.group(1).strip()

        if not dept_url:
            return None

        # 2. 학과 홈페이지 메인 → 소개/진로 링크 발견
        main_resp = requests.get(dept_url + "/", headers=naver_headers, timeout=timeout)
        main_soup = BeautifulSoup(main_resp.text, "html.parser")

        intro_url: str | None = None
        career_url: str | None = None
        for a in main_soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            full = (dept_url + href) if href.startswith("/") else href
            if not full.startswith("https://"):
                continue
            if "진로" in text and not career_url:
                career_url = full
            elif ("소개" in text or "연혁" in text) and not intro_url:
                intro_url = full

        # 3. 소개 페이지 크롤링
        intro_text = ""
        if intro_url:
            r = requests.get(intro_url, headers=naver_headers, timeout=timeout)
            s = BeautifulSoup(r.text, "html.parser")
            for t in s.find_all(["script", "style", "nav", "header", "footer"]):
                t.decompose()
            paras = [p.get_text(strip=True) for p in s.find_all("p") if len(p.get_text(strip=True)) > 60]
            intro_text = " ".join(paras[:2])[:350]

        # 4. 진로 페이지 크롤링
        career_text = ""
        if career_url:
            r = requests.get(career_url, headers=naver_headers, timeout=timeout)
            s = BeautifulSoup(r.text, "html.parser")
            for t in s.find_all(["script", "style", "nav", "header", "footer"]):
                t.decompose()
            paras = [p.get_text(strip=True) for p in s.find_all("p") if len(p.get_text(strip=True)) > 60]
            if paras:
                career_text = " ".join(paras[:2])[:350]
            else:
                text = s.get_text(separator=" ", strip=True)
                cm = _re.search(r"(졸업.{0,20}?(?:생|자|후).{50,350}?(?:분야|업계|기관|진출))", text)
                if cm:
                    career_text = cm.group(1)[:300]

        if not intro_text and not career_text and not naver_snippet:
            return None

        # 5. 답변 포맷
        parts = [f"【충남대학교 {dept_name}】\n"]
        if intro_text:
            parts.append(f"• 학과 소개\n{intro_text}\n")
        elif naver_snippet:
            parts.append(f"• 소개\n{naver_snippet}\n")
        if career_text:
            parts.append(f"\n• 졸업 후 진로\n{career_text}\n")
        parts.append(f"\n• 공식 홈페이지: {dept_url}")
        return "".join(parts)

    except Exception:
        return None
def load_dept_info_answer(question: str) -> str | None:
    """학과 관련 질문에 대해 KB 또는 크롤링으로 답변을 생성한다."""
    # 먼저 golden KB에서 검색
    dept_name = _extract_dept_name(question)
    if not dept_name:
        return None

    # golden_kb.json에서 해당 학과 항목 찾기 (cnu_dept_ 항목 우선)
    try:
        from .io_utils import PROJECT_ROOT
        kb_path = PROJECT_ROOT / "data" / "kb" / "golden_kb.json"
        with open(kb_path, encoding="utf-8") as f:
            items = json.load(f)

        # 1차: title에 학과명 포함 + cnu_dept_ 접두사 항목 우선
        for item in items:
            item_id = item.get("id", "")
            title = item.get("title", "")
            if dept_name in title and item_id.startswith("cnu_dept_"):
                return item.get("content", "")

        # 2차: title에 학과명 포함 (일반 항목)
        for item in items:
            title = item.get("title", "")
            if dept_name in title:
                return item.get("content", "")
    except Exception:
        pass

    # KB에 없으면 웹 크롤링 시도
    crawled = crawl_dept_info(dept_name)
    if crawled:
        return crawled + f"\n\n더 자세한 정보는 충남대학교 홈페이지(www.cnu.ac.kr) → 대학·학과 소개에서 확인하세요."

    return None


# ─── Notice keyword search (fallback for unanswered questions) ────────────────
def search_notice_for_topic(question: str) -> str | None:
    """Search crawled notice data for question keywords.
    Used as fallback when golden KB has no answer."""
    import re as _re

    today = datetime.now().strftime("%Y-%m-%d")
    notice_dir = PROJECT_ROOT / "data" / "kb" / "realtime" / "notice"
    notice_file = notice_dir / f"{today}.json"
    if not notice_file.exists():
        recent = sorted(notice_dir.glob("*.json"), reverse=True)
        if not recent:
            return None
        notice_file = recent[0]

    try:
        import json as _json
        data = _json.loads(notice_file.read_text(encoding="utf-8"))
        items = data.get("items", [])
        individual = [
            it for it in items
            if it.get("title") and "공지사항 목록" not in it.get("title", "")
        ]
        if not individual:
            return None

        stop = {"는", "을", "이", "가", "의", "에", "에서", "으로", "로", "와", "과",
                "도", "만", "뭐", "언제", "어디", "어떻게", "몇", "알려줘", "알려",
                "줘", "봐", "봐요", "세요", "요", "충남대", "충남대학교", "학교", "대학교"}
        words = _re.findall(r'[가-힣A-Za-z0-9]+', question)
        keywords = [w for w in words if len(w) >= 2 and w not in stop]

        if not keywords:
            return None

        # Score each notice by keyword matches in title (×2) and content (×1)
        scored = []
        for it in individual:
            title = it.get("title", "")
            content = it.get("content", "")
            score = (
                sum(2 for kw in keywords if kw in title) +
                sum(1 for kw in keywords if kw in content)
            )
            if score > 0:
                scored.append((score, it))

        if not scored:
            # 로컬 공지 데이터에 없으면 CNU 포털 직접 검색 시도
            for kw in keywords:
                if len(kw) >= 3:
                    portal_ans = crawl_cnu_notice_for_keyword(kw)
                    if portal_ans:
                        return portal_ans
            return None

        scored.sort(key=lambda x: -x[0])
        top = scored[0][1]

        title = top.get("title", "").strip()
        date_str = top.get("date", "")
        board = top.get("board", "")
        content = top.get("content", "").strip()

        lines = []
        entry = f"• [{date_str}] [{board}] {title}" if date_str else f"• [{board}] {title}"
        lines.append(entry)

        if content:
            best_idx = 0
            for kw in keywords:
                idx = content.find(kw)
                if idx >= 0:
                    best_idx = max(0, idx - 50)
                    break
            snippet = content[best_idx:best_idx + 350].strip()
            if snippet:
                lines.append(f"\n{snippet}")

        lines.append("\n자세한 내용은 충남대학교 공지사항(www.cnu.ac.kr)에서 확인하세요.")
        return "\n".join(lines)

    except Exception:
        return None


# ─── CNU portal notice board keyword search ───────────────────────────────────
def crawl_cnu_notice_for_keyword(keyword: str, timeout: int = 10) -> str | None:
    """CNU 포털 공지 게시판에서 키워드로 직접 검색한다."""
    import re as _re
    try:
        session = requests.Session()
        session.headers.update(_HEADERS)
        session.verify = False

        # 일반공지 + 학사공지 게시판 검색
        results = []
        for board_code, menu_dvs_cd, board_name in [
            ("sub07_0701", "0701", "일반공지"),
            ("sub07_0702", "0702", "학사공지"),
        ]:
            url = (
                f"{_BOARD_BASE}?code={board_code}&site_dvs_cd=kr&menu_dvs_cd={menu_dvs_cd}"
                f"&sval={keyword}"
            )
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            r.encoding = "utf-8"
            soup = BeautifulSoup(r.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"]
                title = a.get_text(strip=True)
                if "mode=V" in href and "no=" in href and len(title) > 5:
                    no_match = _re.search(r'no=(\d+)', href)
                    if no_match:
                        results.append({
                            "board": board_name,
                            "title": title,
                            "no": no_match.group(1),
                        })
            if results:
                break  # 일반공지에서 찾으면 학사공지 생략

        if not results:
            return None

        lines = [f"충남대학교 '{keyword}' 관련 공지사항입니다.\n"]
        for post in results[:3]:
            lines.append(f"• [{post['board']}] {post['title']}")

        lines.append("\n자세한 내용은 충남대학교 공지사항(plus.cnu.ac.kr)에서 확인하세요.")
        return "\n".join(lines)
    except Exception:
        return None


# ─── Naver web search fallback ─────────────────────────────────────────────────
def crawl_naver_for_cnu_query(question: str, timeout: int = 8) -> str | None:
    """Search Naver for CNU-specific answer as last-resort fallback."""
    import re as _re
    try:
        naver_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Accept-Language": "ko-KR,ko;q=0.9",
        }
        clean_q = _re.sub(r'[?？]', '', question).strip()
        if "충남대" not in clean_q:
            search_query = f"충남대학교 {clean_q}"
        else:
            search_query = clean_q

        resp = requests.get(
            "https://search.naver.com/search.naver",
            params={"query": search_query},
            headers=naver_headers,
            timeout=timeout,
        )
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        snippets = []

        # 1. CNU 도메인 링크 + 인접 텍스트
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if "cnu.ac.kr" in href:
                parent = a.find_parent(["li", "div", "section"])
                if parent:
                    text = parent.get_text(separator=" ", strip=True)
                    text = _re.sub(r'\s+', ' ', text).strip()
                    if len(text) > 60 and "충남대" in text:
                        snippets.append(text[:400])
                        break

        # 2. 충남대 + 키워드 포함 스니펫
        if not snippets:
            full_text = _re.sub(r'\s+', ' ', soup.get_text(separator=" "))
            kw_list = [w for w in _re.findall(r'[가-힣]{2,}', clean_q) if len(w) >= 3]
            for kw in kw_list:
                idx = full_text.find(kw)
                while idx >= 0:
                    window = full_text[max(0, idx - 30):idx + 300]
                    if "충남대" in window:
                        snippets.append(window.strip())
                        break
                    idx = full_text.find(kw, idx + 1)
                if snippets:
                    break

        if not snippets:
            return None

        return (
            f"충남대학교 '{clean_q}' 관련 검색 결과입니다.\n\n"
            f"{snippets[0][:450]}\n\n(출처: 실시간 웹 검색)"
        )

    except Exception:
        return None


# ─── General web search (for non-CNU questions) ───────────────────────────────
def search_web_for_general(question: str, timeout: int = 8) -> str | None:
    """일반 질문에 대한 네이버 검색 결과를 컨텍스트 문자열로 반환한다."""
    import re as _re
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Accept-Language": "ko-KR,ko;q=0.9",
        }
        resp = requests.get(
            "https://search.naver.com/search.naver",
            params={"query": question},
            headers=headers,
            timeout=timeout,
        )
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        snippets: list[str] = []

        # 1. 지식 패널 / 인물·장소 정보박스
        for sel in [
            ".sc_kbox_title",           # 지식iN 답변 요약
            ".size_ct_s",               # 뷰 검색결과 본문
            ".api_subject_bx",          # AI 요약
            ".fds-comps-right-image-text-block",
        ]:
            for el in soup.select(sel)[:2]:
                text = _re.sub(r'\s+', ' ', el.get_text(separator=' ', strip=True))
                if len(text) > 40:
                    snippets.append(text[:400])

        # 2. 일반 검색 결과 설명문
        if not snippets:
            for item in soup.select("li.bx, .total_wrap"):
                for cls in [".api_txt_lines", ".sumnail_t", ".text_area", ".dsc_txt_s"]:
                    desc = item.select_one(cls)
                    if desc:
                        text = _re.sub(r'\s+', ' ', desc.get_text(separator=' ', strip=True))
                        if len(text) > 40:
                            snippets.append(text[:300])
                            break
                if len(snippets) >= 3:
                    break

        # 3. 최후 수단: 페이지 전체에서 질문 키워드 주변 텍스트
        if not snippets:
            full = _re.sub(r'\s+', ' ', soup.get_text(separator=' '))
            clean_q = _re.sub(r'[?？]', '', question).strip()
            kw_list = [w for w in _re.findall(r'\S{2,}', clean_q) if len(w) >= 2]
            for kw in kw_list:
                idx = full.find(kw)
                if idx >= 0:
                    window = full[max(0, idx - 30):idx + 300].strip()
                    if len(window) > 50:
                        snippets.append(window)
                        break

        if not snippets:
            return None

        # 중복 제거 후 합치기
        seen: set[str] = set()
        unique: list[str] = []
        for s in snippets:
            key = s[:80]
            if key not in seen:
                seen.add(key)
                unique.append(s)

        return "\n\n".join(unique[:3])

    except Exception:
        return None


# ─── Convenience function (used by run_realtime.py) ────────────────────────────
def build_crawler() -> MultiSourceCrawler:
    return MultiSourceCrawler()


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    crawler = MultiSourceCrawler()
    results = crawler.crawl_all()
    print("Crawl complete.")
    for cat, res in results.items():
        n = len(res.get("items", []))
        print(f"  {cat}: {n} items")
