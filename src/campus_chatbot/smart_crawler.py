import os
import json
import time
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
from collections import deque
from datetime import datetime
import io
import PyPDF2
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class SmartAlphaCrawler:
    """초정밀 도메인 분류 및 무제한 게시판 탐색 기능을 갖춘 시니어 레벨 크롤러"""

    def __init__(self, seeds=None):
        self.seeds = seeds or [
            "https://plus.cnu.ac.kr/html/kr/sub05/sub05_051202.html", # 졸업요건 시작
            "https://plus.cnu.ac.kr/html/kr/sub07/sub07_0701.html",   # 일반공지
            "https://plus.cnu.ac.kr/html/kr/sub07/sub07_0702.html",   # 학사공지
            "https://m.cnu.ac.kr/html/kr/sub05/sub05_050401.html",    # 식단
            "https://m.cnu.ac.kr/html/kr/sub05/sub05_050403.html"     # 셔틀
        ]
        self.domain = "cnu.ac.kr"
        self.visited = set()
        self.queue = deque(self.seeds)
        self.documents = []
        self.lock = Lock()
        
        # 도메인별 핵심 키워드 및 가중치 (0: 졸업, 1: 공지, 2: 일정, 3: 식단, 4: 셔틀)
        self.category_specs = {
            0: {"id": "graduation", "keywords": ["graduation", "졸업", "수료", "이수학점", "졸업요건", "sub05_0512"], "priority": 10},
            3: {"id": "cafeteria", "keywords": ["dorm", "cafeteria", "meal", "식단", "메뉴", "food", "sub05_050401", "학생회관"], "priority": 10},
            4: {"id": "shuttle", "keywords": ["shuttle", "bus", "셔틀", "버스", "노선", "운행", "campus_0602", "sub05_050403"], "priority": 10},
            2: {"id": "schedule", "keywords": ["academic_calendar", "schedule", "일정", "학사일정", "기간", "calendar"], "priority": 5},
            1: {"id": "notice", "keywords": ["notice", "공지", "bbs", "board", "post"], "priority": 1} # 폴백 성격
        }

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.verify = False

    def classify_robust(self, url, title, breadcrumb, content):
        """다중 요소를 고려한 초정밀 가중치 분류"""
        scores = {cat_id: 0 for cat_id in self.category_specs.keys()}
        full_text = f"{url.lower()} {title.lower()} {breadcrumb.lower()} {content[:500].lower()}"
        
        for cat_id, spec in self.category_specs.items():
            # URL에 키워드 포함 시 압도적 가중치
            if any(k in url.lower() for k in spec["keywords"]):
                scores[cat_id] += 100
            
            # 메타데이터(제목, 브레드크럼) 가중치
            meta_text = f"{title.lower()} {breadcrumb.lower()}"
            if any(k in meta_text for k in spec["keywords"]):
                scores[cat_id] += 50
                
            # 본문 키워드 빈도수 가중치
            match_count = sum(1 for k in spec["keywords"] if k in content[:1000].lower())
            scores[cat_id] += match_count * 2
            
            # 기본 우선순위 반영
            scores[cat_id] += spec["priority"]

        # 가장 높은 점수의 카테고리 반환
        best_cat = max(scores, key=scores.get)
        return best_cat

    def clean_html(self, soup):
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.extract()
        
        # 충남대 본문 영역 정밀 타격
        main = soup.select_one("#content, #contents, .content, .sub_content, article, .board_list, .ann_table, .meal_menu")
        if main:
            return main.get_text(strip=True, separator=" ")
        return soup.get_text(strip=True, separator=" ")

    def extract_links(self, soup, current_url):
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('javascript:') or href.startswith('#'): continue
            
            full_url = urljoin(current_url, href).split('#')[0].rstrip('/')
            
            # 도메인 체크 및 불필요한 링크(로그인 등) 제외
            if self.domain in full_url and "login" not in full_url.lower():
                links.append(full_url)
        return links

    def process_url(self, url):
        with self.lock:
            if url in self.visited: return []
            self.visited.add(url)

        try:
            res = self.session.get(url, timeout=15)
            if res.status_code != 200: return []
            
            soup = BeautifulSoup(res.content, 'html.parser')
            title = soup.title.string.strip() if soup.title else url
            breadcrumb = " > ".join([bt.get_text(strip=True) for bt in soup.select(".breadcrumb span, .location a")])
            content = self.clean_html(soup)
            
            if len(content) > 150:
                # 고도화된 분류 엔진 호출
                category_id = self.classify_robust(url, title, breadcrumb, content)
                
                doc = {
                    "id": f"robust_{hash(url)}",
                    "url": url,
                    "title": title,
                    "content": content,
                    "category_id": category_id,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "metadata": {"source": url, "breadcrumb": breadcrumb}
                }
                with self.lock:
                    self.documents.append(doc)
            
            return self.extract_links(soup, url)

        except Exception:
            return []

    def run(self, max_workers=10, limit=300):
        print(f"🚀 [Senior Dev] Starting Robust Smart Crawler (Deep & Accurate Mode)...")
        
        processed_count = 0
        while self.queue and processed_count < limit:
            batch = []
            while self.queue and len(batch) < max_workers:
                batch.append(self.queue.popleft())
            
            if not batch: break
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.process_url, url): url for url in batch}
                for future in as_completed(futures):
                    new_links = future.result()
                    for link in new_links:
                        if link not in self.visited:
                            self.queue.append(link)
                    processed_count += 1
                    
                    if processed_count % 20 == 0:
                        print(f"📈 Progress: {processed_count}/{limit} URLs. Docs: {len(self.documents)}")

        self.save_and_distribute()

    def save_and_distribute(self):
        print("📦 Finalizing: Atomic distribution by 5 categories...")
        db_mapping = {
            0: "graduation_db.json",
            1: "notice_db.json",
            2: "schedule_db.json",
            3: "cafeteria_db.json",
            4: "shuttle_db.json"
        }
        
        distributed = {i: [] for i in db_mapping.keys()}
        for doc in self.documents:
            distributed[doc['category_id']].append(doc)
            
        for cat_id, items in distributed.items():
            if not items: continue
            
            filename = db_mapping[cat_id]
            path = os.path.join("data/kb", filename)
            
            existing = []
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    try: existing = json.load(f)
                    except: pass
            
            # URL 기준 고유 문서 유지 (신규 크롤링 데이터 우선)
            merged = {d.get('url'): d for d in existing}
            for it in items:
                merged[it['url']] = it
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(list(merged.values()), f, ensure_ascii=False, indent=2)
            print(f"✅ [Category {cat_id}] {filename} updated. Total docs: {len(merged)}")

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    crawler = SmartAlphaCrawler()
    # 전수 수집을 위해 한도를 300으로 상향
    crawler.run(max_workers=15, limit=300)
