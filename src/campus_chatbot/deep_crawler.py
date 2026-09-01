import os
import json
import time
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from datetime import datetime
import io
import PyPDF2
from docx import Document

class CNUDeepCrawler:
    def __init__(self, seeds=None, max_depth=2):
        self.seeds = seeds or [
            "https://plus.cnu.ac.kr/html/kr/",
            "https://plus.cnu.ac.kr/html/kr/sub07/sub07_0701.html", # 일반공지
            "https://plus.cnu.ac.kr/html/kr/sub07/sub07_0702.html", # 학사공지
            "https://www.cnu.ac.kr/html/kr/campus/campus_0602.html"  # 통학버스
        ]
        self.domain = "cnu.ac.kr"
        self.max_depth = max_depth
        self.visited = set()
        self.queue = deque([(url, 0) for url in self.seeds])
        self.documents = []
        self.failed_urls = []
        
        # 고도화된 브라우저 헤더
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/"
        }
        
        self.exclude_patterns = [
            "youtube.com", "instagram.com", "facebook.com", "naver.com", "google.com",
            "twitter.com", "linkedin.com", "login", "logout", "privacy", "sitemap", "javascript:"
        ]

    def is_valid_url(self, url):
        if not url: return False
        parsed = urlparse(url)
        if not parsed.netloc.endswith(self.domain):
            return False
        if any(p in url.lower() for p in self.exclude_patterns):
            return False
        if re.search(r'\.(jpg|jpeg|png|gif|zip|7z|mp4|avi|exe|css|js)$', url.lower()):
            return False
        return True

    def clean_text(self, soup):
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.extract()
        
        # 특정 레이아웃 요소 제거
        for div in soup.find_all(["div", "section"], class_=re.compile(r'menu|footer|nav|header|top|side|banner|popup|privacy|quick|breadcrumb')):
            div.extract()

        # 본문 영역 탐색 (충남대 사이트 특성 반영)
        content_area = soup.select_one("#content, #contents, .content, .sub_content, article")
        if content_area:
            return content_area.get_text(strip=True, separator=" ")
        
        return soup.get_text(strip=True, separator=" ")

    def extract_pdf(self, content):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                t = page.extract_text()
                if t: text += t + " "
            return text.strip()
        except: return ""

    def get_breadcrumb(self, soup):
        bc = soup.select(".breadcrumb, .location, .path, .navi")
        if bc: return bc[0].get_text(strip=True, separator=" > ")
        return ""

    def crawl(self, limit=50):
        print(f"🚀 Starting Deep Crawl with {len(self.seeds)} seeds...")
        count = 0
        
        session = requests.Session()
        session.headers.update(self.headers)

        while self.queue and count < limit:
            url, depth = self.queue.popleft()
            if url in self.visited or depth > self.max_depth:
                continue
            
            self.visited.add(url)
            print(f"[{count+1}] ({depth}) Visiting: {url}")
            
            try:
                # SSL 검증 건너뛰고 타임아웃 넉넉히 (충남대 서버 응답 변동성 대응)
                response = session.get(url, timeout=15, verify=False)
                if response.status_code != 200:
                    self.failed_urls.append(url)
                    continue

                content_type = response.headers.get('Content-Type', '').lower()
                
                doc_item = {
                    "id": f"crawl_{datetime.now().strftime('%H%M%S')}_{count}",
                    "url": url,
                    "title": "",
                    "content": "",
                    "breadcrumb": "",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "metadata": {"source": url, "kb_layer": "static", "is_realtime": False}
                }

                if 'html' in content_type:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    doc_item["title"] = soup.title.string.strip() if soup.title else url
                    doc_item["content"] = self.clean_text(soup)
                    doc_item["breadcrumb"] = self.get_breadcrumb(soup)
                    
                    # 링크 추출 (BFS)
                    for link in soup.find_all('a', href=True):
                        full_url = urljoin(url, link['href']).split('#')[0].rstrip('/')
                        if self.is_valid_url(full_url) and full_url not in self.visited:
                            self.queue.append((full_url, depth + 1))
                            
                elif 'pdf' in content_type:
                    doc_item["title"] = os.path.basename(url)
                    doc_item["content"] = self.extract_pdf(response.content)
                
                if len(doc_item["content"]) > 150:
                    self.documents.append(doc_item)
                    count += 1
                
                time.sleep(0.1)

            except Exception as e:
                print(f"❌ Failed: {url} ({e})")
                self.failed_urls.append(url)

        self.save_results()
        self.report()

    def save_results(self):
        output_path = "data/kb/all_structured_docs.json"
        if not self.documents: return

        unique_docs_dict = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                    for d in existing:
                        if 'url' in d: unique_docs_dict[d['url']] = d
            except: pass
        
        for d in self.documents:
            if 'url' in d: unique_docs_dict[d['url']] = d
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(list(unique_docs_dict.values()), f, ensure_ascii=False, indent=2)

    def report(self):
        print("\n" + "="*50)
        print("📊 Deep Crawling Final Report")
        print(f"Total Visited: {len(self.visited)}")
        print(f"Total Saved Docs: {len(self.documents)}")
        print(f"Total Failed: {len(self.failed_urls)}")
        print("="*50)

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    crawler = CNUDeepCrawler(max_depth=2)
    crawler.crawl(limit=50)
