from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from .io_utils import PROJECT_ROOT
from .labels import DB_FILENAMES

class HybridIndexer:
    """Integrated indexer for Static KB and Realtime crawled data."""

    def __init__(self, static_kb_dir: Path | None = None, realtime_kb_dir: Path | None = None):
        self.static_kb_dir = static_kb_dir or PROJECT_ROOT / "data" / "kb"
        self.realtime_kb_dir = realtime_kb_dir or PROJECT_ROOT / "data" / "kb" / "realtime"
        
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device="cpu",
        )
        self.chroma_client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma_db"))
        self.collection = self.chroma_client.get_or_create_collection(
            name="campus_kb",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def build_index(self):
        """Index both layers, with realtime overriding static via upsert."""
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)

        print("🛠️  Building Knowledge Base Index...")
        # 1. Index Static KB (Specific DBs)
        self._index_layer(self.static_kb_dir, kb_layer="static", is_realtime=False)

        # 2. Index Legacy all_docs.json (to capture anything missed)
        self._index_all_docs()

        # 3. Index Golden KB (authoritative curated docs — must override static)
        self._index_golden_layer()

        # 4. Index Realtime KB (if exists, overrides everything)
        if self.realtime_kb_dir.exists():
            self._index_realtime_layer()

    def _index_all_docs(self):
        path = self.static_kb_dir / "all_docs.json"
        if not path.exists():
            return

        from .labels import CATEGORY_KEYS

        docs = json.loads(path.read_text(encoding="utf-8"))
        ids = []
        documents = []
        metadatas = []
        seen_ids: set[str] = set()

        for i, doc in enumerate(docs):
            cat_id = doc.get("category_id")
            if cat_id is None:
                continue

            category = CATEGORY_KEYS.get(cat_id)
            if not category:
                continue

            content = self._clean_content(doc.get("content", ""))
            if len(content) < 50:
                continue

            doc_id = str(doc.get("id", f"legacy_{hash(doc.get('url', '') + str(i))}"))
            # Resolve duplicates within this batch
            if doc_id in seen_ids:
                doc_id = f"{doc_id}_{i}"
            seen_ids.add(doc_id)

            ids.append(doc_id)
            documents.append(f"{doc.get('title', '')}\n{content}")
            metadatas.append({
                "doc_id": doc_id,
                "category": category,
                "is_realtime": False,
                "kb_layer": "legacy",
                "date": doc.get("date", "unknown"),
            })

        if ids:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # 카테고리별 관련 키워드 – 이 중 하나도 없으면 해당 문서는 제외
    _CATEGORY_KEYWORDS: dict[str, list[str]] = {
        "cafeteria":  ["식단", "메뉴", "점심", "저녁", "아침", "식당", "학식", "kcal", "식사", "조식", "중식", "석식"],
        "shuttle":    ["버스", "운행", "노선", "정류장", "셔틀", "시간표", "월평", "보운"],
        "schedule":   ["학기", "수강신청", "수강정정", "개강", "종강", "방학", "졸업", "학사일정", "시험", "성적"],
        "graduation": ["졸업", "학점", "이수", "전공필수", "교양필수", "학점인정"],
        "notice":     [],  # 공지사항은 키워드 필터 없음
    }

    # 실시간 디렉토리명 → ChromaDB category 매핑
    _REALTIME_DIR_TO_CATEGORY: dict[str, str] = {
        "meal":   "cafeteria",
        "shuttle": "shuttle",
        "notice":  "notice",
    }

    # 네비게이션 노이즈 패턴 – 이 패턴이 콘텐츠의 상당 부분이면 제외
    _NAV_NOISE_PATTERNS = re.compile(
        r"주요메뉴 바로가기|서브메뉴 바로가기|통합검색|사이트맵 CNU|"
        r"Search Search All|Library Catalog|Alphabetical List|"
        r"THE STRONG CNU|CNU 홍보브로슈어|SNS 및 프린트|facebook 카카오톡",
        re.IGNORECASE,
    )

    def _is_nav_noise(self, content: str) -> bool:
        """Returns True if the content is mostly navigation menu boilerplate."""
        noise_hits = len(self._NAV_NOISE_PATTERNS.findall(content))
        # More than 3 nav markers in 1000 chars → almost certainly a nav page
        density = noise_hits / max(len(content) / 1000, 1)
        return density > 2.5

    def _has_category_keywords(self, content: str, category: str) -> bool:
        required = self._CATEGORY_KEYWORDS.get(category, [])
        if not required:
            return True
        return any(kw in content for kw in required)

    def _clean_content(self, text: str) -> str:
        noise_patterns = [
            r"본문 바로가기", r"주메뉴 바로가기", r"서브메뉴 바로가기",
            r"SNS 및 프린트 URL복사", r"facebook 카카오톡 Naver print",
            r"이 사이트는 자바스크립트를 지원하지 않으면",
            r"페이지 관리자 \| 관리자메일",
            r"Search Search All All Library Catalog",
            r"Home > 대학생활 >",
            r"통합검색 열기 버튼 통합검색 검색 통합검색 닫기버튼",
            r"ENG 통합검색",
            r"THE STRONG CNU CNU 홍보브로슈어.*?(?=\n|$)",
            r"주요메뉴 바로가기.*?(?=\n|$)",
        ]
        for p in noise_patterns:
            text = re.sub(p, "", text, flags=re.DOTALL)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _index_golden_layer(self):
        path = self.static_kb_dir / "golden_kb.json"
        if not path.exists():
            return
        docs = json.loads(path.read_text(encoding="utf-8"))
        from .labels import CATEGORY_KEYS
        ids, documents, metadatas = [], [], []
        seen_ids: set[str] = set()
        for i, doc in enumerate(docs):
            # Support both {id, category_id, ...} and {title, content} structures
            raw_id = doc.get("id") or doc.get("doc_id")
            if raw_id:
                doc_id = str(raw_id)
            else:
                safe = re.sub(r'\W+', '_', doc.get("title", ""))[:30]
                doc_id = f"golden_{i}_{safe}"
            if doc_id in seen_ids:
                doc_id = f"{doc_id}_{i}"
            seen_ids.add(doc_id)

            cat_id = doc.get("category_id")
            category = CATEGORY_KEYS.get(cat_id, "general") if cat_id is not None else "general"
            content = self._clean_content(doc.get("content", ""))
            if len(content) < 20:
                continue

            ids.append(doc_id)
            documents.append(f"{doc.get('title', '')}\n{content}")
            metadatas.append({
                "doc_id": doc_id,
                "category": category,
                "kb_layer": "golden",
                "is_realtime": False,
                "date": doc.get("date", "2026-06-08"),
            })
        if ids:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def _index_layer(self, kb_dir: Path, kb_layer: str, is_realtime: bool):
        for category, filename in DB_FILENAMES.items():
            path = kb_dir / filename
            if not path.exists():
                continue
            
            docs = json.loads(path.read_text(encoding="utf-8"))
            ids = []
            documents = []
            metadatas = []
            
            seen_ids = set()
            for i, doc in enumerate(docs):
                url = doc.get('url', '')
                title = doc.get('title', '')
                
                # Senior Level Pruning & Prioritization
                if "library.cnu.ac.kr" in url:
                    # Libraries are rarely the primary source for academic/graduation info
                    if category != "notice": continue
                    # For notices, only keep if it looks like a major university notice
                    if "board" not in url: continue
                
                if any(d in url for d in ["cicnu.ac.kr", "health.cnu.ac.kr", "startup.cnu.ac.kr"]):
                    continue

                # Detect and prune Homepages
                if url.strip("/") in ["https://plus.cnu.ac.kr/html/kr", "https://www.cnu.ac.kr/html/kr", "https://plus.cnu.ac.kr"]:
                    continue

                # Senior Level: Key Document Identification (ID Boosting)
                doc_id = str(doc.get('id', f"{category}_{i}"))
                if "academic_calendar" in url:
                    doc_id = "OFFICIAL_CALENDAR_2026"
                    title = "2026학년도 충남대학교 공식 학사일정"
                elif "sub05_051202" in url:
                    doc_id = "OFFICIAL_GRADUATION_REQUIREMENTS"
                    title = "충남대학교 공식 졸업이수학점 및 졸업요건"
                
                # Extreme Noise Filter for Schedule
                if category == "schedule" and any(x in title for x in ["설명회", "모집", "모집안내", "갤러리"]):
                    continue

                content = self._clean_content(doc.get('content', ''))
                if len(content) < 150:
                    continue

                # Skip navigation-only pages (boilerplate menus)
                if self._is_nav_noise(content):
                    continue

                # Require category-relevant keywords
                if not self._has_category_keywords(content, category):
                    continue

                if doc_id in seen_ids:
                    doc_id = f"{doc_id}_{i}" 
                seen_ids.add(doc_id)
                
                ids.append(doc_id)
                documents.append(f"{doc.get('title', '')}\n{content}")
                
                meta = doc.get("metadata", {}).copy()
                meta.update({
                    "doc_id": doc_id,
                    "category": category,
                    "is_realtime": is_realtime,
                    "kb_layer": kb_layer,
                    "date": doc.get("date", "unknown")
                })
                clean_meta = {k: v for k, v in meta.items() if isinstance(v, (str, bool, int, float))}
                metadatas.append(clean_meta)

            if ids:
                self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def _index_realtime_layer(self):
        for category_dir in self.realtime_kb_dir.iterdir():
            if not category_dir.is_dir():
                continue

            # Map directory name (e.g. "meal") to ChromaDB category (e.g. "cafeteria")
            category = self._REALTIME_DIR_TO_CATEGORY.get(category_dir.name, category_dir.name)
            for json_file in category_dir.glob("*.json"):
                data = json.loads(json_file.read_text(encoding="utf-8"))
                items = data.get("items", []) # Corrected from merged_items to items
                
                ids = []
                documents = []
                metadatas = []
                
                for i, item in enumerate(items):
                    # Use a unique ID based on title/date to allow upserts
                    safe_title = re.sub(r'\W+', '', item.get('title', ''))[:20]
                    date_info = item.get('crawled_at', 'today').split('T')[0]
                    doc_id = f"realtime_{category}_{date_info}_{safe_title}_{i}"
                    
                    ids.append(doc_id)
                    documents.append(f"{item.get('title', '')}\n{item.get('content', '')}")
                    metadatas.append({
                        "doc_id": doc_id,
                        "category": category,
                        "is_realtime": True,
                        "kb_layer": "realtime",
                        "date": date_info,
                        "source_url": item.get('url', 'unknown')
                    })

                if ids:
                    self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

if __name__ == "__main__":
    import re
    indexer = HybridIndexer()
    indexer.build_index()
    print("Hybrid indexing complete.")
