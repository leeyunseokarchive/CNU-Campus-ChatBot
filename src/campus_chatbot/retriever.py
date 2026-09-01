from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions
from .io_utils import PROJECT_ROOT
from .labels import CATEGORY_KEYS, CATEGORY_TO_LABEL, DB_FILENAMES

@dataclass(frozen=True)
class KnowledgeDoc:
    id: str
    category: str
    title: str
    content: str
    source: str
    metadata: dict[str, Any]

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.content}"

@dataclass(frozen=True)
class RetrievalHit:
    doc: KnowledgeDoc
    score: float

class KnowledgeRetriever:
    """Improved Hybrid Retriever using ChromaDB (Dense) and BM25 (Sparse) with RRF."""

    def __init__(self, kb_dir: Path | None = None, collection: Any | None = None):
        self.kb_dir = kb_dir or PROJECT_ROOT / "data" / "kb"
        
        # Initialize Embedding Function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device="cpu",
        )
        
        # Initialize ChromaDB (Persistent)
        self.chroma_client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma_db"))
        if collection:
            self.collection = collection
        else:
            self.collection = self.chroma_client.get_or_create_collection(
                name="campus_kb",
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
        
        # Load docs directly from ChromaDB to ensure consistency
        self.docs = self._load_docs_from_chroma()
        
        # Build Indices
        self._build_indices()

    def _load_docs_from_chroma(self) -> list[KnowledgeDoc]:
        all_data = self.collection.get()
        docs = []
        for i in range(len(all_data["ids"])):
            meta = all_data["metadatas"][i]
            content_full = all_data["documents"][i]
            parts = content_full.split("\n", 1)
            title = parts[0]
            content = parts[1] if len(parts) > 1 else ""
            
            docs.append(KnowledgeDoc(
                id=all_data["ids"][i],
                category=meta.get("category", "notice"),
                title=title,
                content=content,
                source=meta.get("source", ""),
                metadata=meta
            ))
        return docs

    def _build_indices(self):
        # 1. BM25 Indices
        self.bm25_indices = {}
        self.category_docs = {}
        for category_key in CATEGORY_KEYS.values():
            cat_docs = [d for d in self.docs if d.category == category_key]
            if cat_docs:
                tokenized_corpus = [self._tokenize(d.text) for d in cat_docs]
                self.bm25_indices[category_key] = BM25Okapi(tokenized_corpus)
                self.category_docs[category_key] = cat_docs

    def _tokenize(self, text: str) -> list[str]:
        # Improved regex tokenizer for Korean/English/Numbers
        return re.findall(r'[가-힣0-9A-Za-z]{2,}', text.lower())

    def rrf(self, dense_results: list[str], sparse_results: list[str], k: int = 60) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion with slight bias towards dense for better semantic matching."""
        scores = {}
        for rank, doc_id in enumerate(dense_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.2 / (k + rank) # 1.2 weight for dense
        for rank, doc_id in enumerate(sparse_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def search(
        self,
        query: str,
        category: int | str,
        top_k: int = 3,
    ) -> list[RetrievalHit]:
        category_key = CATEGORY_KEYS[category] if isinstance(category, int) else category
        
        # 0. Authority Keyword Priority Map (maps keywords → golden KB doc IDs)
        priority_map = {
            "수강신청": "OFFICIAL_CALENDAR_2026",
            "개강": "OFFICIAL_CALENDAR_2026",
            "종강": "OFFICIAL_CALENDAR_2026",
            "학사일정": "OFFICIAL_CALENDAR_2026",
            "졸업": "OFFICIAL_GRADUATION_REQUIREMENTS",
            "학점": "OFFICIAL_GRADUATION_REQUIREMENTS",
            "이수": "OFFICIAL_GRADUATION_REQUIREMENTS",
            "셔틀": "OFFICIAL_SHUTTLE_SCHEDULE_2026",
            "셔틀버스": "OFFICIAL_SHUTTLE_SCHEDULE_2026",
            "월평역": "OFFICIAL_SHUTTLE_SCHEDULE_2026",
            "통학버스": "OFFICIAL_SHUTTLE_SCHEDULE_2026",
            "버스 시간": "OFFICIAL_SHUTTLE_SCHEDULE_2026",
            "업데이트": "OFFICIAL_SHUTTLE_UPDATE",
            "최신 공지": "OFFICIAL_NOTICE_2026_JUNE",
            "최신공지": "OFFICIAL_NOTICE_2026_JUNE",
            "공지사항": "OFFICIAL_NOTICE_2026_JUNE",
            "채용 공고": "OFFICIAL_NOTICE_2026_JUNE",
            "최근 공지": "OFFICIAL_NOTICE_2026_JUNE",
        }
        
        doc_map = {d.id: d for d in self.docs}
        boosted_ids = []
        for kw, target_id in priority_map.items():
            if kw in query and target_id in doc_map:
                boosted_ids.append(target_id)

        # 1. Tier 1: Title Boost (Exact Match)
        query_terms = self._tokenize(query)
        for doc in self.docs:
            if doc.category == category_key:
                match_count = sum(1 for term in query_terms if term in doc.title.lower())
                if match_count >= 2:
                    boosted_ids.append(doc.id)
        
        # 1. Dense Retrieval (ChromaDB)
        dense_res = self.collection.query(
            query_texts=[query],
            n_results=15,
            where={"category": category_key}
        )
        dense_ids = dense_res["ids"][0]
        dense_distances = dense_res["distances"][0] if dense_res["distances"] else [0.0]*len(dense_ids)
        
        # 2. Sparse Retrieval (BM25)
        sparse_ids = []
        if category_key in self.bm25_indices:
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25_indices[category_key].get_scores(tokenized_query)
            top_n_idx = np.argsort(bm25_scores)[::-1][:15]
            sparse_ids = [self.category_docs[category_key][i].id for i in top_n_idx if bm25_scores[i] > 0]

        # 3. Hybrid RRF with Title Boost
        scores = {}
        # Apply RRF
        for rank, doc_id in enumerate(dense_ids):
            scores[doc_id] = scores.get(doc_id, 0) + 1.2 / (60 + rank)
        for rank, doc_id in enumerate(sparse_ids):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (60 + rank)
        
        # Apply Boost
        for doc_id in boosted_ids:
            if doc_id in scores:
                scores[doc_id] *= 2.0 # Double score if title matched
            else:
                scores[doc_id] = 0.05 # Add to hits even if not in top 15
                
        fused_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 4. Map back to KnowledgeDoc and return RetrievalHit
        hits = []
        doc_map = {d.id: d for d in self.docs}
        for doc_id, score in fused_results[:top_k]:
            hits.append(RetrievalHit(doc=doc_map[doc_id], score=score))
            
        # 5. Semantic Guard (Threshold)
        if hits:
            top_doc_id = hits[0].doc.id
            if top_doc_id in boosted_ids:
                return hits

            if top_doc_id in dense_ids:
                idx = dense_ids.index(top_doc_id)
                similarity = 1 - dense_distances[idx]
                if similarity < 0.10:
                    return []
            elif not (dense_ids or sparse_ids):
                return []

        return hits
