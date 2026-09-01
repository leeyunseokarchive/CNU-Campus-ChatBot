# CNU Campus ChatBot — 충남대학교 AI 캠퍼스 어시스턴트

충남대학교 재학생을 위한 **RAG 기반 지능형 캠퍼스 챗봇**입니다.
졸업요건, 공지사항, 학사일정, 식단, 셔틀버스 등 캠퍼스 생활 질문에 대해
질문 분류 → 하이브리드 검색 → LLM 답변 생성 파이프라인으로 응답합니다.

> 🎥 데모 영상: [`Termproject_UI영상_이윤석.mov`](./Termproject_UI영상_이윤석.mov) · 발표자료: [`Termproject_발표자료_이윤석.pdf`](./Termproject_발표자료_이윤석.pdf)

---

## 주요 기능

| Task | 설명 |
|------|------|
| **1. 질문 분류기** | `koelectra-base-v3` 파인튜닝 + 키워드 앙상블(Soft Voting)로 질문을 5개 카테고리(졸업요건 / 공지사항 / 학사일정 / 식단 / 셔틀버스)로 분류 — 검증 정확도 **99.3%** |
| **2. RAG 챗봇** | ChromaDB(Dense) + BM25(Sparse) 하이브리드 검색을 RRF로 결합해 지식베이스에서 근거를 검색하고, `Qwen2.5-1.5B-Instruct`로 답변 생성 |
| **3. 실시간 정보 반영** | BeautifulSoup 크롤러가 학교 공지·식단을 주기적으로 수집해 ChromaDB에 실시간 upsert |
| **UI** | FastAPI + SSE 토큰 스트리밍 챗봇 웹 UI |

## 아키텍처

```
사용자 질문
    │
    ▼
[질문 분류기] koelectra-base-v3 파인튜닝 + 키워드 앙상블
    │  (카테고리: graduation / notice / schedule / cafeteria / shuttle)
    ▼
[하이브리드 검색] ChromaDB (Dense 임베딩) + BM25 (Sparse) → RRF 랭킹 융합
    │            + 실시간 크롤링 데이터 upsert (notice / meal)
    ▼
[가드레일] 도메인 외 질문 필터링, 엔티티(날짜·식당 등) 추출
    ▼
[LLM 생성] Qwen/Qwen2.5-1.5B-Instruct → 근거 기반 답변
    ▼
[FastAPI + SSE] 토큰 단위 스트리밍 UI
```

## 기술 스택

- **분류기**: `monologg/koelectra-base-v3-discriminator` 파인튜닝 (HuggingFace Transformers)
- **검색(RAG)**: ChromaDB + sentence-transformers 임베딩, rank-bm25, RRF(Reciprocal Rank Fusion)
- **LLM**: `Qwen/Qwen2.5-1.5B-Instruct`
- **크롤링**: requests + BeautifulSoup4 (공지사항·식단 실시간 수집)
- **서빙**: FastAPI + uvicorn, SSE 스트리밍, HTML/CSS 프론트엔드

## 실행 방법

```bash
# 1. 가상 환경 생성 및 패키지 설치
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 전체 파이프라인 실행 (KB 인덱싱 → 출력 생성 → UI 서버)
bash chatbot.sh
# → http://localhost:7860 접속
```

> **참고**: 파인튜닝된 분류기 가중치(`model.safetensors`)는 용량 문제로 리포지토리에 포함되어 있지 않습니다.
> `src/classifier.ipynb`를 실행하면 `data/train_cls.json`으로 동일한 모델을 재학습해 `model/classifier_finetuned/`에 저장할 수 있습니다.
> (학습 하이퍼파라미터와 결과는 `model/classifier_finetuned/training_metadata.json` 참고)

## 디렉토리 구조

```
.
├── data/
│   ├── train_cls.json          # 분류기 학습 데이터 (605 rows)
│   ├── test_*.json             # 태스크별 평가 데이터
│   └── kb/                     # 지식베이스 JSON (졸업요건·공지·일정·식단·셔틀)
│       └── realtime/           # 실시간 크롤링 결과 (notice / meal)
├── src/
│   ├── classifier.ipynb        # Task 1: 분류기 학습 노트북
│   ├── run_classifier.py       # 분류기 배치 실행
│   ├── run_chatbot.py          # Task 2·3: 챗봇 파이프라인 배치 실행
│   └── campus_chatbot/         # 핵심 모듈
│       ├── classifier_model.py #   분류기 로드·앙상블
│       ├── retriever.py        #   ChromaDB + BM25 하이브리드 검색 (RRF)
│       ├── indexer.py          #   지식베이스 인덱싱
│       ├── llm_generator.py    #   Qwen LLM 답변 생성
│       ├── guardrails.py       #   도메인 외 질문 필터링
│       ├── entity.py           #   날짜·식당 등 엔티티 추출
│       ├── smart_crawler.py    #   실시간 공지·식단 크롤러
│       └── realtime.py         #   실시간 데이터 upsert 파이프라인
├── model/
│   └── classifier_finetuned/   # 파인튜닝 모델 설정·토크나이저 (가중치는 재학습 필요)
├── scripts/rebuild_kb.py       # 지식베이스 재구축
├── chatbot_ui.py               # FastAPI + SSE 챗봇 UI 서버
├── Demo.html                   # 챗봇 프론트엔드
├── chatbot.sh                  # 통합 실행 스크립트
└── requirements.txt
```

## 분류기 성능

| 항목 | 값 |
|------|-----|
| Base model | `monologg/koelectra-base-v3-discriminator` |
| 학습 / 검증 데이터 | 605 / 152 rows |
| Epochs | 5 |
| **검증 정확도** | **99.34%** |
| Eval loss | 0.234 |
