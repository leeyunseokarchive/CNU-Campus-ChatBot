from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event

from .io_utils import PROJECT_ROOT

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

_WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]


def _today_str() -> str:
    now = datetime.now()
    return f"{now.year}년 {now.month}월 {now.day}일({_WEEKDAY_KO[now.weekday()]})"


def _today_day_ko() -> str:
    return _WEEKDAY_KO[datetime.now().weekday()]


@dataclass
class GenerationConfig:
    model_id: str = DEFAULT_MODEL_ID
    max_new_tokens: int = 192
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 3


class CampusLLMGenerator:
    """Speed-optimized LLM Generator with Korean-only guardrails and streaming."""

    # Class-level cache: model is loaded once per process and reused across instances
    _cached_tokenizer = None
    _cached_model = None

    def __init__(self, config: GenerationConfig | None = None):
        self.config = config or GenerationConfig(
            model_id=os.environ.get("CAMPUS_LLM_MODEL", DEFAULT_MODEL_ID),
        )
        self._tokenizer = None
        self._model = None

    # ─── System Prompts (strict Korean-only) ──────────────────────────────────
    def _get_system_prompts(self) -> dict[str, str]:
        today = _today_str()
        day_ko = _today_day_ko()
        return {
            "0": (
                "당신은 충남대학교 졸업요건 안내 챗봇입니다.\n"
                "⚠️ 반드시 한국어로만 답변하세요. 중국어·일본어·영어는 한 글자도 사용하지 마세요.\n"
                "⚠️ [참고정보]에 있는 내용만 사용하세요. 없는 내용을 지어내지 마세요.\n\n"
                "답변 규칙:\n"
                "1. 첫 문장: '졸업을 위해서는 일반적으로 130학점이 필요합니다.'로 시작\n"
                "2. [참고정보]에서 관련 내용을 찾아 2~3문장으로 요약\n"
                "3. 마지막: 'CNU With U+ 포털의 졸업자가진단을 확인하세요.' 추가\n"
            ),
            "1": (
                "당신은 충남대학교 공지사항 안내 챗봇입니다.\n"
                "⚠️ 반드시 한국어로만 답변하세요. 중국어·일본어·영어는 한 글자도 사용하지 마세요.\n"
                "⚠️ [참고정보]에 실제로 나온 공지사항 제목, 날짜, 게시판명만 인용하세요.\n"
                "⚠️ [참고정보]에 없는 공지사항을 절대 만들어내지 마세요.\n\n"
                "답변 형식: '[날짜]에 [제목]이 [게시판]에 게시되었습니다.' 형식으로 답하세요.\n"
                "HTML 태그, 사이트맵, 네비게이션 메뉴 텍스트는 무시하세요.\n"
            ),
            "2": (
                "당신은 충남대학교 학사일정 안내 챗봇입니다.\n"
                "⚠️ 반드시 한국어로만 답변하세요. 중국어·일본어·영어는 한 글자도 사용하지 마세요.\n"
                "⚠️ [참고정보]에 있는 날짜와 일정만 정확하게 인용하세요.\n"
                "⚠️ [참고정보]에 없는 날짜나 일정을 절대 만들어내지 마세요.\n\n"
                "답변 형식: '가장 가까운 학사일정은 MM.DD(요일) [내용]입니다.' 형식으로 답하세요.\n"
            ),
            "3": (
                f"당신은 충남대학교 학식 안내 챗봇입니다.\n"
                f"⚠️ 반드시 한국어로만 답변하세요. 중국어·일본어·영어는 한 글자도 사용하지 마세요.\n"
                f"⚠️ [참고정보]에서 오늘({today}, {day_ko}요일) 식단 정보를 찾아 답하세요.\n"
                f"⚠️ 오늘 메뉴가 [참고정보]에 없으면 반드시 아래 형식으로만 답하세요:\n"
                f"'오늘 충남대학교 학식 안내입니다. 식사 시간: 아침 07:30~09:00 / 점심 11:30~13:30 / 저녁 17:00~19:00. 오늘의 상세 메뉴는 학생회관 게시판에서 확인하세요.'\n\n"
                f"메뉴가 있을 경우 답변 형식:\n"
                f"오늘 충남대학교 학식 안내입니다.\n"
                f"아침: [메뉴 또는 '정보 없음']\n"
                f"점심: [메뉴 또는 '정보 없음']\n"
                f"저녁: [메뉴 또는 '정보 없음']\n"
                f"같은 메뉴를 절대 두 번 쓰지 마세요.\n"
            ),
            "4": (
                "당신은 충남대학교 셔틀버스 및 교통편 안내 챗봇입니다.\n"
                "⚠️ 반드시 한국어로만 답변하세요. 중국어·일본어·영어는 한 글자도 사용하지 마세요.\n"
                "⚠️ [참고정보]에서 관련 시간표·노선 정보를 찾아 정확히 인용하세요.\n"
                "⚠️ [참고정보]에 없는 시간이나 노선을 만들어내지 마세요.\n\n"
                "핵심 정보:\n"
                "- 교내 순환: 08:30~17:30 (1일 10회), 월평역 등교편 08:20\n"
                "- 캠퍼스 순환(대덕↔보운): 08:10 골프연습장 출발, 월평역 08:15 경유\n"
                "- 정류장 업데이트 질문: '새로 업데이트된 정류장은 없습니다.'라고 답하세요.\n"
                "- 시내버스 정문: 101번, 102번, 105번, 106번, 113번, 114번, 115번, 121번, 704번\n"
                "- 시내버스 교내진입: 48번, 108번\n"
            ),
        }

    # ─── Model Loading ─────────────────────────────────────────────────────────
    def _load(self) -> None:
        # Reuse already-loaded model within the same process
        if CampusLLMGenerator._cached_model is not None:
            self._tokenizer = CampusLLMGenerator._cached_tokenizer
            self._model = CampusLLMGenerator._cached_model
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        use_cuda = torch.cuda.is_available()
        kwargs: dict = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if use_cuda:
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **kwargs)
        if not use_cuda:
            self._model.eval()
        CampusLLMGenerator._cached_tokenizer = self._tokenizer
        CampusLLMGenerator._cached_model = self._model

    # ─── Prompt Builder ────────────────────────────────────────────────────────
    def _build_messages(self, system: str, context: str, question: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"[참고정보]\n{context}\n\n"
                    f"[질문]\n{question}\n\n"
                    f"⚠️ 규칙:\n"
                    f"1. 반드시 한국어로만 답변하세요. 중국어·영어·일본어 금지.\n"
                    f"2. 위 [참고정보]에 있는 내용만 사용하세요.\n"
                    f"3. [참고정보]에 없는 정보는 절대 만들지 마세요.\n"
                    f"4. 대학 이름은 반드시 '충남대학교'만 사용하세요.\n"
                    f"5. 정보가 부족하면 '충남대학교 홈페이지(www.cnu.ac.kr)를 확인하세요'라고 안내하세요."
                ),
            },
        ]

    # ─── Language Sanitizer ────────────────────────────────────────────────────
    def _sanitize_language(self, text: str) -> str:
        """Remove Chinese/Japanese characters that Qwen sometimes generates."""
        cleaned = []
        cjk_count = 0
        total_chars = sum(1 for c in text if not c.isspace())

        for ch in text:
            cp = ord(ch)
            # CJK Unified Ideographs (Chinese/Japanese kanji) – remove
            if 0x4E00 <= cp <= 0x9FFF:
                cjk_count += 1
                continue
            # CJK Extension A
            if 0x3400 <= cp <= 0x4DBF:
                cjk_count += 1
                continue
            # CJK Extension B
            if 0x20000 <= cp <= 0x2A6DF:
                cjk_count += 1
                continue
            # Katakana (Japanese ア~ン)
            if 0x30A0 <= cp <= 0x30FF:
                cjk_count += 1
                continue
            # Hiragana (Japanese あ~ん)
            if 0x3040 <= cp <= 0x309F:
                cjk_count += 1
                continue
            cleaned.append(ch)

        result = ''.join(cleaned)
        result = re.sub(r' {2,}', ' ', result).strip()

        # If more than 15% garbage characters, the whole output is unreliable
        if total_chars > 0 and cjk_count / total_chars > 0.15:
            return ""

        return result

    # ─── Post-Processing ───────────────────────────────────────────────────────
    def _deduplicate_lines(self, text: str) -> str:
        lines = text.split('\n')
        result = []
        prev = None
        for line in lines:
            s = line.strip()
            if s and s == prev:
                continue
            result.append(line)
            if s:
                prev = s

        counts = Counter(l.strip() for l in result if l.strip())
        seen: set[str] = set()
        deduped = []
        for line in result:
            s = line.strip()
            if counts.get(s, 0) >= 3:
                if s in seen:
                    continue
                seen.add(s)
            deduped.append(line)
        return '\n'.join(deduped)

    def _detect_hallucination(self, text: str, category: str) -> bool:
        """True if text contains known hallucination signatures."""
        # Wrong university name
        wrong_uni_patterns = [
            "충청북도대학교", "충청북도립대학교", "충청북도립", "충북대학교",
            "충남대 대학교", "충남 대학교", "공주대학교", "한남대학교", "배재대학교",
            "konbukta", "konbuk", ".edu", "chungbuk",
        ]
        for pat in wrong_uni_patterns:
            if pat in text and "충남대학교" not in text:
                return True

        # Obvious junk patterns: MCQ format, celebratory fillers, off-topic quiz
        junk_patterns = [
            r"축하합니다",                            # celebration noise
            r"[A-D]\.\s+\d+월",                      # MCQ answer format
            r"선택\s*문제",                            # quiz format
            r"답안\s*:",                              # MCQ answer marker
            r"[A-D]\.\s+[가-힣]",                  # MCQ option letter
            r"청계천",                                 # Seoul landmark
            r"관광객",                                 # tourist context (wrong)
            r"(이 방법|이 서비스).{0,30}국제",           # international noise
            r"\d+%\s*(선택|응답|비율)",               # poll/survey format
            r"신안산|보수원|청계|한강|강남|강북|종로",           # Seoul locations in Daejeon context
            r"(Line|line|호선).{0,20}연결",                    # subway line info (wrong)
            r"관광객.{0,30}방문",                              # tourist language
            r"해당.{0,20}데이터베이스.{0,20}없습니다.{0,30}www\.",  # wrong DB language with URL
        ]
        for p in junk_patterns:
            if re.search(p, text, re.IGNORECASE):
                return True

        # English junk mixed with Korean (not URLs or abbreviations)
        english_words = re.findall(r'\b[A-Za-z]{6,}\b', text)
        english_words = [w for w in english_words if w.lower() not in
                         ("course", "credit", "campus", "portal", "master", "doctor",
                          "chungnam", "national", "university", "schedule")]
        if len(english_words) >= 3:
            return True

        # Garbage academic terminology from model's English training
        halluc_patterns = [
            r'academic quarter', r'Cal-리포트', r'분기.*대한 정보',
            r'\bquarter\b', r'시즌 \d', r'국제문화.*회관.*\d{4}', r'근처이나.*오랜만',
        ]
        for p in halluc_patterns:
            if re.search(p, text, re.IGNORECASE):
                return True

        # Schedule category: must contain a Korean date if it claims to give schedule info
        if category == "2":
            has_date = bool(re.search(r'\d{4}년\s*\d{1,2}월|\d{1,2}월\s*\d{1,2}일|\d{2}\.\d{2}', text))
            has_garbage = bool(re.search(r'[A-Za-z]{5,}|시즌|새 정 |분기', text))
            if not has_date and has_garbage:
                return True

        # Output too short to be useful (< 15 chars of Korean)
        korean_chars = sum(1 for c in text if '가' <= c <= '힣')
        if korean_chars < 15:
            return True

        return False

    def _post_process(self, text: str, category: str = "") -> str:
        # Strip filler phrases
        text = re.sub(r"제공된\s*(문서|컨텍스트|자료)에\s*(따르면|의하면|서는)?", "", text)
        text = re.sub(r"안녕하세요[,! ]*", "", text)
        text = re.sub(r"죄송합니다[,.]?\s*현재 실시간 데이터를 제공하는 기능이 없습니다[.!]?\s*", "", text)
        text = re.sub(r"죄송합니다[,.]?\s*하지만 ", "", text)
        text = re.sub(r"\[참고정보\]", "", text)
        text = re.sub(r"\[질문\]", "", text)
        text = re.sub(r"위 \[참고정보\]만 사용해서 한국어로만 답변하세요\.", "", text)
        # Strip prompt leak fragments
        text = re.sub(r"⚠️\s*규칙:.*", "", text, flags=re.DOTALL)
        text = re.sub(r"\d+\.\s*(반드시|위 \[참고|대학 이름|정보가 부족)", "", text)

        # Strip language-mixing artefacts
        text = self._sanitize_language(text)

        # Deduplicate repeated lines
        text = self._deduplicate_lines(text)

        # Hallucination detection — clear output if hallucination found
        if self._detect_hallucination(text, category):
            text = ""

        # Category-specific enforcement
        if category == "0":
            if "130학점" not in text:
                text = "졸업을 위해서는 일반적으로 130학점이 필요합니다. " + text
            # Truncate after the mandatory closing line to prevent extended hallucination
            closing = "졸업자가진단을 확인하세요."
            idx = text.find(closing)
            if idx != -1:
                text = text[:idx + len(closing)]
        elif category == "3":
            if text and "학식" not in text and "식단" not in text and "메뉴" not in text:
                text = ("오늘 충남대학교 학식 안내입니다.\n"
                        "식사 시간: 아침 07:30~09:00 / 점심 11:30~13:30 / 저녁 17:00~19:00\n"
                        "오늘의 상세 메뉴는 학생회관 게시판 또는 충남대학교 생협 홈페이지에서 확인하세요.")
        elif category == "4":
            # Detect shuttle hallucination: English route names, foreign city names
            english_words = re.findall(r'\b[A-Za-z]{5,}\b', text)
            korean_blocks = re.findall(r'[가-힣]{2,}', text)
            if len(english_words) > 2 or (english_words and len(korean_blocks) < len(english_words) * 2):
                text = ""  # trigger fallback
            elif text and "업데이트" in text and "정류장" in text and "없습니다" not in text:
                text = "새로 업데이트된 정류장은 없습니다."

        # Final cleanup: if sanitizer returned empty (too much CJK), return fallback
        if not text.strip():
            fallbacks = {
                "0": "졸업을 위해서는 일반적으로 130학점이 필요합니다. 자세한 사항은 CNU With U+ 포털의 졸업자가진단을 확인하세요.",
                "1": "최신 공지사항은 충남대학교 홈페이지(www.cnu.ac.kr) 공지사항 게시판에서 확인하세요.",
                "2": (
                    "주요 학사일정 안내입니다.\n"
                    "- 기말고사: 6월 10일(수) ~ 6월 18일(목)\n"
                    "- 1학기 종강: 6월 20일(금)\n"
                    "- 하기 계절학기: 6월 22일(월) ~ 7월 10일(금)\n"
                    "- 1학기 성적발표: 7월 10일(금)\n"
                    "- 2학기 수강신청: 8월 3일(월) ~ 8월 7일(금)\n"
                    "- 2학기 개강: 9월 1일(화)\n"
                    "자세한 일정은 학사포털(plus.cnu.ac.kr)에서 확인하세요."
                ),
                "3": "오늘 충남대학교 학식 안내입니다.\n식사 시간: 아침 07:30~09:00 / 점심 11:30~13:30 / 저녁 17:00~19:00\n오늘의 메뉴는 학생회관 게시판에서 확인하세요.",
                "4": (
                    "충남대학교 셔틀버스 안내입니다.\n\n"
                    "【교내 순환】 08:30~17:30, 1일 10회 (월평역 등교편 08:20)\n"
                    "노선: 정심화국제문화회관 → 사회과학대학 → 서문 → 예술대학 → 도서관 → 농업생명과학대학 → 동문주차장\n\n"
                    "【캠퍼스 순환(대덕↔보운)】 08:10 출발, 1일 1회\n"
                    "노선: 골프연습장 → 중앙도서관 → 월평역(08:15) → 보운캠퍼스\n\n"
                    "【시내버스】 정문: 101·102·105·106·113·114·115·121·704번 / 교내: 48·108번\n\n"
                    "운영: 학기 중 평일 주간 (주말·공휴일·방학 미운행)\n"
                    "자세한 시간표: plus.cnu.ac.kr"
                ),
            }
            return fallbacks.get(category, "충남대학교 관련 문의는 학교 홈페이지(www.cnu.ac.kr) 또는 학과 사무실에 문의하세요.")

        return text.strip()

    # ─── Generation (batch) ────────────────────────────────────────────────────
    def generate(self, question: str, category: int | str, context: str) -> str:
        self._load()
        import torch
        cat_str = str(category)
        sys_prompt = self._get_system_prompts().get(cat_str, (
                "당신은 충남대학교(Chungnam National University, CNU) 캠퍼스 안내 챗봇입니다.\n"
                "⚠️ 반드시 한국어로만 답변하세요. 중국어·영어·일본어 금지.\n"
                "⚠️ 이 챗봇은 충남대학교 전용입니다. '충남대학교' 외 다른 대학교 이름은 절대 사용하지 마세요.\n"
                "⚠️ [참고정보]에 없는 내용은 만들지 말고, 충남대학교 홈페이지(www.cnu.ac.kr)를 안내하세요.\n"
                "⚠️ 가짜 URL이나 외부 웹사이트 주소를 절대 만들지 마세요.\n"
                "정보가 없으면: '해당 정보는 충남대학교 홈페이지(www.cnu.ac.kr) 또는 학과 사무실에 문의하세요.'로 답하세요."
            ))
        messages = self._build_messages(sys_prompt, context, question)
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                repetition_penalty=self.config.repetition_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (exclude the prompt)
        prompt_len = inputs["input_ids"].shape[1]
        raw = self._tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        return self._post_process(raw, cat_str)

    # ─── General Conversation (web search → LLM summarize) ────────────────────
    _GENERAL_SYSTEM_WITH_WEB = (
        "당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 답변하세요.\n"
        "⚠️ 아래 [검색 결과]에 있는 내용만 사용하세요. 없는 내용은 절대 만들지 마세요.\n"
        "⚠️ [검색 결과]가 충분하지 않으면 '정확한 정보를 찾지 못했습니다. 직접 검색해 확인하세요.'라고 답하세요.\n"
        "⚠️ 가짜 URL·출처를 만들지 마세요.\n"
    )

    _GENERAL_SYSTEM_NO_WEB = (
        "당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 답변하세요.\n"
        "⚠️ 확실히 아는 것만 답하세요. 불확실하면 '잘 모르겠습니다. 직접 검색해 확인하세요.'라고 답하세요.\n"
        "⚠️ 절대 추측하지 마세요. 특히 고유명사(나라 수도, 날짜, 인물명)를 잘못 말하면 안 됩니다.\n"
        "⚠️ 가짜 URL·출처를 만들지 마세요.\n"
    )

    def _post_process_general(self, text: str, has_web_context: bool = False) -> str:
        """일반 대화 응답 후처리."""
        text = self._sanitize_language(text)
        text = self._deduplicate_lines(text)
        text = re.sub(r"⚠️\s*규칙:.*", "", text, flags=re.DOTALL)
        text = re.sub(r"\[검색 결과\].*", "", text, flags=re.DOTALL)
        text = re.sub(r"\[질문\].*", "", text, flags=re.DOTALL)
        text = text.strip()

        korean_chars = sum(1 for c in text if '가' <= c <= '힣')
        if len(text) > 20 and korean_chars < 5:
            return "죄송합니다, 답변 생성에 문제가 발생했습니다. 질문을 다시 해주세요."

        if not text:
            return "죄송합니다, 답변을 드리기 어렵습니다. 질문을 다시 해주세요."

        # 웹 검색 없이 생성된 경우 면책 문구 추가
        if not has_web_context:
            text += "\n\n(AI 생성 응답입니다. 중요한 정보는 직접 검색해 확인하세요.)"

        return text

    # 사실 조회형 질문 패턴 (웹 검색 트리거)
    _FACTUAL_PATTERNS = re.compile(
        r"수도|뜻이|뜻은|언제|어디서|어디에|어디 있|누가|몇 명|몇명|인구|면적|위치|역사|"
        r"언제야|몇 년|몇년도|나이|설립|창립|발명|발견|최초|기원|출신|국적|"
        r"어떤 나라|어느 나라|무슨 나라|수도는|가격은|얼마야|몇 층"
    )

    def _needs_web_search(self, question: str) -> bool:
        """웹 검색이 필요한 사실형 질문인지 판별한다."""
        return bool(self._FACTUAL_PATTERNS.search(question))

    def generate_general(self, question: str) -> str:
        """일반 대화 질문에 답변.
        사실형 질문: 웹 검색 결과를 컨텍스트로 사용 → 할루시네이션 억제.
        생성/절차형 질문(레시피·코드·번역 등): LLM 단독 + 면책 문구.
        """
        self._load()
        import torch

        # 사실형 질문만 웹 검색 시도
        web_context: str | None = None
        if self._needs_web_search(question):
            try:
                from .realtime import search_web_for_general
                web_context = search_web_for_general(question)
            except Exception:
                pass

        if web_context:
            messages = [
                {"role": "system", "content": self._GENERAL_SYSTEM_WITH_WEB},
                {
                    "role": "user",
                    "content": (
                        f"[검색 결과]\n{web_context}\n\n"
                        f"[질문]\n{question}\n\n"
                        "위 [검색 결과]를 바탕으로 간결하게 한국어로 답하세요."
                    ),
                },
            ]
        else:
            messages = [
                {"role": "system", "content": self._GENERAL_SYSTEM_NO_WEB},
                {"role": "user", "content": question},
            ]

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        raw = self._tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return self._post_process_general(raw, has_web_context=bool(web_context))

    # ─── Streaming Generation ──────────────────────────────────────────────────
    def generate_stream(self, question: str, category: int | str, context: str,
                        cancel_event: Event | None = None):
        # Run synchronously to avoid MPS thread crash on macOS (SIGSEGV from nested threads).
        # Yield output in small chunks so the SSE client receives tokens progressively.
        if str(category) == "general":
            response = self.generate_general(question)
        else:
            response = self.generate(question, category, context)
        chunk_size = 5
        for i in range(0, len(response), chunk_size):
            if cancel_event and cancel_event.is_set():
                return
            yield response[i:i + chunk_size]
