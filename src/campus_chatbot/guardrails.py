from __future__ import annotations

import re

from .labels import CATEGORY_KEYWORDS


GREETING_PATTERNS = (
    "안녕",
    "안녕하세요",
    "하이",
    "hello",
    "hi",
    "반가워",
    "고마워",
    "감사",
    "땡큐",
    "잘가",
)

HELP_PATTERNS = (
    "뭐 할 수",
    "무엇을 할 수",
    "기능",
    "도움말",
    "사용법",
    "help",
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_greeting_or_help(question: str) -> bool:
    norm = normalize(question)
    compact = norm.replace(" ", "")
    if any(pattern in compact for pattern in GREETING_PATTERNS):
        return True
    return any(pattern in norm for pattern in HELP_PATTERNS)


def has_domain_signal(question: str) -> bool:
    norm = normalize(question)
    compact = norm.replace(" ", "")
    for keywords in CATEGORY_KEYWORDS.values():
        for keyword in keywords:
            key = normalize(keyword).replace(" ", "")
            if key and key in compact:
                return True
    return False


def canned_response(question: str) -> str | None:
    if is_greeting_or_help(question):
        return (
            "안녕하세요. 충남대학교 캠퍼스 챗봇입니다. "
            "졸업요건, 학교 공지사항, 학사일정, 식단 안내, 통학/셔틀 버스에 대해 질문해 주세요."
        )
    if not has_domain_signal(question):
        return (
            "이 챗봇은 충남대학교 졸업요건, 공지사항, 학사일정, 식단, 통학/셔틀 버스 질문에 답변하도록 만들어졌습니다. "
            "해당 주제로 다시 질문해 주세요."
        )
    return None

