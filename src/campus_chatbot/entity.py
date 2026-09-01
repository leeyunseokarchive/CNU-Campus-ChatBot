from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .labels import CATEGORY_KEYS


DEPARTMENT_ALIASES = {
    "건축학과": ["건축학과", "건축", "건축과"],
    "의예과": ["의예과", "의예", "의과대학 의예과"],
    "의학과": ["의학과", "의대", "의과대학 의학과"],
    "수의예과": ["수의예과", "수의예"],
    "수의학과": ["수의학과", "수의대", "수의과대학"],
    "간호학과": ["간호학과", "간호대", "간호대학"],
    "약학과": ["약학과", "약대", "약학대학"],
    "컴퓨터융합학부": ["컴퓨터융합학부", "컴융", "컴퓨터", "컴퓨터공학", "컴퓨터공학과"],
}

BUILDING_ALIASES = {
    "제1학생회관": ["제1학생회관", "1학생회관", "1학", "1학식", "제1학식"],
    "제2학생회관": ["제2학생회관", "2학생회관", "2학", "2학식", "제2학식"],
    "제3학생회관": ["제3학생회관", "3학생회관", "3학", "3학식", "제3학식"],
    "제4학생회관": ["제4학생회관", "4학생회관", "4학", "4학식", "제4학식"],
    "생활과학대학": ["생활과학대학", "생과대", "생활과학대", "생활대"],
}

STOP_ALIASES = {
    "정문": ["정문", "충남대학교입구", "충남대입구", "학교입구"],
    "유성온천역": ["유성온천역", "유성온천"],
    "월평역": ["월평역", "월평"],
    "중앙도서관": ["중앙도서관", "충남대도서관", "도서관 정류장"],
    "충대농대종점": ["충대농대", "충대농대종점", "농대종점"],
    "산학연교육연구관": ["산학연", "산학연교육연구관"],
}

MEAL_ALIASES = {
    "조식": ["조식", "아침"],
    "중식": ["중식", "점심", "런치"],
    "석식": ["석식", "저녁", "저녁밥"],
}


@dataclass(frozen=True)
class QueryEntities:
    category: str
    department: str | None = None
    building: str | None = None
    stop: str | None = None
    year: str | None = None
    date: str | None = None
    meal: str | None = None
    major_type: str | None = None
    raw: dict[str, str] = field(default_factory=dict)

    def metadata_filter(self) -> dict[str, str]:
        filters: dict[str, str] = {"category": self.category}
        if self.category in {"graduation", "notice"} and self.department:
            filters["department"] = self.department
        if self.category == "cafeteria" and self.building:
            filters["building"] = self.building
        if self.category == "cafeteria" and self.meal:
            filters["meal"] = self.meal
        if self.category == "shuttle" and self.stop:
            filters["stop"] = self.stop
        if self.category == "graduation" and self.major_type:
            filters["major_type"] = self.major_type
        if self.category == "graduation" and not self.department and not self.major_type:
            filters["department"] = "공통"
        if self.year:
            filters["year"] = self.year
        if self.date:
            filters["date"] = self.date
        return filters

    def primary_entity(self) -> str | None:
        return self.department or self.building or self.stop


def _find_alias(text: str, alias_map: dict[str, list[str]]) -> str | None:
    compact = text.replace(" ", "")
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            if alias.replace(" ", "") in compact:
                return canonical
    return None


def _extract_year(text: str) -> str | None:
    match = re.search(r"(20\d{2})", text)
    if match:
        return match.group(1)
    match = re.search(r"(\d{2})\s*학번", text)
    if match:
        yy = int(match.group(1))
        return f"20{yy:02d}" if yy < 50 else f"19{yy:02d}"
    return None


def _extract_date(text: str) -> str | None:
    today = datetime.now()
    if any(key in text for key in ("오늘", "금일")):
        return today.strftime("%Y-%m-%d")
    if "내일" in text:
        # Avoid importing timedelta in the common path until needed.
        from datetime import timedelta

        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    match = re.search(r"(20\d{2})[.\-/년 ]+(\d{1,2})[.\-/월 ]+(\d{1,2})", text)
    if match:
        year, month, day = match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return None


def extract_entities(question: str, label: int) -> QueryEntities:
    category = CATEGORY_KEYS[label]
    department = _find_alias(question, DEPARTMENT_ALIASES) if category in {"graduation", "notice"} else None
    building = _find_alias(question, BUILDING_ALIASES) if category == "cafeteria" else None
    stop = _find_alias(question, STOP_ALIASES) if category == "shuttle" else None
    meal = _find_alias(question, MEAL_ALIASES) if category == "cafeteria" else None
    compact = question.replace(" ", "")
    major_type = None
    if category == "graduation":
        if any(key in compact for key in ("복전", "복수전공")):
            major_type = "복수전공"
        elif "부전공" in compact:
            major_type = "부전공"
    year = _extract_year(question)
    date = _extract_date(question) if category == "cafeteria" else None

    raw = {}
    if department:
        raw["department"] = department
    if building:
        raw["building"] = building
    if stop:
        raw["stop"] = stop
    if meal:
        raw["meal"] = meal
    if year:
        raw["year"] = year
    if date:
        raw["date"] = date
    if major_type:
        raw["major_type"] = major_type

    return QueryEntities(
        category=category,
        department=department,
        building=building,
        stop=stop,
        year=year,
        date=date,
        meal=meal,
        major_type=major_type,
        raw=raw,
    )
