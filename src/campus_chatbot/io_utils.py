from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def candidate_input_paths(filename: str) -> list[Path]:
    return [
        Path("/data") / filename,
        PROJECT_ROOT / "data" / filename,
        Path.cwd() / "data" / filename,
        PROJECT_ROOT / filename,
        Path.cwd() / filename,
    ]


def resolve_input_path(filename: str, required: bool = True) -> Path | None:
    for path in candidate_input_paths(filename):
        if path.exists():
            return path
    if required:
        tried = ", ".join(str(p) for p in candidate_input_paths(filename))
        raise FileNotFoundError(f"{filename} not found. Tried: {tried}")
    return None


def output_dirs() -> list[Path]:
    dirs = [PROJECT_ROOT / "outputs", Path.cwd() / "outputs"]
    if os.access("/", os.W_OK):
        dirs.insert(0, Path("/outputs"))
    unique: list[Path] = []
    for path in dirs:
        if path not in unique:
            unique.append(path)
    return unique


def write_output_json(filename: str, data: Any) -> Path:
    written: list[Path] = []
    for directory in output_dirs():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / filename
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written.append(path)
        except OSError:
            continue
    if not written:
        raise OSError(f"Could not write {filename} to any outputs directory.")
    return written[0]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_question_like(data: Any) -> Iterable[tuple[Any, str]]:
    if isinstance(data, list):
        for item in data:
            yield from _iter_question_like(item)
        return

    if isinstance(data, dict):
        for key in ("data", "questions", "items", "test", "examples"):
            value = data.get(key)
            if isinstance(value, list):
                yield from _iter_question_like(value)
                return

        for key in ("question", "user", "text", "query", "input", "utterance"):
            value = data.get(key)
            if isinstance(value, str):
                yield data, value
                return
            if isinstance(value, list):
                yield from _iter_question_like(value)
                return

        if all(isinstance(v, str) for v in data.values()):
            for value in data.values():
                yield data, value
        return

    if isinstance(data, str):
        yield data, data


def load_questions(filename: str, required: bool = True) -> list[str]:
    path = resolve_input_path(filename, required=required)
    if path is None:
        return []
    questions = [question.strip() for _, question in _iter_question_like(load_json(path))]
    return [question for question in questions if question]
