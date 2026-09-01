from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(rel: str) -> Path:
    """rel 경로를 프로젝트 루트 기준 절대 경로로 변환. CWD에 무관하게 동작."""
    p = Path(rel)
    if p.is_absolute():
        return p
    # io_utils candidate 순서와 동일
    for candidate in [
        _PROJECT_ROOT / rel,
        Path.cwd() / rel,
    ]:
        if candidate.exists():
            return candidate
    return _PROJECT_ROOT / rel  # 존재 안 해도 PROJECT_ROOT 기준 반환


def build_chat_output(input_path: str = "data/test_chat.json",
                      output_path: str = "outputs/chat_output.json") -> None:
    """Generate chat_output.json using static KB + LLM.

    No real-time crawling: uses the pre-indexed ChromaDB (static KB).
    Works for any question — the classifier routes to the correct KB category,
    the retriever fetches relevant docs, and the LLM generates the answer.
    """
    src = _resolve(input_path)
    if not src.exists():
        print(f"⚠️  {src} not found. Skipping chat_output.json.")
        return

    data = json.loads(src.read_text(encoding="utf-8"))
    questions: list[str] = [
        item.get("user") or item.get("question", "")
        for item in data
        if item.get("user") or item.get("question")
    ]

    print(f"[chat] 골든 벤치마크 + 크롤 데이터 + LLM으로 {len(questions)}개 질문 처리 중...")
    from campus_chatbot.chatbot import CampusChatBot
    bot = CampusChatBot()

    results: list[dict] = []
    for q in questions:
        print(f"  Q: {q[:70]}")
        answer = bot.answer(q)
        results.append({"user": q, "model": answer})

    dst = _resolve(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote {len(results)} entries to {dst}")


def main() -> None:
    # Task 3 first: crawl → index → realtime_output
    # (must run before chat_output so _check_benchmarks() can use today's meal/notice data)
    from campus_chatbot.run_realtime import run_realtime_pipeline
    run_realtime_pipeline(
        str(_resolve("data/test_realtime.json")),
        str(_PROJECT_ROOT / "outputs" / "realtime_output.json"),
    )

    # Task 2: chat_output — golden benchmarks + real-time meal/notice data + LLM
    build_chat_output(
        str(_resolve("data/test_chat.json")),
        str(_PROJECT_ROOT / "outputs" / "chat_output.json"),
    )


if __name__ == "__main__":
    main()
