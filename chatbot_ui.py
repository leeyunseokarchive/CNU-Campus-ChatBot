"""CNU Campus Chatbot UI — FastAPI + Demo.html (Colab / 로컬 공통)"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from campus_chatbot.chatbot import CampusChatBot

app = FastAPI()
_bot_state: dict = {"bot": None, "ready": False}


def _prewarm() -> None:
    try:
        bot = CampusChatBot()
        bot.generator._load()
        _bot_state["bot"] = bot
        _bot_state["ready"] = True
    except Exception as e:
        print(f"초기화 오류: {e}")
        _bot_state["ready"] = True


threading.Thread(target=_prewarm, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(ROOT / "Demo.html")


@app.get("/api/chat")
def chat_sse(question: str = ""):
    def generate():
        if not question.strip():
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
            return

        for _ in range(240):
            if _bot_state["bot"] is not None:
                break
            time.sleep(0.5)

        if _bot_state["bot"] is None:
            msg = "AI 초기화에 실패했습니다. 페이지를 새로고침해 주세요."
            yield f"data: {json.dumps({'token': msg, 'done': True})}\n\n"
            return

        bot = _bot_state["bot"]
        try:
            bot.cancel()
        except Exception:
            pass

        try:
            for chunk in bot.answer_stream(question):
                yield f"data: {json.dumps({'token': chunk, 'done': False})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'token': f'오류: {e}', 'done': True})}\n\n"
            return

        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _get_colab_url(port: int) -> str | None:
    """Colab 프록시 URL 반환. 노트북 셀 컨텍스트에서만 동작."""
    try:
        from google.colab.output import eval_js
        url = eval_js(f"google.colab.kernel.proxyPort({port})")
        return url if url else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-port", type=int, default=7860)
    # chatbot.sh에서 백그라운드 기동 시 URL 출력을 억제할 때 사용
    parser.add_argument("--quiet", action="store_true",
                        help="URL 출력 억제 (chatbot.sh가 대신 출력)")
    args = parser.parse_args()

    _in_colab = os.path.isdir("/content")

    if not args.quiet:
        if _in_colab:
            url = _get_colab_url(args.server_port)
            if url:
                print(f"\n✅ 챗봇 UI 접속 URL: {url}\n")
            else:
                print(f"\n✅ UI 시작됨 (포트 {args.server_port})")
                print("접속 URL을 얻으려면 새 셀에서:")
                print(f"  from google.colab.output import eval_js")
                print(f"  print(eval_js('google.colab.kernel.proxyPort({args.server_port})'))\n")
        else:
            print(f"UI 접속: http://localhost:{args.server_port}")

    uvicorn.run(app, host="0.0.0.0", port=args.server_port, log_level="warning")


if __name__ == "__main__":
    main()
