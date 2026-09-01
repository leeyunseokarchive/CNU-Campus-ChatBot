#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"
PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

PORT=7860

# Python 감지 (venv → python3 → python)
if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "Python을 찾을 수 없습니다." >&2; exit 1
fi

echo "===== CampusChatBot 시작 ====="
mkdir -p outputs

# 1. 라이브러리 설치 + Drive 마운트 (Colab 환경)
if [ -d "/content" ]; then
    echo "[1/3] 라이브러리 설치 중..."
    pip install -q -r requirements.txt 2>/dev/null || true
    echo "[1/3] 라이브러리 설치 완료"

    echo "[1/3] Google Drive 마운트 중..."
    $PYTHON -c "
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
print('✅ Google Drive 마운트 완료')
" 2>/dev/null || echo "⚠️  Drive 마운트 실패 (무시하고 계속)"
fi

# 2. chat_output.json / realtime_output.json 생성
echo "[2/3] chat_output.json, realtime_output.json 생성 중..."
$PYTHON src/run_chatbot.py 2>/dev/null || echo "⚠️  출력 생성 중 오류 발생 (계속 진행)"
echo "[2/3] 출력 파일 생성 완료"

# 3. UI 실행
echo "[3/3] UI 시작 중..."

if [ -d "/content" ]; then
    # ── Colab 환경 ──────────────────────────────────────────────
    # 서버를 백그라운드로 띄우고 URL을 별도로 가져온다.
    # eval_js는 Colab 노트북 셀이 시작한 프로세스 트리 안에서는 동작한다.

    $PYTHON chatbot_ui.py --server-port $PORT --quiet &
    SERVER_PID=$!

    # 서버가 실제로 응답할 때까지 최대 30초 대기
    echo "서버 기동 대기 중..."
    for i in $(seq 1 30); do
        sleep 1
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "⚠️  UI 서버가 시작 도중 종료되었습니다." >&2
            exit 1
        fi
        if $PYTHON -c "
import urllib.request, sys
try:
    urllib.request.urlopen('http://localhost:$PORT', timeout=1)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
            break
        fi
    done

    # Colab 프록시 URL 획득 (eval_js)
    COLAB_URL=$($PYTHON -c "
import sys
try:
    from google.colab.output import eval_js
    url = eval_js('google.colab.kernel.proxyPort($PORT)')
    if url:
        print(url, end='')
except Exception:
    pass
" 2>/dev/null || true)

    echo ""
    if [ -n "$COLAB_URL" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅  챗봇 UI 접속 URL: $COLAB_URL"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    else
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅  UI 서버 실행 중 (포트 $PORT)"
        echo ""
        echo "접속 URL을 얻으려면 새 Colab 셀에서 아래 코드 실행:"
        echo "  from google.colab.output import eval_js"
        echo "  print(eval_js('google.colab.kernel.proxyPort($PORT)'))"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi
    echo ""

    # 서버가 종료될 때까지 대기 (셀을 살아있게 유지)
    wait $SERVER_PID

else
    # ── 로컬 환경 ────────────────────────────────────────────────
    echo "UI 접속: http://localhost:$PORT"
    $PYTHON chatbot_ui.py --server-port $PORT
fi
