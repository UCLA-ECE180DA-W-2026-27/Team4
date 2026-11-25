#!/usr/bin/env bash

# Start computer object detection, wait for it to become ready, and then start
# Raspberry Pi hand guidance over SSH. Run this script on the Mac.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_NAME="${1:-cereal}"
OCR_OPTION="${2:-}"
MAC_USB_IP="${MAC_USB_IP:-10.12.194.2}"
PI_SSH_HOST="${PI_SSH_HOST:-piuser@10.12.194.1}"
MAC_PYTHON="${MAC_PYTHON:-/opt/anaconda3/envs/tetris/bin/python}"
PI_PYTHON="${PI_PYTHON:-/home/piuser/yes/envs/tetris/bin/python}"
SERVER_PORT="${SERVER_PORT:-8000}"
SERVER_PID=""

case "$OCR_OPTION" in
    "")
        USE_OCR=0
        ;;
    --ocr)
        USE_OCR=1
        ;;
    *)
        echo "Usage: $0 \"OBJECT\" [--ocr]" >&2
        echo "Use --ocr only for an exact branded product with visible text." >&2
        exit 2
        ;;
esac

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ ! -x "$MAC_PYTHON" ]]; then
    echo "Mac tetris Python not found: $MAC_PYTHON" >&2
    echo "Set MAC_PYTHON to the Python executable in your tetris environment." >&2
    exit 1
fi

echo "Starting computer detector for: $TARGET_NAME"
SERVER_ARGS=(
    "$SCRIPT_DIR/object_detection_server.py"
    --host "$MAC_USB_IP"
    --port "$SERVER_PORT"
    --object "$TARGET_NAME"
)
if [[ "$USE_OCR" -eq 1 ]]; then
    SERVER_ARGS+=(--ocr-verify)
    echo "Apple Vision OCR verification enabled for this branded product."
else
    echo "Apple Vision OCR verification disabled (generic-object mode)."
fi
"$MAC_PYTHON" "${SERVER_ARGS[@]}" &
SERVER_PID=$!

echo "Waiting for the computer detector to become ready..."
SERVER_READY=0
for _ in {1..180}; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "The computer detector stopped before becoming ready." >&2
        wait "$SERVER_PID" || true
        exit 1
    fi
    if curl --ipv4 --noproxy '*' --fail --silent --max-time 1 \
        "http://$MAC_USB_IP:$SERVER_PORT/health" >/dev/null; then
        SERVER_READY=1
        break
    fi
    sleep 1
done

if [[ "$SERVER_READY" -ne 1 ]]; then
    echo "Timed out waiting for the computer detector." >&2
    exit 1
fi

echo "Computer detector ready. Starting hand guidance on $PI_SSH_HOST."
printf -v REMOTE_TARGET '%q' "$TARGET_NAME"
printf -v REMOTE_URL '%q' "http://$MAC_USB_IP:$SERVER_PORT/detect"
printf -v REMOTE_PYTHON '%q' "$PI_PYTHON"

ssh -t "$PI_SSH_HOST" \
    "cd /home/piuser/180DATeam4/hand_object_detection && \
    $REMOTE_PYTHON hand_orientation.py \
      --pi \
      --camera 0 \
      --camera-facing away \
      --object-detector remote \
      --remote-url $REMOTE_URL \
      --object $REMOTE_TARGET \
      --speak-guidance \
      --no-local-display"
