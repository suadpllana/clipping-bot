#!/usr/bin/env bash
# Bring up the clipbot UI from a clean checkout.
#   ./run.sh            start the UI on :8000
#   ./run.sh --port N   start it somewhere else
set -euo pipefail
cd "$(dirname "$0")"

# Homebrew is not always on PATH for non-login shells.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is not installed. Install it and re-run:"
  case "$(uname -s)" in
    Darwin) echo "  brew install ffmpeg" ;;
    Linux)  echo "  sudo apt install ffmpeg" ;;
    *)      echo "  winget install Gyan.FFmpeg" ;;
  esac
  exit 1
fi

# librosa needs 3.10+; the system python on macOS is still 3.9.
find_python() {
  for p in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$p" >/dev/null 2>&1 && \
       "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      command -v "$p"; return 0
    fi
  done
  return 1
}

if [ ! -x .venv/bin/python ]; then
  PY="$(find_python)" || {
    echo "Need Python 3.10 or newer. On macOS:  brew install python@3.12"; exit 1; }
  echo "creating .venv with $PY"
  "$PY" -m venv .venv
fi

# Install only when something is actually missing, so restarts stay instant.
if ! .venv/bin/python -c 'import fastapi, uvicorn, numpy' 2>/dev/null; then
  echo "installing dependencies…"
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt
fi

exec .venv/bin/python -m clipbot ui "$@"
