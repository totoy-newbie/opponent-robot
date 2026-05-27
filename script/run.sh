#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT/.venv"

# Detect Python inside the virtual env on Linux/macOS and Windows
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
if [ -x "$VENV_DIR/Scripts/python.exe" ]; then
  PYTHON="$VENV_DIR/Scripts/python.exe"
  PIP="$VENV_DIR/Scripts/pip.exe"
fi

if [ ! -x "$PYTHON" ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  if command -v python3 >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON=python3
  elif command -v python >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON=python
  else
    echo "Error: Python is not installed or not on PATH."
    exit 1
  fi
  "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
fi

echo "Using virtual environment: $VENV_DIR"
"$PIP" install --upgrade pip
"$PIP" install -r "$ROOT/requirements.txt"

cd "$ROOT"
#"$PYTHON" -m src.app "$@"
"$PYTHON" -m src.app --source 1 --model "src/lib/hand_landmarker.task"
