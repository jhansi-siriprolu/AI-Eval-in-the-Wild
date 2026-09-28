#!/usr/bin/env bash
# Sets up an isolated Python environment and installs everything the
# project's scripts and Lesson 05 need. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "⚠️  This project targets Apple Silicon (arm64). Detected: $(uname -m)"
  echo "   MLX will not install/run on an Intel Mac. Exiting."
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "Creating virtual environment (.venv)..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo "Installing project requirements..."
pip install -r requirements.txt

echo
echo "✅ Environment ready. Activate it in future sessions with:"
echo "   source .venv/bin/activate"
echo
echo "Next: python scripts/02_download_and_test.py"
