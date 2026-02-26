#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

python3 deutsch_trainer.py

# Close Terminal window after script finishes.
osascript -e 'tell application "Terminal" to close front window' >/dev/null 2>&1 &
exit 0
