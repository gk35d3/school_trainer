#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# Fast-forward local branch from origin.
git fetch --all --prune
git pull --ff-only

# Close Terminal window after script finishes.
osascript -e 'tell application "Terminal" to close front window' >/dev/null 2>&1 &
exit 0
