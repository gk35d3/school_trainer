#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# Fast-forward local branch from origin.
git fetch --all --prune
git pull --ff-only

echo
read -rp "Update complete. Press Enter to close..."
