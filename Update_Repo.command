#!/bin/bash
set -euo pipefail

find_repo() {
  local script_dir candidate
  script_dir="$(cd "$(dirname "$0")" && pwd)"

  if [[ -n "${SCHOOL_TRAINER_REPO:-}" ]] && [[ -d "$SCHOOL_TRAINER_REPO/.git" ]]; then
    echo "$SCHOOL_TRAINER_REPO"
    return 0
  fi

  if [[ -d "$script_dir/.git" ]]; then
    echo "$script_dir"
    return 0
  fi

  if command -v mdfind >/dev/null 2>&1; then
    while IFS= read -r candidate; do
      if [[ -d "$candidate/.git" ]] && [[ -f "$candidate/mathe_trainer.py" ]] && [[ -f "$candidate/deutsch_trainer.py" ]]; then
        echo "$candidate"
        return 0
      fi
    done < <(mdfind "kMDItemFSName == 'school_trainer'c")
  fi

  while IFS= read -r candidate; do
    candidate="$(dirname "$candidate")"
    if [[ -d "$candidate/.git" ]] && [[ -f "$candidate/mathe_trainer.py" ]]; then
      echo "$candidate"
      return 0
    fi
  done < <(find "$HOME" -maxdepth 6 -type f -name "mathe_trainer.py" 2>/dev/null)

  return 1
}

REPO_DIR="$(find_repo || true)"
if [[ -z "$REPO_DIR" ]]; then
  osascript -e 'display alert "school_trainer not found" message "Set SCHOOL_TRAINER_REPO or place launcher in the repo."' >/dev/null 2>&1 || true
  osascript -e 'tell application "Terminal" to close front window' >/dev/null 2>&1 &
  exit 1
fi

cd "$REPO_DIR"

# Fast-forward local branch from origin.
git fetch --all --prune
git pull --ff-only

# Close Terminal window after script finishes.
osascript -e 'tell application "Terminal" to close front window' >/dev/null 2>&1 &
exit 0
