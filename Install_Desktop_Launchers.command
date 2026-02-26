#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DESKTOP_DIR="$HOME/Desktop"

mkdir -p "$DESKTOP_DIR"

ditto "$REPO_DIR/Run_Math_Trainer.command" "$DESKTOP_DIR/Run_Math_Trainer.command"
ditto "$REPO_DIR/Run_German_Trainer.command" "$DESKTOP_DIR/Run_German_Trainer.command"
ditto "$REPO_DIR/Update_Repo.command" "$DESKTOP_DIR/Update_Repo.command"

chmod a+x "$DESKTOP_DIR/Run_Math_Trainer.command" "$DESKTOP_DIR/Run_German_Trainer.command" "$DESKTOP_DIR/Update_Repo.command"

echo "Desktop launchers installed in: $DESKTOP_DIR"
osascript -e 'display notification "Desktop launchers installed" with title "school_trainer"' >/dev/null 2>&1 || true

# Close Terminal window after script finishes.
osascript -e 'tell application "Terminal" to close front window' >/dev/null 2>&1 &
exit 0
