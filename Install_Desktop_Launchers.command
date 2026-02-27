#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DESKTOP_DIR="$HOME/Desktop"

mkdir -p "$DESKTOP_DIR"

# Generate standalone launchers that directly run the matching task.
cat > "$DESKTOP_DIR/Run_Math_Trainer.command" <<EOF
#!/bin/bash
set -euo pipefail
cd "$REPO_DIR"
python3 -m apps.mathe_trainer
osascript -e 'tell application "Terminal" to close front window' >/dev/null 2>&1 &
exit 0
EOF

cat > "$DESKTOP_DIR/Run_German_Trainer.command" <<EOF
#!/bin/bash
set -euo pipefail
cd "$REPO_DIR"
python3 -m apps.deutsch_trainer
osascript -e 'tell application "Terminal" to close front window' >/dev/null 2>&1 &
exit 0
EOF

cat > "$DESKTOP_DIR/Update_Repo.command" <<EOF
#!/bin/bash
set -euo pipefail
cd "$REPO_DIR"
git fetch --all --prune
git pull --ff-only
osascript -e 'tell application "Terminal" to close front window' >/dev/null 2>&1 &
exit 0
EOF

chmod a+x "$DESKTOP_DIR/Run_Math_Trainer.command" "$DESKTOP_DIR/Run_German_Trainer.command" "$DESKTOP_DIR/Update_Repo.command"

# Attach custom icons from internet-downloaded PNG files saved in this repo.
apply_icon() {
  local source_png="$1"
  local target_file="$2"
  local tmp_rsrc
  tmp_rsrc="$(mktemp /tmp/codex_icon.XXXXXX)"
  if sips -i "$source_png" >/dev/null 2>&1 \
    && DeRez -only icns "$source_png" > "$tmp_rsrc" 2>/dev/null \
    && Rez -append "$tmp_rsrc" -o "$target_file" >/dev/null 2>&1 \
    && SetFile -a C "$target_file" >/dev/null 2>&1; then
    :
  fi
  rm -f "$tmp_rsrc"
}

apply_icon "$REPO_DIR/assets/icons/math.png" "$DESKTOP_DIR/Run_Math_Trainer.command"
apply_icon "$REPO_DIR/assets/icons/german.png" "$DESKTOP_DIR/Run_German_Trainer.command"
apply_icon "$REPO_DIR/assets/icons/update.png" "$DESKTOP_DIR/Update_Repo.command"

echo "Desktop launchers installed in: $DESKTOP_DIR"
osascript -e 'display notification "Desktop launchers installed" with title "school_trainer"' >/dev/null 2>&1 || true

# Close Terminal window after script finishes.
osascript -e 'tell application "Terminal" to close front window' >/dev/null 2>&1 &
exit 0
