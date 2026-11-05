#!/usr/bin/env bash
# Install (or remove) the macOS LaunchAgent that keeps the paper loop running across logins and crashes.
#   bash itrader/online/ops/install_launchd.sh            install and start
#   bash itrader/online/ops/install_launchd.sh --remove   stop and remove
# Stop any loop you started by hand first:  pkill -f "itrader.online paper"
# Note: macOS may ask whether "python" can access the Documents folder the first time; allow it,
# or move the repo outside Documents. Paper mode only: live trading stays behind its four locks.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
LABEL="com.itrader.online.paper"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
if [ "${1:-}" = "--remove" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$DEST"
  echo "removed $LABEL"
  exit 0
fi
[ -x "$REPO/.venv/bin/python" ] || { echo "no venv at $REPO/.venv: run itrader/online/ops/setup.sh first"; exit 1; }
mkdir -p "$HOME/Library/LaunchAgents" "$REPO/data/online/runs"
sed "s|__REPO__|$REPO|g" "$REPO/itrader/online/ops/com.itrader.online.paper.plist" > "$DEST"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$DEST"
echo "installed and started $LABEL"
echo "log:    $REPO/data/online/runs/paper.log"
echo "status: launchctl print gui/$(id -u)/$LABEL | head -20"
