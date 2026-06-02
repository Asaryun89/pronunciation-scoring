#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# sync_from_server.sh
#
# Pull training outputs from the remote GPU server back to local machine.
#
# Usage (Git Bash / WSL / Linux / Mac):
#   bash sync_from_server.sh                # pull checkpoints + logs
#   bash sync_from_server.sh --ckpt-only    # checkpoints only
#   bash sync_from_server.sh --logs-only    # logs only
#   bash sync_from_server.sh --dry-run      # preview, no transfer
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SSH_USER="root"
SSH_HOST="125.190.184.215"
SSH_PORT=61377
SSH_KEY="$HOME/.ssh/id_rsa"
REMOTE_DIR="/workspace/pronunciation-scoring"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── arg parsing ───────────────────────────────────────────────────────────────
DRY_RUN=false; CKPT_ONLY=false; LOGS_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --dry-run)   DRY_RUN=true ;;
    --ckpt-only) CKPT_ONLY=true ;;
    --logs-only) LOGS_ONLY=true ;;
  esac
done

_dry=(); $DRY_RUN && _dry=(--dry-run)

_rsync() {
  rsync -avz --progress "${_dry[@]}" \
    -e "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o BatchMode=yes" \
    "$@"
}

_header() { echo; echo "-->  $*"; }

echo "============================================================"
echo "  Pulling from: $SSH_USER@$SSH_HOST:$REMOTE_DIR"
$DRY_RUN && echo "  MODE: DRY RUN — nothing transferred"
echo "============================================================"

# ── Checkpoints ───────────────────────────────────────────────────────────────
if ! $LOGS_ONLY; then
  mkdir -p "$LOCAL_DIR/checkpoints/scorer"

  _header "checkpoints/scorer/"
  _rsync \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/checkpoints/scorer/" \
    "$LOCAL_DIR/checkpoints/scorer/"
fi

# ── Logs ─────────────────────────────────────────────────────────────────────
if ! $CKPT_ONLY; then
  mkdir -p "$LOCAL_DIR/logs"

  _header "logs/"
  _rsync \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/logs/" \
    "$LOCAL_DIR/logs/"
fi

echo
echo "============================================================"
$DRY_RUN && echo "  DRY RUN complete." || echo "  Pull complete."
echo "============================================================"
