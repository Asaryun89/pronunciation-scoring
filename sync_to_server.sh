#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# sync_to_server.sh
#
# Upload the pronunciation-scoring codebase to a remote GPU training server.
#
# Usage (Git Bash / WSL / Linux / Mac):
#   bash sync_to_server.sh                  # full sync: code + checkpoint
#   bash sync_to_server.sh --dry-run        # preview file list, no transfer
#   bash sync_to_server.sh --code-only      # skip pretrain checkpoint upload
#   bash sync_to_server.sh --ckpt-only      # checkpoint only, no code
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# SERVER SETTINGS  ← edit these before first use
# ─────────────────────────────────────────────────────────────────────────────
SSH_USER="root"
SSH_HOST="125.190.184.215"            # e.g. 192.168.1.100  or  ssh.vast.ai
SSH_PORT=61377
SSH_KEY="$HOME/.ssh/id_rsa"          # private key (omit -i line below for password)
REMOTE_DIR="/workspace/pronunciation-scoring"
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_TAR="$LOCAL_DIR/checkpoints/pretrain/best_model.tar.gz"
ASR_CACHE="$LOCAL_DIR/data/asr_transcripts.pkl"

# ── arg parsing ───────────────────────────────────────────────────────────────
DRY_RUN=false; CODE_ONLY=false; CKPT_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --dry-run)   DRY_RUN=true ;;
    --code-only) CODE_ONLY=true ;;
    --ckpt-only) CKPT_ONLY=true ;;
  esac
done

# ── helpers ───────────────────────────────────────────────────────────────────
_ssh_opts=(-p "$SSH_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no -o BatchMode=yes)

_dry=(); $DRY_RUN && _dry=(--dry-run)

_rsync() {
  rsync -avz --progress "${_dry[@]}" \
    -e "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o BatchMode=yes" \
    "$@"
}

_header() { echo; echo "-->  $*"; }

# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Syncing to: $SSH_USER@$SSH_HOST:$REMOTE_DIR"
$DRY_RUN && echo "  MODE: DRY RUN — file list preview only, nothing sent"
echo "============================================================"

# ── Step 1: remote directory structure ───────────────────────────────────────
if ! $DRY_RUN; then
  _header "Creating remote directories..."
  ssh "${_ssh_opts[@]}" "$SSH_USER@$SSH_HOST" \
    "mkdir -p '$REMOTE_DIR'/{model,data,training,configs,tests} \
              '$REMOTE_DIR'/checkpoints/{pretrain,scorer}       \
              '$REMOTE_DIR'/logs/scorer"
  echo "      OK"
fi

# ── Step 2: Python source (skip when --ckpt-only) ────────────────────────────
if ! $CKPT_ONLY; then

  _header "model/"
  _rsync \
    --exclude="__pycache__/" --exclude="*.pyc" \
    "$LOCAL_DIR/model/" \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/model/"

  _header "data/  (no kmeans binaries)"
  _rsync \
    --exclude="__pycache__/" --exclude="*.pyc" \
    --exclude="kmeans/" \
    --exclude="asr_transcripts.pkl" \
    "$LOCAL_DIR/data/" \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/data/"

  _header "training/"
  _rsync \
    --exclude="__pycache__/" --exclude="*.pyc" \
    "$LOCAL_DIR/training/" \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/training/"

  _header "configs/  (scoring + pretrain + base)"
  _rsync \
    "$LOCAL_DIR/configs/scoring_config.yaml" \
    "$LOCAL_DIR/configs/pretrain.yaml" \
    "$LOCAL_DIR/configs/base.yaml" \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/configs/"

  _header "tests/"
  _rsync \
    --exclude="__pycache__/" --exclude="*.pyc" \
    "$LOCAL_DIR/tests/" \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/tests/"

  _header "root scripts + requirements"
  _rsync \
    "$LOCAL_DIR/pretrain.py" \
    "$LOCAL_DIR/evaluate_scorer.py" \
    "$LOCAL_DIR/requirements.txt" \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/"

fi

# ── Step 3: pretrain checkpoint — transfer pre-compressed tar.gz (skip when --code-only)
if ! $CODE_ONLY; then
  echo
  if [ -f "$PRETRAIN_TAR" ]; then
    TAR_SIZE=$(du -sh "$PRETRAIN_TAR" | cut -f1)
    _header "checkpoints/pretrain/best_model.tar.gz  ($TAR_SIZE)"
    # No -z: file is already gzipped
    rsync -av --progress \
      "${_dry[@]}" \
      -e "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o BatchMode=yes" \
      "$PRETRAIN_TAR" \
      "$SSH_USER@$SSH_HOST:$REMOTE_DIR/checkpoints/pretrain/"

    if ! $DRY_RUN; then
      _header "Extracting on server..."
      ssh "${_ssh_opts[@]}" "$SSH_USER@$SSH_HOST" \
        "tar -xzf '$REMOTE_DIR/checkpoints/pretrain/best_model.tar.gz' \
             -C   '$REMOTE_DIR/checkpoints/pretrain/' && \
         rm       '$REMOTE_DIR/checkpoints/pretrain/best_model.tar.gz'"
      echo "      done — best_model.pt extracted, archive removed."
    fi
  else
    echo
    echo "  WARNING: compressed checkpoint not found at:"
    echo "    $PRETRAIN_TAR"
    echo "  Create it first:  tar -czf checkpoints/pretrain/best_model.tar.gz \\"
    echo "                         -C checkpoints/pretrain best_model.pt"
  fi
fi

# ── Step 4: ASR cache (optional — only if it exists locally) ─────────────────
if ! $CODE_ONLY && ! $CKPT_ONLY && [ -f "$ASR_CACHE" ]; then
  CACHE_SIZE=$(du -sh "$ASR_CACHE" | cut -f1)
  _header "data/asr_transcripts.pkl  ($CACHE_SIZE)"
  _rsync \
    "$ASR_CACHE" \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/data/asr_transcripts.pkl"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo
echo "============================================================"
if $DRY_RUN; then
  echo "  DRY RUN complete — run without --dry-run to transfer."
else
  echo "  Sync complete."
  echo
  echo "  Next on the server:"
  echo "    ssh -p $SSH_PORT $SSH_USER@$SSH_HOST"
  echo "    cd $REMOTE_DIR"
  echo "    pip install -r requirements.txt"
  if [ ! -f "$ASR_CACHE" ]; then
    echo "    python -m data.speechocean_asr --config configs/scoring_config.yaml"
  fi
  echo "    python evaluate_scorer.py --sanity-check \\"
  echo "           --config configs/scoring_config.yaml \\"
  echo "           --checkpoint checkpoints/pretrain/best_model.pt"
  echo "    python training/train_scorer.py --config configs/scoring_config.yaml"
fi
echo "============================================================"
