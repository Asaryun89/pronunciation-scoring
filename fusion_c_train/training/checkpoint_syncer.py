"""
Asynchronous checkpoint syncer — Vast.ai → local Windows machine.

After each checkpoint save the trainer calls ``syncer.sync_async(path)``.
The transfer runs in a daemon background thread so training is never blocked.

Supported transfer methods
──────────────────────────
• ``rsync``  — preferred; requires rsync on the server and OpenSSH on Windows.
• ``scp``    — simpler fallback; same SSH requirements.
• ``rclone`` — push to any rclone remote (GDrive, S3, Backblaze …),
               then sync to local with ``rclone sync`` on Windows.

Config section (add to pretrain.yaml / finetune_config.yaml)
─────────────────────────────────────────────────────────────
sync:
  enabled:   false      # flip to true to activate

  method:    "rsync"    # "rsync" | "scp" | "rclone"

  # ── rsync / scp ──────────────────────────────────────────────────────
  dest_user: "myuser"
  dest_host: "192.168.1.100"   # local Windows IP (or hostname via VPN)
  dest_port: 22                # Windows OpenSSH port
  ssh_key:   "~/.ssh/id_rsa"   # private key on the Vast.ai instance
  dest_path: "/c/Users/myuser/checkpoints/"  # MSYS2 path or /cygdrive/…
  ssh_opts:  ""                # extra SSH flags, e.g. "-o ProxyJump=…"

  # ── rclone ───────────────────────────────────────────────────────────
  rclone_remote:  "gdrive:ml/checkpoints/"  # rclone remote:path
  rclone_args:    ""           # extra rclone flags

  # ── common ───────────────────────────────────────────────────────────
  timeout_s:      300          # per-transfer timeout (seconds)
  max_retries:    3            # retry attempts with exponential back-off
  delete_after:   false        # remove checkpoint from server after sync
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class CheckpointSyncer:
    """
    Fire-and-forget checkpoint syncer.

    Usage::

        syncer = CheckpointSyncer(cfg["sync"])
        syncer.sync_async(Path("checkpoints/pretrain/epoch_001.pt"))
        # … training continues …
        syncer.wait_all()   # call once at the end of training

    When ``sync.enabled`` is ``false`` every method is a no-op.
    """

    def __init__(self, sync_cfg: Optional[Dict] = None) -> None:
        cfg = sync_cfg or {}
        self.enabled: bool = cfg.get("enabled", False)
        self._cfg          = cfg
        self._threads: List[threading.Thread] = []
        self._lock         = threading.Lock()

        if self.enabled:
            self._validate()

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def sync_async(self, checkpoint_path: Path) -> None:
        """
        Schedule a background sync of *checkpoint_path* to the configured
        destination.  Returns immediately; transfer happens in a daemon thread.
        """
        if not self.enabled:
            return
        t = threading.Thread(
            target  = self._sync_with_retry,
            args    = (Path(checkpoint_path),),
            name    = f"sync-{Path(checkpoint_path).name}",
            daemon  = True,
        )
        with self._lock:
            self._threads.append(t)
        t.start()
        log.debug("Sync thread started for %s", checkpoint_path)

    def upload_and_delete(self, local_path: Path, name: str) -> None:
        """
        Synchronously upload *local_path* to the configured rclone remote,
        then delete the local file.  Blocks until the transfer finishes.
        If all retries fail the local file is kept and an error is logged.
        """
        if not self.enabled or self._cfg.get("method") != "rclone":
            return
        remote_dir = self._cfg.get("rclone_remote", "").rstrip("/")
        extra      = self._cfg.get("rclone_args", "").split()
        max_ret    = self._cfg.get("max_retries", 3)

        for attempt in range(1, max_ret + 1):
            try:
                self._run(["rclone", "copy", str(local_path), remote_dir,
                           "--no-traverse"] + extra)
                log.info("[sync] ✓ %s → %s  (attempt %d/%d)",
                         name, remote_dir, attempt, max_ret)
                local_path.unlink(missing_ok=True)
                return
            except Exception as exc:
                log.warning("[sync] Upload attempt %d/%d failed for %s: %s",
                            attempt, max_ret, name, exc)
                if attempt < max_ret:
                    time.sleep(2 ** attempt)
        log.error("[sync] All upload attempts failed for %s — kept at %s.",
                  name, local_path)

    def delete_remote_checkpoint(self, name: str) -> None:
        """Delete a named file from the remote checkpoint directory."""
        if not self.enabled or self._cfg.get("method") != "rclone":
            return
        remote_path = self._cfg.get("rclone_remote", "").rstrip("/") + "/" + name
        try:
            result = subprocess.run(
                ["rclone", "deletefile", remote_path],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                log.debug("[sync] Deleted remote: %s", name)
            else:
                log.warning("[sync] Failed to delete remote %s: %s",
                            name, result.stderr[:200].strip())
        except Exception as exc:
            log.warning("[sync] Error deleting remote %s: %s", name, exc)

    def download_checkpoint(self, name: str, dest_dir: Path) -> Optional[Path]:
        """
        Download *name* from the remote checkpoint directory to *dest_dir*.
        Returns the local Path on success, None on failure.
        """
        if not self.enabled or self._cfg.get("method") != "rclone":
            return None
        remote_path = self._cfg.get("rclone_remote", "").rstrip("/") + "/" + name
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                ["rclone", "copy", remote_path, str(dest_dir)],
                capture_output=True, text=True,
                timeout=self._cfg.get("timeout_s", 600),
            )
            local = dest_dir / name
            if result.returncode == 0 and local.exists():
                log.info("[sync] Downloaded %s → %s", name, local)
                return local
            log.warning("[sync] Failed to download %s: %s",
                        name, result.stderr[:200].strip())
        except Exception as exc:
            log.warning("[sync] Error downloading %s: %s", name, exc)
        return None

    def sync_dir_async(self, local_dir: Path, remote_path: str) -> None:
        """
        Schedule a background rclone copy of an entire local directory.
        Only supported when ``sync.method`` is ``'rclone'``.
        No-op when disabled or when using rsync/scp.
        """
        if not self.enabled or self._cfg.get("method") != "rclone":
            return
        if not remote_path:
            return
        t = threading.Thread(
            target  = self._sync_dir_with_retry,
            args    = (Path(local_dir), remote_path),
            name    = f"sync-dir-{Path(local_dir).name}",
            daemon  = True,
        )
        with self._lock:
            self._threads.append(t)
        t.start()
        log.debug("Dir-sync thread started for %s → %s", local_dir, remote_path)

    def _sync_dir_with_retry(self, local_dir: Path, remote_path: str) -> None:
        max_retries = self._cfg.get("max_retries", 3)
        extra = self._cfg.get("rclone_args", "").split()
        for attempt in range(1, max_retries + 1):
            try:
                self._run(["rclone", "copy", str(local_dir), remote_path] + extra)
                log.info("[sync] ✓ logs %s → %s  (attempt %d/%d)",
                         local_dir.name, remote_path, attempt, max_retries)
                return
            except Exception as exc:
                log.warning("[sync] Dir-sync attempt %d/%d failed (%s): %s",
                            attempt, max_retries, local_dir.name, exc)
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        log.error("[sync] All dir-sync attempts failed for %s.", local_dir)

    def wait_all(self, timeout: float = 600.0) -> None:
        """
        Block until all pending sync threads finish (or until *timeout* seconds
        elapses).  Call once at the very end of training to flush the queue.
        """
        if not self.enabled:
            return
        with self._lock:
            pending = [t for t in self._threads if t.is_alive()]
        if not pending:
            return
        log.info("Waiting for %d pending sync transfer(s) …", len(pending))
        deadline = time.monotonic() + timeout
        for t in pending:
            remaining = max(0.0, deadline - time.monotonic())
            t.join(timeout=remaining)
            if t.is_alive():
                log.warning("Sync thread %s did not finish within timeout.", t.name)
        log.info("All sync transfers done.")

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        method = self._cfg.get("method", "rsync")
        if method not in ("rsync", "scp", "rclone"):
            raise ValueError(
                f"sync.method must be 'rsync', 'scp', or 'rclone'; got {method!r}"
            )

        if shutil.which(method) is None:
            log.warning(
                "'%s' binary not found in PATH — sync will fail at runtime. "
                "Install it on the Vast.ai instance before training.",
                method,
            )

        if method in ("rsync", "scp"):
            for key in ("dest_user", "dest_host", "dest_path", "ssh_key"):
                if not self._cfg.get(key):
                    raise ValueError(
                        f"sync.{key} is required when sync.method='{method}'. "
                        f"Add it to your config's [sync] section."
                    )
        elif method == "rclone":
            if not self._cfg.get("rclone_remote"):
                raise ValueError(
                    "sync.rclone_remote is required when sync.method='rclone'."
                )

    def _sync_with_retry(self, path: Path) -> None:
        max_retries = self._cfg.get("max_retries", 3)
        for attempt in range(1, max_retries + 1):
            try:
                self._do_sync(path)
                log.info(
                    "[sync] ✓ %s → %s  (attempt %d/%d)",
                    path.name,
                    self._destination_label(),
                    attempt, max_retries,
                )
                if self._cfg.get("delete_after", False) and path.exists():
                    path.unlink()
                    log.debug("[sync] Deleted local copy: %s", path)
                return
            except Exception as exc:
                log.warning(
                    "[sync] Attempt %d/%d failed for %s: %s",
                    attempt, max_retries, path.name, exc,
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)   # 2 s, 4 s, 8 s …
        log.error("[sync] All %d attempts failed for %s.", max_retries, path.name)

    def _do_sync(self, path: Path) -> None:
        method = self._cfg.get("method", "rsync")
        if method == "rsync":
            self._rsync(path)
        elif method == "scp":
            self._scp(path)
        else:
            self._rclone(path)

    def _rsync(self, path: Path) -> None:
        cfg     = self._cfg
        key     = Path(cfg["ssh_key"]).expanduser()
        port    = cfg.get("dest_port", 22)
        extra   = cfg.get("ssh_opts", "").strip()
        ssh_cmd = (
            f"ssh -p {port} -i {key} "
            f"-o StrictHostKeyChecking=no -o BatchMode=yes {extra}"
        ).strip()
        dest    = f"{cfg['dest_user']}@{cfg['dest_host']}:{cfg['dest_path']}"
        self._run(["rsync", "-az", "--progress", "-e", ssh_cmd, str(path), dest])

    def _scp(self, path: Path) -> None:
        cfg  = self._cfg
        key  = Path(cfg["ssh_key"]).expanduser()
        port = cfg.get("dest_port", 22)
        dest = f"{cfg['dest_user']}@{cfg['dest_host']}:{cfg['dest_path']}"
        self._run([
            "scp",
            "-P", str(port),
            "-i", str(key),
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            str(path),
            dest,
        ])

    def _rclone(self, path: Path) -> None:
        cfg    = self._cfg
        remote = cfg["rclone_remote"]
        extra  = cfg.get("rclone_args", "").split()
        self._run(["rclone", "copy", str(path), remote, "--progress"] + extra)

    def _run(self, cmd: List[str]) -> None:
        timeout = self._cfg.get("timeout_s", 300)
        result  = subprocess.run(
            cmd,
            capture_output = True,
            text           = True,
            timeout        = timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed (rc={result.returncode}):\n"
                f"  CMD: {' '.join(str(c) for c in cmd)}\n"
                f"  STDERR: {result.stderr[:800].strip()}"
            )

    def _destination_label(self) -> str:
        cfg    = self._cfg
        method = cfg.get("method", "rsync")
        if method == "rclone":
            return cfg.get("rclone_remote", "<rclone remote>")
        return f"{cfg.get('dest_user', '?')}@{cfg.get('dest_host', '?')}:{cfg.get('dest_path', '?')}"


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper — call from pretrain.py / train.py / finetune_fusionC.py
# ─────────────────────────────────────────────────────────────────────────────

def build_syncer(cfg: dict) -> CheckpointSyncer:
    """
    Construct a ``CheckpointSyncer`` from the full training config dict.
    Returns a disabled no-op syncer if the ``sync`` section is absent or
    ``sync.enabled`` is ``false``.
    """
    sync_cfg = cfg.get("sync", {})
    syncer   = CheckpointSyncer(sync_cfg)
    if syncer.enabled:
        method = sync_cfg.get("method", "rsync")
        dest   = syncer._destination_label()
        log.info("[sync] Auto-sync ENABLED  method=%s  dest=%s", method, dest)
    else:
        log.debug("[sync] Auto-sync disabled.")
    return syncer
