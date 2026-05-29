"""
Thin logging abstraction over wandb and/or TensorBoard.

Both backends are optional — if neither is installed the trainer falls
back to Python's stdlib ``logging`` module so training always works.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class TrainLogger:
    """
    Unified logger that fans metrics out to wandb, TensorBoard, or both.

    Args:
        use_wandb:   Initialise a wandb run.
        use_tb:      Open a TensorBoard SummaryWriter.
        tb_log_dir:  Directory for TensorBoard event files.
        project:     wandb project name.
        run_name:    wandb / TB run identifier.
        config:      Hyper-parameter dict logged to wandb as run config.
    """

    def __init__(
        self,
        use_wandb: bool = False,
        use_tb:    bool = False,
        tb_log_dir: Optional[str] = None,
        project:    str = "multi-res-hubert",
        run_name:   Optional[str] = None,
        config:     Optional[Dict[str, Any]] = None,
    ) -> None:
        self._wandb: Any = None
        self._tb:    Any = None

        if use_wandb:
            try:
                import wandb
                wandb.init(project=project, name=run_name, config=config or {})
                self._wandb = wandb
                log.info("WandB run started  project=%s  run=%s", project, run_name)
            except ImportError:
                log.warning("wandb not installed — WandB logging disabled.")
            except Exception as exc:
                log.warning("WandB init failed (%s) — continuing without it.", exc)

        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=tb_log_dir)
                log.info("TensorBoard writer → %s", tb_log_dir)
            except ImportError:
                log.warning("tensorboard not installed — TB logging disabled.")

    # ──────────────────────────────────────────────────────────────────────

    def log(self, metrics: Dict[str, float], step: int) -> None:
        """
        Fan ``metrics`` out to all enabled backends.

        Args:
            metrics: ``{tag: scalar_value}`` — all values must be Python floats
                     or 0-d tensors (will be converted with ``.item()``).
            step:    Global training step (x-axis for all backends).
        """
        # Normalise tensor values to Python float
        clean: Dict[str, float] = {
            k: float(v) if not hasattr(v, "item") else v.item()
            for k, v in metrics.items()
        }

        if self._wandb is not None:
            self._wandb.log(clean, step=step)

        if self._tb is not None:
            for k, v in clean.items():
                self._tb.add_scalar(k, v, global_step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a text summary (e.g. evaluation table)."""
        if self._wandb is not None:
            self._wandb.log({tag: self._wandb.Html(f"<pre>{text}</pre>")}, step=step)
        if self._tb is not None:
            self._tb.add_text(tag, text, global_step=step)

    def finish(self) -> None:
        """Flush and close all backends."""
        if self._wandb is not None:
            self._wandb.finish()
        if self._tb is not None:
            self._tb.close()

    # ──────────────────────────────────────────────────────────────────────

    def __enter__(self) -> "TrainLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.finish()
