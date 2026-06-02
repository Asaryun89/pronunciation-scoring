"""
TrainLogger — thin wrapper around TensorBoard / WandB for pre-training.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class TrainLogger:
    """
    Lightweight logging wrapper.

    Supports TensorBoard (always available if tensorboard is installed)
    and optional Weights & Biases.

    Args:
        use_wandb:  Enable WandB logging.
        use_tb:     Enable TensorBoard logging.
        tb_log_dir: TensorBoard log directory.
        project:    WandB project name.
        run_name:   Run name for both backends.
        config:     Config dict logged as hyperparameters.
    """

    def __init__(
        self,
        use_wandb:  bool = False,
        use_tb:     bool = True,
        tb_log_dir: str  = "logs/pretrain",
        project:    str  = "multi-res-hubert",
        run_name:   str  = "pretrain",
        config:     Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tb_writer = None
        self._wandb_run  = None

        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=tb_log_dir)
                log.info("TensorBoard logging → %s", tb_log_dir)
            except ImportError:
                log.warning("tensorboard not installed — TB logging disabled.")

        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project = project,
                    name    = run_name,
                    config  = config or {},
                    resume  = "allow",
                )
                log.info("WandB logging enabled (project=%s).", project)
            except Exception as exc:
                log.warning("WandB init failed (%s) — WandB logging disabled.", exc)

    def log(self, metrics: Dict[str, float], step: int) -> None:
        """Log a dict of scalar metrics at the given step."""
        if self._tb_writer is not None:
            for k, v in metrics.items():
                try:
                    self._tb_writer.add_scalar(k, v, global_step=step)
                except Exception:
                    pass

        if self._wandb_run is not None:
            try:
                import wandb
                self._wandb_run.log(metrics, step=step)
            except Exception:
                pass

    def finish(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
