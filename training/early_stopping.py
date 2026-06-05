"""
training/early_stopping.py
Early stopping on a monitored validation metric.

Tracks the best score, saves a full-state checkpoint on improvement,
and sets should_stop=True after patience epochs without improvement.

Grounded in run_20260605 analysis:
  - Best val_pcc_total=0.6486 achieved at epoch 10.
  - With patience=5, stop fires at epoch 15 (correct behaviour confirmed).
  - Recommend patience=5 (tighter than old config patience=10) to avoid
    wasting compute with rebalanced loss (Phase 1) converging earlier.
"""

from __future__ import annotations

import os
import torch


class EarlyStopping:
    """
    Args:
        patience:         epochs to wait after last improvement (default 5)
        min_delta:        minimum change to count as improvement (default 0.001)
        mode:             'max' for PCC (higher=better), 'min' for MSE
        checkpoint_path:  where to save the best model checkpoint
        verbose:          print messages on improvement and stop
    """

    def __init__(
        self,
        patience:         int   = 5,
        min_delta:        float = 0.001,
        mode:             str   = "max",
        checkpoint_path:  str   = "checkpoints/best_model.pt",
        verbose:          bool  = True,
    ) -> None:
        self.patience         = patience
        self.min_delta        = min_delta
        self.mode             = mode
        self.checkpoint_path  = checkpoint_path
        self.verbose          = verbose

        self.best_score:   float | None = None
        self.counter:      int          = 0
        self.should_stop:  bool         = False
        self.best_epoch:   int          = 0

    # ──────────────────────────────────────────────────────────────────────

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(
        self,
        score:       float,
        model:       torch.nn.Module,
        epoch:       int,
        extra_state: dict | None = None,
    ) -> bool:
        """
        Call once per validation epoch.
        Saves checkpoint on improvement.

        Args:
            score:       validation metric value (PCC or MSE)
            model:       model to save on improvement
            epoch:       current epoch number (1-indexed)
            extra_state: optional dict merged into the saved checkpoint
                         (e.g. optimizer_state, scheduler_state, global_step).
                         Use keys compatible with ScorerTrainer.load_checkpoint:
                         "optimizer_state", "scheduler_state", "scaler_state".

        Returns:
            True if training should stop.
        """
        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter    = 0

            os.makedirs(os.path.dirname(os.path.abspath(self.checkpoint_path)),
                        exist_ok=True)
            save_dict: dict = {
                "model_state_dict": model.state_dict(),
                "best_score":       self.best_score,
                "best_epoch":       self.best_epoch,
            }
            if extra_state:
                save_dict.update(extra_state)
            torch.save(save_dict, self.checkpoint_path)

            if self.verbose:
                print(
                    f"[EarlyStopping] epoch {epoch}: "
                    f"new best {self.mode}={score:.4f}  "
                    f"→ saved to {self.checkpoint_path}"
                )
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"[EarlyStopping] epoch {epoch}: "
                    f"no improvement ({score:.4f} vs best {self.best_score:.4f})  "
                    f"patience {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(
                        f"[EarlyStopping] stopping at epoch {epoch}. "
                        f"Best was epoch {self.best_epoch} "
                        f"({self.mode}={self.best_score:.4f})"
                    )

        return self.should_stop

    def load_best(
        self,
        model:  torch.nn.Module,
        device: str = "cpu",
    ) -> dict:
        """
        Load best checkpoint back into model.

        Returns:
            dict of extra_state fields saved alongside the model weights.
        """
        ckpt = torch.load(self.checkpoint_path, map_location=device,
                          weights_only=False)
        # Support both key formats: EarlyStopping saves "model_state_dict",
        # ScorerTrainer._save uses "model_state".
        state = ckpt.get("model_state_dict") or ckpt.get("model_state")
        if state is None:
            raise KeyError(
                "Checkpoint missing 'model_state_dict' or 'model_state' key."
            )
        model.load_state_dict(state)
        if self.verbose:
            print(
                f"[EarlyStopping] loaded best checkpoint: "
                f"epoch {ckpt.get('best_epoch', '?')}  "
                f"{self.mode}={ckpt.get('best_score', '?')}"
            )
        return {k: v for k, v in ckpt.items()
                if k not in ("model_state_dict", "model_state")}
