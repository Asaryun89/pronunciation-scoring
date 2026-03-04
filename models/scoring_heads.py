from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class ScoringConfig:
    emb_dim: int
    utt_dim: int
    prosody_dim: int = 5
    hidden: int = 256

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadScorer(nn.Module):
    """
    Heads:
      - word_score: [0,1]
      - utt_score:  [0,100]
      - prosody_score: [0,1] (or 0-100)
    """
    def __init__(self, cfg: ScoringConfig):
        super().__init__()
        self.word_head = MLP(cfg.emb_dim, cfg.hidden, 1)
        self.utt_head = MLP(cfg.utt_dim, cfg.hidden, 1)
        self.prosody_head = MLP(cfg.prosody_dim, cfg.hidden // 2, 1)

    def forward_word(self, z_word: torch.Tensor) -> torch.Tensor:
        # z_word: (N, D)
        return torch.sigmoid(self.word_head(z_word)).squeeze(-1)

    def forward_utt(self, z_utt: torch.Tensor) -> torch.Tensor:
        # z_utt: (1, UttDim)
        # map to 0..100 via sigmoid
        return (torch.sigmoid(self.utt_head(z_utt)).squeeze(-1) * 100.0)

    def forward_prosody(self, p: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.prosody_head(p)).squeeze(-1)