from .multi_res_hubert import (
    MultiResHuBERT,
    MultiResHuBERTOutput,
    DownsamplingModule,
    UpsamplingModule,
    ConvFeatureMasking,
    UnitPredictionHead,
    PronunciationScoreHead,
)
from .audio_encoder          import AudioEncoder
from .text_encoder_qwen3     import Qwen3MeanPoolEncoder
from .text_projection        import TokenTextProjection
from .cross_attention_fusion import CrossAttentionFusion, TextProjection
from .scoring_head           import MLPScoringHead, SCORE_DIMS
from .pronunciation_scorer   import PronunciationScorer
from .losses                 import (
    PronunciationScoringLoss,
    DimensionHuberLoss,
    ListMLERankingLoss,
    ConcordanceCorrelationLoss,
)

__all__ = [
    # Multi-res HuBERT backbone (pre-training + shared)
    "MultiResHuBERT",
    "MultiResHuBERTOutput",
    "DownsamplingModule",
    "UpsamplingModule",
    "ConvFeatureMasking",
    "UnitPredictionHead",
    "PronunciationScoreHead",
    # Scorer pipeline
    "AudioEncoder",
    "Qwen3MeanPoolEncoder",
    "TokenTextProjection",
    "CrossAttentionFusion",
    "TextProjection",           # legacy — kept for test compat
    "MLPScoringHead",
    "SCORE_DIMS",
    "PronunciationScorer",
    # Losses
    "PronunciationScoringLoss",
    "DimensionHuberLoss",
    "ListMLERankingLoss",
    "ConcordanceCorrelationLoss",
]
