"""
notebook_infer — clean inference API for Jupyter notebooks.

Usage
-----
from notebook_infer import ScoreConfig, load_predictor, score_file, score_array
from notebook_infer.display import show_scores, show_waveform

cfg  = ScoreConfig(checkpoint="ckpt_hubert_multitask/best.pt", device="cpu")
pred = load_predictor(cfg)

result = score_file("audio/sample.wav", pred, cfg)
show_scores(result)
show_waveform("audio/sample.wav", result)
"""

from .pipeline import ScoreConfig, load_predictor, score_file, score_array  # noqa: F401
