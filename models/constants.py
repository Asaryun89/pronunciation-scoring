# Shared constants between training (models/train.py) and inference (inference/predictor.py).
# Any change here must be reflected in both pipelines.

# Output dimension order for the sentence head
SENT_DIMS = ["total", "accuracy", "fluency", "prosodic", "completeness"]

# SpeechOcean762 score scales
SENT_SCALE  = 10.0  # utterance-level and word-level labels are 0–10
PHONE_SCALE =  2.0  # phones-accuracy labels are 0–2

# Audio prosody feature keys — must match utils/prosody.prosody_to_vector()
PROSODY_DIMS = ["rms", "zcr", "peak_rate", "f0_mean", "f0_std"]

# Sentence-head dimensions that are trained on — excludes completeness (idx 4)
# because SpeechOcean learners almost always score 10/10, causing constant predictions
ACTIVE_SENT_IDXS = [0, 1, 2, 3]   # total, accuracy, fluency, prosodic

# Approximate HuBERT-base frame rate for precomputing word span indices
HUBERT_FRAME_HZ = 50.0
