# Shared constants for the inference pipeline.

# Output dimension order for the sentence scoring head
SENT_DIMS = ["total", "accuracy", "fluency", "prosodic", "completeness"]

# SpeechOcean762 utterance-level score scale (labels are 0–10)
SENT_SCALE = 10.0
