"""
stage2_scoring — Stage 2 of the pronunciation scoring pipeline.

Trains HubertMultiTask (HuBERT audio stream + optional text stream fused via
cross-attention) to predict utterance-level pronunciation scores.

Modules:
  train            — entry point (python -m stage2_scoring.train)
  hubert_multitask — HubertMultiTask model (training + inference forward)
  scoring_heads    — CrossAttentionFusion and MLPScoringHead blocks
  constants        — score dims/scales shared with inference
  collate          — batch collator (audio decode, VAD, prosody targets)
  validate         — SpeechOcean762 schema validation
"""
