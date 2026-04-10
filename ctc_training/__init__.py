"""
ctc_training — Stage 1 of the pronunciation scoring pipeline.

Modules:
  config    — hyperparameters and paths
  processor — vocab building and Wav2Vec2Processor construction
  data      — dataset loading and preprocessing
  collator  — DataCollatorCTCWithPadding
  metrics   — WER compute_metrics factory
  model     — HuBERT-CTC model builder
"""
