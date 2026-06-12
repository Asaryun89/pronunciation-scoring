"""
stage1_ctc — Stage 1 of the pronunciation scoring pipeline (CTC fine-tuning).

Modules:
  train     — entry point (python -m stage1_ctc.train)
  config    — hyperparameters and paths
  processor — vocab building and Wav2Vec2Processor construction
  data      — dataset loading and preprocessing
  collator  — DataCollatorCTCWithPadding
  metrics   — WER compute_metrics factory
  model     — HuBERT-CTC model builder
"""
