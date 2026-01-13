# Pronunciation Scoring (PyTorch)

This project provides an end-to-end, modular PyTorch baseline for Computer-Aided Pronunciation Training (CAPT). It takes a 16 kHz WAV file, extracts log-Mel features, and predicts an overall pronunciation score (0-100) using a CNN + BiLSTM model with attention pooling.

## Features
- Reference text to phonemes via G2P
- Phoneme alignment (forced align / CTC / HMM)
- Audio preprocessing to waveform/log-Mel, SSL speech encoder embeddings
- Pronunciation encoder (Conv1D + Transformer MSA)
- Phoneme-level scoring head (Neural GOP / CTC / DNN) with aggregation
- Optional LLM adapter for feedback and final outputs

## New Workflow

```
Reference Text
   |
   v
+-------------------------------+
| G2P (Text -> Phonemes)        |
+-------------------------------+
   |
   v
+-------------------------------+
| Phoneme Alignment Module      |
| (Forced Align / CTC / HMM)    |
+-------------------------------+
   | Canonical phonemes
   |
----+-----------------------------
                               
Audio Input                    
   |                           
   v                           
+-----------------------------+ 
| Waveform / Log-Mel          | 
+-----------------------------+ 
   |                           
   v                           
+-----------------------------+ 
| SSL Speech Encoder          | 
| (Data2Vec2 / HuBERT / W2V2) | 
+-----------------------------+ 
   | Acoustic embeddings        
   v                            
+-----------------------------+ 
| Pronunciation Encoder       | 
| (Conv1D + Transformer MSA)  | 
+-----------------------------+ 
   | Frame-level features       
   v                            
+-----------------------------+
| Phoneme-Level Scoring Head  |
| (Neural GOP / CTC / DNN)    |
+-----------------------------+
   |
   v
+-----------------------------+
| Score Aggregation           |
| (phone + word + sentence)   |
+-----------------------------+
   |
   v
+-----------------------------+
| Frozen LLM + Adapter        |
| (Interpret + feedback)      |
+-----------------------------+
   |
   v
+-----------------------------+
| Final Outputs               |
| - Overall Score (0-100)     |
| - Accuracy/Fluency/Prosody  |
| - Per-phoneme feedback      |
| - NL explanation            |
+-----------------------------+
```

## Scoring Dimensions
- Overall score (0-100): single regression target used for training and evaluation.
- Optional breakdowns you can add in your dataset/labels: accuracy, fluency, and prosody.

## Project Structure
```
.
├── config/
│   └── config.yaml
├── data/
│   ├── dataset.py              # Dummy dataset with phoneme targets
│   └── collate.py              # Padding and batch collation
├── docs/
├── features/
│   └── logmel.py               # Log-Mel feature computation
├── models/
│   └── pronunciation_model.py  # SSL encoder, pronunciation encoder, scoring head
├── training/
│   ├── train.py
│   └── loss.py
├── inference/
│   └── infer.py                # G2P, alignment, scoring, and feedback
├── utils/
│   ├── alignment.py            # Forced alignment stub
│   ├── audio.py                # Audio preprocessing helpers
│   ├── g2p.py                  # G2P stub
│   ├── logger.py               # Logger setup
│   └── metrics.py              # Metrics utilities
├── requirements.txt
└── README.md
```

## Training (dummy data)
```
python training/train.py --config config/config.yaml --output checkpoints/pronunciation.pt
```

## Inference
```
python inference/infer.py --audio path/to/audio.wav --text "Your reference text" --checkpoint checkpoints/pronunciation.pt
python inference/infer.py --audio path/to/audio.wav --text "Your reference text" --checkpoint checkpoints/pronunciation.pt --feedback-out outputs/feedback.json
```

## Notes
- All example data is randomly generated in `data/dataset.py`.
