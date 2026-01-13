# Pronunciation Scoring (PyTorch)

This project provides an end-to-end, modular PyTorch baseline for Computer-Aided Pronunciation Training (CAPT). It takes a 16 kHz WAV file and reference text, extracts MFCC-based features, and predicts an overall pronunciation score (0-100) plus phoneme-level scores using a CNN + BiLSTM model.

## Features
- WAV input (16 kHz), normalize and trim silence
- MFCC + delta + delta-delta features
- Forced alignment placeholder (phoneme timestamps provided or uniform segments)
- CNN + BiLSTM regression model
- Config-driven training and inference

## Project Structure
```
.
├── config/
│   └── config.yaml
├── data/
│   ├── dataset.py
│   └── collate.py
├── features/
│   └── mfcc.py
├── models/
│   └── pronunciation_model.py
├── training/
│   ├── train.py
│   └── loss.py
├── inference/
│   └── infer.py
├── utils/
│   ├── audio.py
│   ├── metrics.py
│   └── logger.py
├── requirements.txt
└── README.md
```

## Setup
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Training (dummy data)
```
python training/train.py --config config/config.yaml --output checkpoints/pronunciation.pt
```

## Inference
```
python inference/infer.py --audio path/to/audio.wav --text "reference text" --checkpoint checkpoints/pronunciation.pt
```

## Notes
- Forced alignment is a placeholder. Provide real phoneme timestamps to replace the uniform segments.
- All example data is randomly generated in `data/dataset.py`.
