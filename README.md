# English Pronunciation Scoring System

A modular pronunciation scoring pipeline built on self-supervised speech representations. The system combines HuBERT embeddings, lightweight ASR alignment, and multi-head scoring to evaluate pronunciation at phoneme, word, and utterance levels.

## Overview

The project is designed for both research and production-style deployment, with focus on:

- Pronunciation accuracy scoring
- Prosody and fluency modeling
- Error localization for actionable learner feedback
- Structured output for downstream applications

## Objectives

- Evaluate learner pronunciation using deep contextual speech representations
- Provide phoneme-level and word-level error localization
- Generate an overall pronunciation score (0-100)
- Model prosody, fluency, and rhythm
- Produce structured feedback output for user-facing applications

## Core Components

### 1. HuBERT Encoder

A self-supervised Transformer-based model that extracts rich acoustic representations from raw waveform input.

- Input: 16 kHz waveform
- Output: frame-level embeddings with shape `[T x D]`

Where:

- `T`: number of time frames
- `D`: embedding dimension per frame

HuBERT captures:

- Phonetic structure
- Coarticulation effects
- Accent variation
- Prosody and rhythm
- Temporal dependencies

### 2. Lightweight ASR Alignment Model

Provides temporal segmentation of words and phonemes.

- Purpose: alignment only (not scoring)
- Typical options:
  - Small CTC-based ASR
  - Whisper-tiny
  - Distilled wav2vec2

Outputs:

- Transcript
- Word timestamps
- Optional phoneme timestamps

Example:

```text
Input speech: "She sells sea shells"

ASR output:
{
  "transcript": "She sells sea shells",
  "word_timestamps_sec": [
    {"word": "She", "start": 0.12, "end": 0.40},
    {"word": "sells", "start": 0.41, "end": 0.80},
    {"word": "sea", "start": 0.81, "end": 1.05},
    {"word": "shells", "start": 1.06, "end": 1.45}
  ]
}
```

### 3. Alignment Mapping

Maps HuBERT frame embeddings to linguistic segments.

Each segment looks like:

```
frames 0–120   → "She"
frames 121–300 → "sells"
frames 301–410 → "sea"
frames 411–580 → "shells"
```

For each segment:

```text
z_segment = Pool(E[t_start:t_end])
```

Where:

- `E` = HuBERT frame embeddings
- `Pool` = mean pooling or attention pooling

### 4. Multi-Head Neural Scoring Network

Predicts pronunciation quality at multiple granularities.

- Phoneme head
  - Input: phoneme embedding
  - Output: score in `[0, 1]`
- Word head
  - Input: aggregated phoneme embeddings
  - Output: score in `[0, 1]`
- Utterance head
  - Input: global embedding
  - Output: overall score `(0-100)`
- Prosody head
  - Inputs: pitch (F0 statistics), energy contour, speaking rate, pause duration
  - Outputs: fluency score, rhythm score, intonation score

## Pronunciation Grading Outputs

- Phoneme-level scores
- Word-level scores
- Utterance-level overall score
- Prosody and fluency evaluation
- Structured feedback output

## Feature Highlights

- HuBERT encoder for acoustic representation (act as feature extraction, conduct to utterance evaluation)
- Lightweight ASR for word/phoneme alignment (get timestamps for word/phoneme/sentence)
- Multi-head neural scoring network
- Error localization (mispronounced words/segments)

## Dataset Preparation

Reference speech:

- 10-20 English sentences
- Spoken by native speakers
- Clean recordings (16 kHz, mono WAV)

Learner speech:

- Same sentence set as reference
- Spoken by Vietnamese learners
- Recorded under similar acoustic conditions

## System Workflow

```text
                           ┌───────────────────────────┐
                           │        Audio Input        │
                           │   (16 kHz mono waveform)  │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                        ┌────────────────────────────────┐
                        │  Preprocessing Module          │
                        │  - Voice Activity Detection    │
                        │  - Silence trimming            │
                        │  - Optional denoising          │
                        └────────────────┬───────────────┘
                                         │
                                         ▼
        ┌────────────────────────────────────────────────────────────┐
        │                    Parallel Processing                     │
        └────────────────────────────────────────────────────────────┘
                 │                                            │
                 ▼                                            ▼
     ┌─────────────────────────┐                 ┌─────────────────────────┐
     │   HuBERT Encoder        │                 │  Lightweight ASR Model  │
     │  (Self-Supervised SSL)  │                 │  (CTC / Whisper-tiny)   │
     │                         │                 │                         │
     │ Output: Frame-level     │                 │ Output: Transcript +    │
     │ embeddings [T × D]      │                 │ Word/Phone timestamps   │
     └─────────────┬───────────┘                 └─────────────┬───────────┘
                   │                                           │
                   └────────────────────┬──────────────────────┘
                                        ▼
                        ┌────────────────────────────────┐
                        │  Alignment Mapping Module      │
                        │  - Map frames to segments      │
                        │  - Build segment frame ranges  │
                        └────────────────┬───────────────┘
                                         │
                                         ▼
                        ┌────────────────────────────────┐
                        │  Segment Pooling Module        │
                        │  - Mean/Attention pooling      │
                        │  - Produce segment vectors     │
                        └────────────────┬───────────────┘
                                         │
                                         ▼
              ┌────────────────────────────────────────────────┐
              │            Multi-Head Scoring Network          │
              │  - Phoneme Scoring Head (0–1)                  │
              │  - Word Scoring Head (0–1)                     │
              │  - Utterance Scoring Head (0–100)              │
              │  - Prosody Scoring Head                        │
              └────────────────┬───────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────────────────────┐
              │          Score Fusion & Calibration            │
              │  Combine accuracy + fluency + prosody          │
              └────────────────┬───────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────────────────────┐
              │            Feedback Generation Module          │
              │  - Highlight low-score words                   │
              │  - Detect mispronounced phonemes               │
              │  - Generate structured JSON output             │
              └────────────────────────────────────────────────┘
```

## Run

```bash
python main.py --learner-audio path/to/learner.wav --reference-audio path/to/reference.wav
```
