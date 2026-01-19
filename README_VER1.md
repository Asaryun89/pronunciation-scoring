# English Pronunciation Scoring System (Baseline)

This project provides a classical signal-processing–based baseline for English Pronunciation Scoring.
The system compares a learner’s speech with a native speaker reference recording and identifies pronunciation errors using time alignment and acoustic feature distance.
The system processes 16 kHz WAV audio, extracts MFCC and pitch features, aligns speech segments using Dynamic Time Warping (DTW) or Forced Alignment, and computes pronunciation scores based on Euclidean distance.

## Objectives

- Compare learner pronunciation with native speaker pronunciation
- Detect mispronounced words and phonemes
- Provide an interpretable pronunciation score (0–100)
- Visualize acoustic differences between speakers

## Features

- Native–learner speech comparison
- Word-level and phoneme-level alignment
- MFCC + pitch feature extraction
- DTW-based temporal matching
- Distance-based pronunciation scoring
- Error localization (which word is wrong)

## Dataset Preparation
- 10–20 English sentences
- Spoken by native speakers
- Clean recording (16 kHz, mono WAV)
- Learner Speech
- Same sentences
- Spoken by Vietnamese learners
- Recorded under similar conditions

## System Workflow

```
Reference Sentence Text
        |
        v
+-----------------------------------+
| Sentence Selection (10–20 lines)  |
+-----------------------------------+
        |
        v
+-----------------------------------+
| Native Speaker Recording (WAV)    |
+-----------------------------------+
        |
        v
+-----------------------------------+
| Reference Audio Preprocessing     |
| - Resample to 16 kHz              |
| - Silence trimming                |
+-----------------------------------+
        |
        v
+-----------------------------------+
| Feature Extraction (Reference)    |
| - MFCC                            |
| - Pitch (F0)                      |
+-----------------------------------+
        |
        |  Reference acoustic features
        |
----------------------------------------------
                                            
Learner Audio Input                         |
        |                                   |
        v                                   |
+-----------------------------------+       |
| Learner Recording (WAV)           |       |
+-----------------------------------+       |
        |                                   |
        v                                   |
+-----------------------------------+       |
| Learner Audio Preprocessing       |       |
| - Resample to 16 kHz              |       |
| - Silence trimming                |       |
+-----------------------------------+       |
        |                                   |
        v                                   |
+-----------------------------------+       |
| Feature Extraction (Learner)      |       |
| - MFCC                            |       |
| - Pitch (F0)                      |       |
+-----------------------------------+       |
        |                                   |
        |  Learner acoustic features        |
        v                                   v
+--------------------------------------------------+
| Alignment Module                                 |
| - Dynamic Time Warping (DTW)                     |
|   or                                             |
| - Forced Alignment (Montreal Forced Aligner)     |
+--------------------------------------------------+
        |
        | Aligned feature pairs
        v
+--------------------------------------------------+
| Feature Comparison                               |
| - MFCC distance (Euclidean)                      |
| - Pitch difference (Hz)                          |
+--------------------------------------------------+
        |
        v
+--------------------------------------------------+
| Pronunciation Scoring                            |
| Score = f(distance)                              |
| Smaller distance → higher score                  |
+--------------------------------------------------+
        |
        v
+--------------------------------------------------+
| Aggregation                                      |
| - Frame → word score                             |
| - Word → sentence score                          |
+--------------------------------------------------+
        |
        v
+--------------------------------------------------+
| Final Outputs                                    |
| - Overall score (0–100)                          |
| - Word-level error detection                     |
| - Distance visualization                         |
+--------------------------------------------------+
```
