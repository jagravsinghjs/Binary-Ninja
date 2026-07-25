# Voice Emotion Module

Extracts acoustic audio features (pitch, energy, dynamics) across conversation turns using signal processing algorithms.

## DFD Level 0

[ Input: audio/test.wav ]
       │
       ▼
[ Algorithm: Librosa (Acoustic Feature Extraction) ]
       │
       ▼
[ Output: output/transcript_with_emotion.json ]

## Overview
- **Input:** Audio recording (`audio/test.wav`).
- **Algorithm/Tool:** `Librosa` (heuristic signal processing algorithm for pitch, std, and energy).
- **Output:** Structured JSON enriched with acoustic feature metrics per audio segment (`transcript_with_emotion.json`).