# Voice Chat / Acoustic Feature Module

Extracts acoustic audio metrics (pitch, energy, spectral dynamics) across conversation segments using signal processing algorithms.

## DFD Level 0

[ Input: audio/test.wav + input/transcript.json ]
       │
       ▼
[ Algorithm: Librosa (Acoustic Feature Extraction) ]
       │
       ▼
[ Output: output/transcript_with_emotion.json ]

## Overview
- **Input:** Raw audio file (`audio/test.wav`) and generated transcript (`input/transcript.json` from `speech_to_text`).
- **Algorithm/Tool:** Librosa (heuristic signal processing algorithm for pitch, mean, std, and energy metrics).
- **Output:** Structured JSON enriched with acoustic feature metrics mapped to transcript segments (`transcript_with_emotion.json`).