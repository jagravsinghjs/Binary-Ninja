# Voice Emotion Module

Processes a single audio file/transcript turn to extract acoustic features and compile single-run analysis.

## DFD Level 0

[ Input: Single Audio / Transcript File ]
       │
       ▼
[ Process: Librosa Acoustic Feature Extraction ]
       │
       ▼
[ Output: output/transcript_with_emotion.json ]

## Overview
- **Execution:** Single-file execution model.
- **Input:** Standalone audio recording (`audio/test.wav`) or transcript file.
- **Algorithm:** `Librosa` (heuristic signal processing for pitch, energy, and speech dynamics).
- **Output:** Feature-enriched JSON file containing acoustic metrics for report generation.