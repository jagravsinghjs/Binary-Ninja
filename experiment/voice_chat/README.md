# Voice Chat / Emotion Module

Analyzes input transcript segments to detect emotional tones across conversation turns.

## DFD Level 0

[ Input: input/transcript.json ]
       │
       ▼
[ Model: j-hartmann/emotion-english-distilroberta-base ]
       │
       ▼
[ Output: output/transcript_with_emotion.json ]

## Overview
- **Input:** Segmented transcript (`input/transcript.json`).
- **Model:** HuggingFace `j-hartmann/emotion-english-distilroberta-base` classification pipeline.
- **Output:** JSON file updated with emotion confidence distributions per transcript chunk (`transcript_with_emotion.json`).