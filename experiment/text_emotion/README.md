# Text Emotion Module

Classifies emotional states and confidence scores from transcript text segments.

## DFD Level 0

[ Input: input/transcript.json ]
       │
       ▼
[ Model: j-hartmann/emotion-english-distilroberta-base ]
       │
       ▼
[ Output: output/transcript_with_emotion.json ]

## Overview
- **Input:** Timestamped JSON transcript (`input/transcript.json`).
- **Model:** HuggingFace `j-hartmann/emotion-english-distilroberta-base`.
- **Output:** Enhanced JSON containing sentiment/emotion scores per segment (`transcript_with_emotion.json`).
