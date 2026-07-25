# Text Emotion Module

Classifies emotional states and confidence scores from transcript text segments.

## DFD Level 0

[ Input: input/transcript.json ]
       │
       ▼
[ Model: j-hartmann ]
       │
       ▼
[ Output: output/transcript_with_emotion.json ]

## Overview
- **Input:** Timestamped JSON transcript (`input/transcript.json`).
- **Model:** `j-hartmann`.
- **Output:** Enhanced JSON containing sentiment/emotion scores per segment (`transcript_with_emotion.json`).

> **Note / Current Status:** This module is currently **not merged** into the main active pipeline. It will be merged with Librosa acoustic features in a future iteration to provide a hybrid text + audio emotion score for better overall response accuracy.