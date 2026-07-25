# Speech to Text Module

Transcribes input audio recordings into structured plain text and timestamped JSON data.

## DFD Level 0

[ Input: audio/test.wav ]
       │
       ▼
[ Model: faster-whisper (medium, cuda/float16) ]
       │
       ▼
[ Output: output/transcript.txt & transcript.json ]

## Overview
- **Input:** Audio file (`audio/test.wav`).
- **Model:** `faster-whisper` (medium model, GPU float16 precision).
- **Outputs:** Plain text transcript (`transcript.txt`) and timestamped JSON (`transcript.json`).