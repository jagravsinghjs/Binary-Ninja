# Chat LLM Module

Processes transcript and emotion/acoustic feature data using a local LLM to generate clinician-facing mental state reports and visual timelines.

## DFD Level 0

[ Input: output/transcript_with_emotion.json ]
       │
       ▼
[ Model: Ollama (qwen2.5:7b-instruct) ]
       │
       ▼
[ Output: output/report.json & mental_state.png ]

## Overview
- **Execution:** Single batch inference per complete conversation session.
- **Input:** Feature-enriched transcript JSON (`transcript_with_emotion.json`).
- **Model:** Local Ollama runner with `qwen2.5:7b-instruct`.
- **Outputs:** 
  1. `output/report.json` - Segment-level stress/mood timeline and clinician summary.
  2. `output/mental_state.png` - Visual plot of emotional/mental state dynamics over time.