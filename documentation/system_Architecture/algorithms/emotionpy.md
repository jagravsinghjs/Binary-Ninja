## Algorithm 3: Text Emotion Enrichment (`emotion.py`)

**Input:** `transcript_with_emotion.json` from Algorithm 2 (words + acoustic features)
**Output:** `transcript_with_emotion.json` (adds text-emotion scores per segment)

1. Load `j-hartmann/emotion-english-distilroberta-base` classifier (GPU)
2. For each segment in input:
   - `prediction ← classifier(segment.text)`
   - `emotions ← {label: score for each (label, score) in prediction}`
   - Attach `emotions` to segment as `text_emotion`
3. Save segments (now containing words, acoustic features, and text emotion) as `transcript_with_emotion.json`

---