## Algorithm 1: Speech-to-Text Transcription (`transcribe.py`)

**Input:** Audio file (`.wav`)
**Output:** `transcript.txt`, `transcript.json`

1. Load `WhisperModel` (medium, device = CUDA, compute_type = float16)
2. `segments, info ← model.transcribe(audio_path, beam_size = 5)`
3. `transcript ← empty list`
4. For each segment in segments:
   - Extract `start`, `end`, `text` from segment
   - Write formatted line to `transcript.txt`
   - Append `{start, end, text}` to `transcript`
5. Save `transcript` as `transcript.json`