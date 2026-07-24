from pathlib import Path
from faster_whisper import WhisperModel
import time

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
AUDIO_PATH = BASE_DIR / "audio" / "test.wav"
OUTPUT_PATH = BASE_DIR / "output" / "transcript.txt"

# ------------------------------------------------------------------
# Load Model
# ------------------------------------------------------------------

print("Loading Whisper model...")

start = time.time()

model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="float16"
)

print(f"Model loaded in {time.time()-start:.2f} sec\n")

# ------------------------------------------------------------------
# Transcribe
# ------------------------------------------------------------------

segments, info = model.transcribe(
    str(AUDIO_PATH),
    beam_size=5
)

segments = list(segments)

print("=" * 60)
print(f"Language : {info.language}")
print(f"Confidence : {info.language_probability:.2f}")
print("=" * 60)

OUTPUT_PATH.parent.mkdir(exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    for segment in segments:

        line = (
            f"[{segment.start:.2f}s - "
            f"{segment.end:.2f}s] "
            f"{segment.text.strip()}"
        )

        print(line)
        f.write(line + "\n")

print("\nTranscript saved to:")
print(OUTPUT_PATH)

import json

transcript = []

OUTPUT_PATH.parent.mkdir(exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    for segment in segments:

        line = (
            f"[{segment.start:.2f}s - "
            f"{segment.end:.2f}s] "
            f"{segment.text.strip()}"
        )

        print(line)
        f.write(line + "\n")

        transcript.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })

print("\nTranscript saved to:")
print(OUTPUT_PATH)

json_path = OUTPUT_PATH.parent / "transcript.json"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(transcript, f, indent=4, ensure_ascii=False)

print(f"\nJSON saved to: {json_path}")