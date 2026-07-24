from pathlib import Path
import json

from transformers import pipeline

# ----------------------------------------------------
# Paths
# ----------------------------------------------------

BASE_DIR = Path(__file__).parent

INPUT_PATH = BASE_DIR / "input" / "transcript.json"
OUTPUT_PATH = BASE_DIR / "output" / "transcript_with_emotion.json"

OUTPUT_PATH.parent.mkdir(exist_ok=True)

# ----------------------------------------------------
# Load Emotion Model
# ----------------------------------------------------

print("Loading emotion model...")

classifier = pipeline(
    task="text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=0          # GPU
)

print("Model Loaded!\n")

# ----------------------------------------------------
# Read Transcript
# ----------------------------------------------------

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    transcript = json.load(f)

# ----------------------------------------------------
# Predict Emotion
# ----------------------------------------------------

results = []

for segment in transcript:

    text = segment["text"]

    prediction = classifier(text)[0]

    emotions = {}

    for item in prediction:
        emotions[item["label"]] = round(item["score"], 4)

    segment["text_emotion"] = emotions

    results.append(segment)

# ----------------------------------------------------
# Save Output
# ----------------------------------------------------

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Saved to:\n{OUTPUT_PATH}")