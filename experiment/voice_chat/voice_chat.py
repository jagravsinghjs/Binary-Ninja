"""
voice_chat.py
Live, voice-based conversation with the assistant. Per turn:
  record -> transcribe (faster-whisper) -> acoustic features (librosa)
  -> text emotion (j-hartmann) -> conversational reply (Ollama)
Raw audio is NEVER written to disk or kept after a turn is processed —
only the transcript text, extracted features, and eventual report persist.

At the end of the session (you choose to stop between turns), a report
is generated automatically and saved to output/session_<timestamp>/.

Usage:
    python voice_chat.py
"""

import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faster_whisper import WhisperModel
from transformers import pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b-instruct"
SAMPLE_RATE = 16000

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

CONVO_SYSTEM_PROMPT = """You are a warm, present listener. Someone is checking in by voice, between their
doctor visits, just to talk through how they're doing — casual, no pressure, like talking to a friend.
This is NOT a diagnostic session and you are NOT a therapist. You are not here to fix anything today —
you're here so that later, their doctor has an honest picture of how things have actually been, instead
of the patient having to reconstruct weeks from memory in a rushed 15-minute appointment.

After what the person says, you'll sometimes see a bracketed note like [voice cues: ...] — this is
acoustic and text-emotion analysis of HOW they said it, not something they said out loud. Use it only
to calibrate your tone (e.g. if the cues suggest more distress than the words alone convey, be a little
gentler) — NEVER mention, quote, or reference these cues directly. The person does not know this
analysis is happening in the background.

Your job each turn:
- Briefly acknowledge what they just said, in your own words (1 sentence) — show you actually heard it.
- Then ask ONE genuinely curious, open-ended follow-up, specific to what they just said — not generic,
  not yes/no.

Hard rules:
- Do NOT try to wrap up or close out the conversation, ever — ending the session is entirely the
  person's decision, made outside this conversation, never something you initiate or hint at.
- Do NOT jump to advice or coping tips mid-conversation. Just listen and ask.
- Do NOT diagnose or use clinical labels.
- Keep replies short: 2-3 sentences, casual, like a friend actually paying attention.
- 0-1 emoji per reply, only if natural — do not force it every turn.

Respond with plain conversational text only — no JSON, no formatting.
"""

REPORT_SYSTEM_PROMPT = """You are a clinical decision-support assistant, NOT a therapist and NOT a
diagnostic tool. You are given a full voice-based conversation between a patient and a listening
assistant, recorded between doctor visits. The goal is to give the doctor an accurate memory of how
the patient has actually been doing — not a diagnosis, not an illness score.

For each turn you're given the transcribed words, acoustic features (pitch variability, energy
variability, pause ratio, arousal label), and text-based emotion scores.

Your job:
1. For EACH turn, assign a "distress_score" from 0 (calm/settled) to 10 (highly distressed/agitated),
   weighing the words, the acoustic cues, AND the text emotion scores together — not any single signal
   alone.
2. Write a "clinician_summary" (4-6 sentences): the patient's apparent trajectory across the
   conversation, notable themes, and anything worth exploring further at the appointment. Frame
   everything as decision-support, never diagnosis ("may indicate", "consider asking about" — never
   "patient has X").
3. Write a "patient_message" (4-6 sentences, ~60-90 words) to close the session — warm, casual, like a
   friend, 2-4 single simple emoji placed mid-sentence (never combine two emoji with a joiner character).
   Validate what they shared across the WHOLE conversation. Do not diagnose. Do not give false
   reassurance if the content suggests real distress. Never suggest medication or treatment techniques.
   You may offer ONE small suggestion only if clearly tailored to what THEY specifically described as
   available (do not suggest friends/family if they said those aren't an option) — otherwise skip it.
   Never mention "doctor" or "appointment" in this message — handled separately.

Return ONLY valid JSON, no markdown fences, no preamble, in exactly this shape:
{
  "turns": [
    {"turn_index": <int>, "distress_score": <int 0-10>, "note": "<short phrase why>"}
  ],
  "clinician_summary": "<string>",
  "patient_message": "<string>"
}
"""

END_PHRASES = None  # no longer used — session end is now a manual keyboard action, not detected speech


# ------------------------------------------------------------------
# Model loading (once, at startup)
# ------------------------------------------------------------------

def load_models():
    print("Loading Whisper model...")
    t0 = time.time()
    whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")
    print(f"Whisper loaded in {time.time()-t0:.1f}s")

    print("Loading text-emotion model...")
    t0 = time.time()
    emotion_classifier = pipeline(
        task="text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=0,
    )
    print(f"Emotion model loaded in {time.time()-t0:.1f}s\n")

    return whisper_model, emotion_classifier


# ------------------------------------------------------------------
# Audio recording (in-memory only, never written to disk)
# ------------------------------------------------------------------

def record_turn_audio(save_path):
    input("Press Enter to start speaking...")
    print("Recording... press Enter again to stop.")

    frames = []

    def callback(indata, frames_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback)
    with stream:
        input()

    if not frames:
        return False

    audio = np.concatenate(frames, axis=0).flatten()
    sf.write(save_path, audio, SAMPLE_RATE)
    return True


# ------------------------------------------------------------------
# Acoustic feature extraction (same logic as voice_emotion.py)
# ------------------------------------------------------------------

def extract_acoustic_features(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    if len(y) == 0:
        return {}

    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=450, sr=sr)
    f0_voiced = f0[~np.isnan(f0)]

    if len(f0_voiced) > 2:
        median_f0 = np.median(f0_voiced)
        clean = f0_voiced[(f0_voiced > median_f0 * 0.6) & (f0_voiced < median_f0 * 1.6)]
        if len(clean) > 0:
            f0_voiced = clean

    pitch_mean = float(np.mean(f0_voiced)) if len(f0_voiced) else 0.0
    pitch_std = float(np.std(f0_voiced)) if len(f0_voiced) else 0.0

    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
    spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))

    intervals = librosa.effects.split(y, top_db=25)
    voiced_duration = sum((e - s) for s, e in intervals) / sr
    total_duration = len(y) / sr
    pause_ratio = 1 - (voiced_duration / total_duration) if total_duration > 0 else 0.0

    return {
        "pitch_mean_hz": round(pitch_mean, 2),
        "pitch_std_hz": round(pitch_std, 2),
        "energy_mean": round(energy_mean, 4),
        "energy_std": round(energy_std, 4),
        "zero_crossing_rate": round(zcr, 4),
        "spectral_centroid": round(spec_cent, 2),
        "pause_ratio": round(pause_ratio, 3),
    }


def classify_arousal(features):
    if not features:
        return "unknown"
    pitch_std = features.get("pitch_std_hz", 0)
    energy_std = features.get("energy_std", 0)
    pause_ratio = features.get("pause_ratio", 0)
    score = 0
    if pitch_std > 40:
        score += 1
    if energy_std > 0.02:
        score += 1
    if pause_ratio > 0.3:
        score -= 1
    if score >= 2:
        return "high_arousal"
    elif score <= -1:
        return "low_arousal"
    return "neutral"


# ------------------------------------------------------------------
# Ollama call
# ------------------------------------------------------------------

def call_ollama_chat(messages, system_prompt, format_json=False):
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "stream": False,
    }
    if format_json:
        payload["format"] = "json"

    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    if "message" not in data or "content" not in data.get("message", {}):
        print("Unexpected Ollama response shape:")
        print(json.dumps(data, indent=2))
        raise RuntimeError("Ollama did not return message.content")

    return data["message"]["content"]


# ------------------------------------------------------------------
# Post-processing helpers
# ------------------------------------------------------------------

def redistribute_trailing_emoji(text):
    text = re.sub(
        "([\U0001F300-\U0001FAFF\U00002600-\U000027BF])\u200d(?=[\U0001F300-\U0001FAFF\U00002600-\U000027BF])",
        r"\1 ",
        text,
    )
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u200d\uFE0F]+"
    )
    single_emoji_pattern = re.compile(
        "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
    )
    trailing_match = re.search(r"(\s*(?:" + emoji_pattern.pattern + r"\s*)+)$", text)
    if not trailing_match:
        return text
    trailing_block = trailing_match.group(0)
    emojis = single_emoji_pattern.findall(trailing_block)
    if len(emojis) < 2:
        return text
    body = text[: trailing_match.start()].strip()
    sentences = re.split(r"(?<=[.!?])\s+", body)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return text
    rebuilt = []
    for i, sentence in enumerate(sentences):
        if i < len(emojis):
            m = re.match(r"^(.*?)([.!?]?)$", sentence)
            core, punct = m.group(1), m.group(2)
            sentence = f"{core} {emojis[i]}{punct}"
        rebuilt.append(sentence)
    return " ".join(rebuilt)


def should_nudge_toward_doctor(turns):
    scores = [t["distress_score"] for t in turns]
    if len(scores) < 2:
        return False
    mid = len(scores) // 2
    first_half = scores[:mid] or [scores[0]]
    second_half = scores[mid:]
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    return (avg_second - avg_first) >= 1.5 or scores[-1] >= 6


# ------------------------------------------------------------------
# Main conversation loop
# ------------------------------------------------------------------

def run_conversation(whisper_model, emotion_classifier, temp_dir):
    print("\nSession started. After each reply, press Enter to keep talking, or type 'report' to end and generate the report.\n")

    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "turn_tmp.wav")

    messages = []
    turns = []
    start_time = time.time()
    turn_index = 0

    opener = "Hey, how's it going? What's on your mind today?"
    print(f"Assistant: {opener}\n")
    messages.append({"role": "assistant", "content": opener})

    while True:
        got_audio = record_turn_audio(temp_path)
        if not got_audio:
            print("No audio captured, try again.")
            continue

        segments, info = whisper_model.transcribe(temp_path, beam_size=5)
        segments = list(segments)
        turn_text = " ".join(s.text.strip() for s in segments).strip()

        if not turn_text:
            print("Didn't catch that, try again.")
            os.remove(temp_path)
            continue

        print(f"You said: {turn_text}")

        elapsed = round(time.time() - start_time, 1)

        acoustic = extract_acoustic_features(temp_path)
        arousal = classify_arousal(acoustic)

        emotion_raw = emotion_classifier(turn_text)[0]
        text_emotion = {item["label"]: round(item["score"], 4) for item in emotion_raw}
        top_emotions = sorted(text_emotion.items(), key=lambda x: -x[1])[:3]

        os.remove(temp_path)  # done with the audio — comment this out if you want to keep turn recordings

        context_note = (
            f"[voice cues: pitch_std={acoustic.get('pitch_std_hz', 0)}, "
            f"energy_std={acoustic.get('energy_std', 0)}, "
            f"pause_ratio={acoustic.get('pause_ratio', 0)}, "
            f"arousal={arousal}, top_text_emotion={top_emotions}]"
        )
        user_content = f"{turn_text}\n{context_note}"
        messages.append({"role": "user", "content": user_content})

        reply = call_ollama_chat(messages, CONVO_SYSTEM_PROMPT)
        print(f"\nAssistant: {reply}\n")
        messages.append({"role": "assistant", "content": reply})

        turns.append({
            "turn_index": turn_index,
            "elapsed_seconds": elapsed,
            "text": turn_text,
            "acoustic_features": acoustic,
            "arousal_label": arousal,
            "text_emotion": text_emotion,
        })
        turn_index += 1

        action = input("Press Enter to keep talking, or type 'report' to end and generate the report: ").strip().lower()
        if action == "report":
            break

    return turns


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------

def generate_report(turns, output_dir):
    transcript_for_model = [
        {
            "turn_index": t["turn_index"],
            "text": t["text"],
            "acoustic_features": t["acoustic_features"],
            "arousal_label": t["arousal_label"],
            "text_emotion": t["text_emotion"],
        }
        for t in turns
    ]
    user_content = "Conversation turns:\n\n" + json.dumps(transcript_for_model, indent=2)

    raw = call_ollama_chat(
        [{"role": "user", "content": user_content}], REPORT_SYSTEM_PROMPT, format_json=True
    )

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    result = json.loads(cleaned.strip())

    result["patient_message"] = redistribute_trailing_emoji(result["patient_message"])

    if should_nudge_toward_doctor(result["turns"]):
        result["patient_message"] += " If this keeps building, it could help to flag it with your doctor too."

    elapsed_by_index = {t["turn_index"]: t["elapsed_seconds"] for t in turns}
    for turn in result["turns"]:
        turn["elapsed_seconds"] = elapsed_by_index.get(turn["turn_index"], 0)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(result, f, indent=4)

    xs = [t["elapsed_seconds"] for t in result["turns"]]
    ys = [t["distress_score"] for t in result["turns"]]
    plt.figure(figsize=(9, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.ylim(0, 10)
    plt.xlabel("Time into session (s)")
    plt.ylabel("Distress / stress indicator (0-10)")
    plt.title("Session distress indicator over time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    graph_path = os.path.join(output_dir, "mental_state.png")
    plt.savefig(graph_path, dpi=150)
    plt.close()

    print(f"\nReport written to {report_path}")
    print(f"Graph written to {graph_path}")
    print("\nClinician summary:\n", result["clinician_summary"])
    print("\nClosing message:\n", result["patient_message"])


if __name__ == "__main__":
    whisper_model, emotion_classifier = load_models()
    temp_dir = os.path.join(SCRIPT_DIR, "tmp")
    turns = run_conversation(whisper_model, emotion_classifier, temp_dir)

    if not turns:
        print("No turns recorded, nothing to report on.")
        sys.exit(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(SCRIPT_DIR, "output", f"session_{timestamp}")
    generate_report(turns, output_dir)