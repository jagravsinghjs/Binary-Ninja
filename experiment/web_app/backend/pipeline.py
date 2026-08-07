"""
pipeline.py
Same inference pipeline as voice_chat.py (Whisper -> acoustic features -> text
emotion -> Ollama), refactored so the FastAPI backend can call it per-request
instead of holding conversation state in a global python list. Conversation
history is rebuilt from whatever's in SQLite for that session_id, which is
what makes this safe across multiple concurrent users/sessions.

v2 change — generate_report() robustness:
qwen2.5:7b-instruct is a small local model, and even with format_json=True
it doesn't always return clean JSON — a truncated response, stray text
wrapped around the object, or an outright malformed reply all used to blow
up generate_report() with an unhandled exception, which (before main.py's
matching fix) meant the report silently never got saved. Now: one retry
with a stricter reminder if the first parse fails, a salvage pass that
extracts the {...} substring if the model added preamble/fences despite
being told not to, and a check that the required keys are actually present
before handing the result back — so a genuine failure raises one clear,
readable error instead of a confusing downstream KeyError.
"""

import json
import os
import re
import subprocess
import tempfile

import edge_tts
import numpy as np
import requests
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
)

# librosa/audioread need a system `ffmpeg` on PATH to decode non-WAV formats
# (like the browser's .webm uploads) — and it specifically has to be a file
# literally named `ffmpeg`/`ffmpeg.exe`, found via shutil.which(). Prepending
# imageio-ffmpeg's bundled binary's folder to PATH doesn't work because that
# binary has a versioned filename (e.g. ffmpeg-win-x86_64-v7.0.2.exe), not
# `ffmpeg.exe` — so PATH lookup never finds it no matter where it sits.
# Instead: call that exact binary directly via subprocess to convert the
# upload to a plain .wav before anything else touches it. WAV needs no
# ffmpeg at all — soundfile reads it natively. (Whisper was never actually
# affected by this — faster-whisper decodes via its own bundled PyAV, not
# system ffmpeg — only the acoustic-feature step needed this.)
import imageio_ffmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

from faster_whisper import WhisperModel
from transformers import pipeline as hf_pipeline

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b-instruct"

OPENER = "Hey, how's it going? What's on your mind today?"

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

REPORT_RETRY_REMINDER = """Your previous reply could not be parsed as JSON. Return ONLY the JSON
object described below — no markdown code fences, no preamble, no trailing commentary, nothing
before the opening brace or after the closing brace:
{
  "turns": [
    {"turn_index": <int>, "distress_score": <int 0-10>, "note": "<short phrase why>"}
  ],
  "clinician_summary": "<string>",
  "patient_message": "<string>"
}
"""


# ------------------------------------------------------------------
# Model loading (once, at FastAPI startup)
# ------------------------------------------------------------------

def load_models():
    whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")
    emotion_classifier = hf_pipeline(
        task="text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=0,
    )
    return whisper_model, emotion_classifier


# ------------------------------------------------------------------
# Acoustic feature extraction (identical logic to voice_chat.py)
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
# Per-turn processing (runs on a worker thread via asyncio.to_thread —
# synchronous and GPU-bound, same as it was in the CLI)
# ------------------------------------------------------------------

def convert_to_wav(input_path):
    """Converts any browser-uploaded audio (webm/ogg/whatever) to a plain
    16kHz mono WAV by calling imageio-ffmpeg's exact binary path directly —
    no PATH lookup, no shutil.which(), so it can't be broken by a filename
    mismatch or a Windows PATH that didn't take. Caller is responsible for
    deleting the returned path when done."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        FFMPEG_EXE, "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-f", "wav", wav_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        os.remove(wav_path)
        raise RuntimeError(
            f"ffmpeg failed to convert uploaded audio: {result.stderr.decode(errors='replace')}"
        )
    return wav_path


def process_turn(whisper_model, emotion_classifier, audio_path):
    wav_path = convert_to_wav(audio_path)
    try:
        segments, info = whisper_model.transcribe(wav_path, beam_size=5)
        text = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            return None

        acoustic = extract_acoustic_features(wav_path)
        arousal = classify_arousal(acoustic)

        emotion_raw = emotion_classifier(text)[0]
        text_emotion = {item["label"]: round(item["score"], 4) for item in emotion_raw}

        return {
            "text": text,
            "acoustic_features": acoustic,
            "arousal_label": arousal,
            "text_emotion": text_emotion,
        }
    finally:
        os.remove(wav_path)  # the converted copy never outlives this turn either


def format_user_turn(turn):
    """turn: dict with text / acoustic_features / arousal_label / text_emotion
    (works whether it just came off process_turn() or was rehydrated from DB)."""
    top_emotions = sorted(turn["text_emotion"].items(), key=lambda x: -x[1])[:3]
    context_note = (
        f"[voice cues: pitch_std={turn['acoustic_features'].get('pitch_std_hz', 0)}, "
        f"energy_std={turn['acoustic_features'].get('energy_std', 0)}, "
        f"pause_ratio={turn['acoustic_features'].get('pause_ratio', 0)}, "
        f"arousal={turn['arousal_label']}, top_text_emotion={top_emotions}]"
    )
    return f"{turn['text']}\n{context_note}"


def build_message_history(db_turns):
    """Rebuilds the Ollama message list from persisted turns instead of an
    in-memory python list. This is the core multi-user fix: conversation
    state lives in SQLite, keyed by session_id, so it survives a backend
    restart and never leaks between sessions or users."""
    messages = [{"role": "assistant", "content": OPENER}]
    for t in db_turns:
        acoustic = json.loads(t["acoustic_features"]) if t["acoustic_features"] else {}
        text_emotion = json.loads(t["text_emotion"]) if t["text_emotion"] else {}
        fake_turn = {
            "text": t["text"],
            "acoustic_features": acoustic,
            "arousal_label": t["arousal_label"],
            "text_emotion": text_emotion,
        }
        messages.append({"role": "user", "content": format_user_turn(fake_turn)})
        if t["assistant_reply"]:
            messages.append({"role": "assistant", "content": t["assistant_reply"]})
    return messages


# ------------------------------------------------------------------
# Ollama
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
        raise RuntimeError(f"Ollama did not return message.content: {json.dumps(data)}")

    return data["message"]["content"]


# ------------------------------------------------------------------
# Post-processing (identical to voice_chat.py)
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
# Report generation — takes DB turn rows, returns a report dict
# (persistence and graph file writing are left to the caller)
# ------------------------------------------------------------------

def _extract_json_object(raw):
    """Best-effort salvage for when the model wraps the JSON in markdown
    fences or stray commentary despite format_json=True and being told not
    to. Strips ``` fences if present, then — if the whole string still
    doesn't parse — falls back to the substring between the first '{' and
    the last '}'."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        return json.loads(candidate)  # let this raise if it's still bad

    raise json.JSONDecodeError("No JSON object found in model output", cleaned, 0)


def _parse_report_json(raw):
    if not raw.strip():
        raise RuntimeError(
            "Ollama returned an empty response — likely a timeout or the "
            "model erroring mid-generation."
        )
    result = _extract_json_object(raw)  # may raise json.JSONDecodeError

    required = {"turns", "clinician_summary", "patient_message"}
    missing = required - result.keys()
    if missing:
        raise RuntimeError(f"Model's JSON was missing required field(s): {', '.join(missing)}")
    if not isinstance(result["turns"], list) or not result["turns"]:
        raise RuntimeError("Model's JSON had an empty or invalid 'turns' list")
    for t in result["turns"]:
        if "turn_index" not in t or "distress_score" not in t:
            raise RuntimeError("Model's JSON had a turn missing 'turn_index' or 'distress_score'")

    return result


def generate_report(db_turns):
    transcript_for_model = [
        {
            "turn_index": t["turn_index"],
            "text": t["text"],
            "acoustic_features": json.loads(t["acoustic_features"]) if t["acoustic_features"] else {},
            "arousal_label": t["arousal_label"],
            "text_emotion": json.loads(t["text_emotion"]) if t["text_emotion"] else {},
        }
        for t in db_turns
    ]
    user_content = "Conversation turns:\n\n" + json.dumps(transcript_for_model, indent=2)
    messages = [{"role": "user", "content": user_content}]

    # Small local models occasionally don't produce clean JSON on the first
    # try. One retry with a stricter reminder — appended as the model's own
    # prior turn plus a follow-up instruction — resolves most of these
    # without the person ever seeing a failure.
    last_error = None
    raw = None
    for attempt in range(2):
        try:
            raw = call_ollama_chat(messages, REPORT_SYSTEM_PROMPT, format_json=True)
            result = _parse_report_json(raw)
            break
        except (json.JSONDecodeError, RuntimeError) as e:
            last_error = e
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": raw or ""},
                    {"role": "user", "content": REPORT_RETRY_REMINDER},
                ]
                continue
            debug_path = os.path.join(tempfile.gettempdir(), "setu_last_raw_response.txt")
            with open(debug_path, "w") as f:
                f.write(raw or "")
            raise RuntimeError(
                f"Ollama's report output wasn't usable after a retry ({last_error}) — "
                f"raw output saved to {debug_path}"
            )

    result["patient_message"] = redistribute_trailing_emoji(result["patient_message"])

    if should_nudge_toward_doctor(result["turns"]):
        result["patient_message"] += " If this keeps building, it could help to flag it with your doctor too."

    elapsed_by_index = {t["turn_index"]: t["elapsed_seconds"] for t in db_turns}
    for turn in result["turns"]:
        turn["elapsed_seconds"] = elapsed_by_index.get(turn["turn_index"], 0)

    return result


# ------------------------------------------------------------------
# Text-to-speech (edge-tts — a network call to Microsoft's TTS service,
# not local GPU work, so this does NOT need to go behind inference_lock
# in main.py; it doesn't compete with Whisper/emotion/Ollama for the GPU)
# ------------------------------------------------------------------

EDGE_TTS_VOICE = "en-US-AriaNeural"


async def synthesize_speech(text, out_path):
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    await communicate.save(out_path)


def make_graph(turns, out_path):
    xs = [t["elapsed_seconds"] for t in turns]
    ys = [t["distress_score"] for t in turns]
    plt.figure(figsize=(9, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.ylim(0, 10)
    plt.xlabel("Time into session (s)")
    plt.ylabel("Distress / stress indicator (0-10)")
    plt.title("Session distress indicator over time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ------------------------------------------------------------------
# PDF export — full session detail (transcript + distress timeline +
# clinician summary + patient message) as a single downloadable file.
# Not a doctor-portal feature: no new auth/roles, just an artifact the
# patient can save or forward however they choose (e.g. email it to
# their doctor themselves).
# ------------------------------------------------------------------

def generate_session_pdf(session, user_email, db_turns, report, out_path):
    """session: dict from db.get_session (id, started_at, ended_at, status).
    user_email: the owning patient's email, for a header line only.
    db_turns: list of turn rows from db.get_turns (has the transcript text).
    report: dict from db.get_report (clinician_summary, patient_message,
            graph_path, turns=[{turn_index, distress_score, note, elapsed_seconds}])."""
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=8
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"], fontSize=8, leading=11, textColor=colors.grey
    )

    text_by_index = {t["turn_index"]: t["text"] for t in db_turns}

    doc = SimpleDocTemplate(
        out_path, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    story = []

    story.append(Paragraph("Setu Session Report", title_style))
    story.append(Paragraph(
        f"Patient: {user_email} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Session started: {session['started_at']} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Ended: {session.get('ended_at') or 'in progress'}",
        small_style,
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(
        "This is a decision-support artifact generated from a voice check-in between "
        "doctor visits. It is not a diagnosis.",
        small_style,
    ))
    story.append(Spacer(1, 0.15 * inch))

    if report.get("graph_path") and os.path.exists(report["graph_path"]):
        story.append(RLImage(report["graph_path"], width=6.5 * inch, height=3.25 * inch))
        story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Clinician Summary", heading_style))
    story.append(Paragraph(report["clinician_summary"], body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Message Shown to Patient", heading_style))
    story.append(Paragraph(report["patient_message"], body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Turn-by-Turn Detail", heading_style))
    table_data = [["#", "Time (s)", "Distress", "What was said", "Note"]]
    for t in sorted(report["turns"], key=lambda x: x["turn_index"]):
        transcript = text_by_index.get(t["turn_index"], "")
        if len(transcript) > 200:
            transcript = transcript[:200] + "..."
        table_data.append([
            str(t["turn_index"]),
            f"{t.get('elapsed_seconds', 0):.0f}",
            str(t["distress_score"]),
            Paragraph(transcript, body_style),
            Paragraph(t.get("note", ""), body_style),
        ])

    table = Table(table_data, colWidths=[0.3 * inch, 0.6 * inch, 0.6 * inch, 3.2 * inch, 1.8 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f6f5e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e3df")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f5f3")]),
    ]))
    story.append(table)

    doc.build(story)