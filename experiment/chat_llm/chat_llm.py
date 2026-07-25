"""
chat_llm.py
Reads transcript_with_emotion.json (whisper text + acoustic features),
sends it to a local Ollama model, and produces:
  1. output/report.json      - per-segment stress/mood indicator timeline
                                + a clinician-facing summary
                                + a short, honest (not falsely reassuring) message for the patient
  2. output/mental_state.png - line graph: x = timestamp (s), y = distress/stress indicator (0-10)

Usage:
    python chat_llm.py
    python chat_llm.py path/to/transcript_with_emotion.json path/to/output_dir
"""

import json
import os
import sys
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b-instruct"

SYSTEM_PROMPT = """You are a clinical decision-support assistant, NOT a therapist and NOT a diagnostic tool.
You are given a transcript of a patient's voice-note reflections, segment by segment, along with
acoustic features (pitch variability, energy variability, pause ratio, arousal label) extracted from
their speech.

Your job:
1. For EACH segment, assign a "distress_score" from 0 (calm/settled) to 10 (highly distressed/agitated).
   Base this on BOTH the words used and the acoustic cues. Do not just copy the arousal_label — use it
   as one signal among several.
2. Write a short "clinician_summary" (3-5 sentences) describing the patient's apparent trajectory across
   the session (e.g. "started elevated, settled somewhat by segment 4"), notable phrases, and any
   acoustic patterns worth a clinician's attention. This is a decision-support note, not a diagnosis —
   phrase it that way (e.g. "may indicate", "consider exploring", not "patient has X disorder").
3. Write a "patient_message" (4-6 sentences, ~60-90 words, warm, casual, plain language, 2-4 emoji placed
   mid-sentence right next to the word/feeling they relate to — NOT all bunched at the very end) to show
   the patient. This should read like a friend who actually listened, NOT like a hospital notice.
   Use only single, simple emoji characters (e.g. 😥 💪 🙂 😮) — never combine two emoji together with a
   joiner, that produces broken/garbled symbols.
   Example of GOOD emoji placement: "Sounds like today was a lot 😥 but you pushed through it, and
   that says something 💪. Proud of you for showing up."
   Example of BAD emoji placement (decoration tacked on at the end): "Sounds like today was a lot but
   you pushed through it. Proud of you for showing up. 😥💪"
   Rules:
   - Validate what they actually said — do not invent feelings they didn't express.
   - Do NOT just tell them what they want to hear. Do not give false reassurance ("everything is fine!")
     if the content suggests real distress.
   - Do NOT diagnose or use clinical labels.
   - You CAN offer one small, casual, everyday suggestion — but ONLY if it is clearly tailored to what
     THIS person specifically described, not a generic default. Do not reach for "go for a walk" or
     "talk to a friend" as a template regardless of content. Examples of matching content to suggestion:
     if they mentioned trouble sleeping, a suggestion should relate to rest, not movement; if they
     mentioned friend conflict or not wanting to involve people, do NOT suggest talking to a friend;
     if nothing in what they said points to a natural, specific suggestion, SKIP the suggestion
     entirely — silence is better than a generic one that doesn't fit their actual situation.
   - NEVER suggest medication, supplements, specific therapy techniques (e.g. CBT exercises), or anything
     that sounds like a treatment protocol. That is not your role.
   - Do NOT mention "doctor," "appointment," "clinician," or "review" in this message under any
     circumstances — that decision is handled separately, outside of what you write.
   - Only if the content shows signs of real, significant distress (not everyday stress) should you gently
     suggest they lean on someone they trust or a support line — and even then, phrase it like a friend
     would, not like a referral. Otherwise skip that entirely and just respond to what they said.

Return ONLY valid JSON, no markdown fences, no preamble, in exactly this shape:
{
  "segments": [
    {"start": <float>, "end": <float>, "distress_score": <int 0-10>, "note": "<1 short phrase why>"}
  ],
  "clinician_summary": "<string>",
  "patient_message": "<string>"
}
"""


def call_ollama(transcript_segments):
    user_content = "Transcript segments with acoustic features:\n\n" + json.dumps(
        transcript_segments, indent=2
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "format": "json",  # ask ollama to constrain to valid JSON
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    if "message" not in data or "content" not in data.get("message", {}):
        print("Unexpected Ollama response shape:")
        print(json.dumps(data, indent=2))
        raise RuntimeError("Ollama did not return message.content — see raw response above")

    raw = data["message"]["content"]

    if not raw.strip():
        print("Full raw Ollama response object:")
        print(json.dumps(data, indent=2))
        raise RuntimeError(
            "Ollama returned an empty content string. Common causes: request timed out "
            "mid-generation, model errored internally, or 'format: json' rejected the prompt. "
            "Check the object printed above (look at 'done_reason' / 'total_duration')."
        )

    # strip markdown fences if the model added them despite format=json
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # save what we got so it's inspectable instead of losing it
        debug_path = os.path.join(SCRIPT_DIR, "output", "_last_raw_response.txt")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        with open(debug_path, "w") as f:
            f.write(raw)
        print(f"Could not parse model output as JSON. Raw output saved to {debug_path}")
        raise


def make_graph(segments, out_path):
    xs = [s["start"] for s in segments]
    ys = [s["distress_score"] for s in segments]

    plt.figure(figsize=(9, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.ylim(0, 10)
    plt.xlabel("Time (s)")
    plt.ylabel("Distress / stress indicator (0-10)")
    plt.title("Session distress indicator over time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def redistribute_trailing_emoji(text):
    """
    If the model dumps all emoji at the very end of the message (common failure
    mode for smaller models despite prompt instructions), pull them off and
    interleave one after each sentence instead. If emoji are already spread
    out, leave the text untouched.
    """
    import re

    # defensive cleanup: strip zero-width joiners between emoji, since a small
    # model will sometimes glue two unrelated emoji together into a broken glyph
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

    # emoji block sitting at the very end of the string (allowing trailing whitespace)
    trailing_match = re.search(r"(\s*(?:" + emoji_pattern.pattern + r"\s*)+)$", text)
    if not trailing_match:
        return text  # nothing bunched at the end, assume it's already interleaved

    trailing_block = trailing_match.group(0)
    emojis = single_emoji_pattern.findall(trailing_block)  # individual emoji, not merged runs
    if len(emojis) < 2:
        return text  # a single trailing emoji is fine, not worth rewriting

    body = text[: trailing_match.start()].strip()

    # naive sentence split on ., !, ? followed by a space
    sentences = re.split(r"(?<=[.!?])\s+", body)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return text

    rebuilt = []
    for i, sentence in enumerate(sentences):
        if i < len(emojis):
            # insert before the final punctuation mark of the sentence
            m = re.match(r"^(.*?)([.!?]?)$", sentence)
            core, punct = m.group(1), m.group(2)
            sentence = f"{core} {emojis[i]}{punct}"
        rebuilt.append(sentence)

    return " ".join(rebuilt)


def should_nudge_toward_doctor(segments):
    """
    Deterministic check based on the actual distress_score numbers, not the
    model's judgment. Nudge toward the doctor only when the trend is
    genuinely climbing (things are getting worse, not better) or the session
    ends at a high level — i.e. exactly when a friendly message alone isn't
    really enough.
    """
    scores = [s["distress_score"] for s in segments]
    if len(scores) < 2:
        return False

    mid = len(scores) // 2
    first_half = scores[:mid] or [scores[0]]
    second_half = scores[mid:]
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)

    rising_trend = (avg_second - avg_first) >= 1.5
    ends_high = scores[-1] >= 6

    return rising_trend or ends_high


def main(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r") as f:
        transcript_segments = json.load(f)

    result = call_ollama(transcript_segments)

    if "patient_message" in result:
        result["patient_message"] = redistribute_trailing_emoji(result["patient_message"])

    if "segments" in result and should_nudge_toward_doctor(result["segments"]):
        result["patient_message"] += (
            " If this keeps building, it could help to flag it with your doctor too."
        )

    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(result, f, indent=4)

    graph_path = os.path.join(output_dir, "mental_state.png")
    make_graph(result["segments"], graph_path)

    print(f"Report written to {report_path}")
    print(f"Graph written to {graph_path}")
    print("\nClinician summary:\n", result["clinician_summary"])
    print("\nPatient message:\n", result["patient_message"])


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        SCRIPT_DIR, "input", "transcript_with_emotion.json"
    )
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SCRIPT_DIR, "output")
    main(input_path, output_dir)