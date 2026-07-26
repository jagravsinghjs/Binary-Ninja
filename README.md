# Setu

**A voice check-in app that protects the 15 minutes a psychiatrist actually has with a patient.**

## How to run this

### Prerequisites (one-time)
- A **CUDA-capable GPU** — Whisper and the emotion classifier both load onto GPU.
- **[Ollama](https://ollama.com)** installed, with the model pulled:
  ```bash
  ollama pull qwen2.5:7b-instruct
  ```
- **ffmpeg** on your system PATH (needed to decode browser-recorded audio).

### Steps

```bash
# 1. Open the project folder
cd ~/Binary-Ninja

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Go to the backend
cd experiment/web_app/backend

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Open it

Go to **http://localhost:8000** in a browser. The backend serves the frontend on the same
origin — no separate frontend server, no CORS setup needed.

Sign up with any email + password (8+ characters) to create an account, then start a session
and start talking. End the session to generate the report, distress graph, and PDF export.

---

## The problem

In India, seeing a psychiatrist is still stigmatised — people treat it as an admission of being
"abnormal," when really it's for everyday emotional struggles: breakups, academic pressure,
loneliness, burnout. Even at top institutions, students under real distress often don't talk to
a doctor, parent, or friend about it.

For those who do see a psychiatrist, the system is broken in a quieter way: appointments are
15 minutes, once a week or once a month. Most of that time gets spent on the patient trying to
recall — often inaccurately — how they've felt over the past few weeks. The actual clinical
conversation barely happens.

## The idea

We are not replacing the psychiatrist. **We are protecting their 15 minutes.**

Between doctor visits, the patient can check in — as often as they like, no pressure, no one
watching live — by talking to Setu. It listens not just to *what* they say but *how* they say
it, since people in distress often say "I'm fine" while everything about their voice says
otherwise.

We are not diagnosing. We are not scoring illness. **We are building a memory the doctor can
use.**

## How it works

1. **Patient talks** — records a short voice note through the browser (mic capture, no app
   install needed).
2. **Whisper** (`faster-whisper`) transcribes the audio.
3. **librosa** extracts acoustic features from the same audio — pitch mean/variability, energy,
   zero-crossing rate, spectral centroid, pause ratio — and a heuristic arousal label
   (`neutral` / `high_arousal` / `low_arousal`).
4. A **text-emotion classifier** (`j-hartmann/emotion-english-distilroberta-base`) scores the
   transcribed text itself for emotion.
5. Transcript + acoustic features + text emotion are sent to a **local LLM**
   (`qwen2.5:7b-instruct` via Ollama), which holds a live conversation with the patient, turn
   by turn — listening and asking genuine follow-ups, never rushing to advice or wrapping the
   conversation up early.
6. At the end of a session, the same pipeline generates:
   - a **distress-score timeline** (0–10 per turn, plotted as a graph)
   - a **clinician-facing summary** (decision-support language — "may indicate," "consider
     exploring" — never a diagnosis)
   - a short **patient-facing message** — warm, honest, no false reassurance, occasionally
     nudging toward the doctor if the distress trend is genuinely worsening (that decision is
     made deterministically from the score trend, not left to the LLM)
   - the message can also be **spoken aloud** (`edge-tts`)
   - the full session exports as a **PDF** (graph + summary + message + turn-by-turn table) via
     `reportlab`

## Stack

| Layer | Tech |
|---|---|
| Transcription | `faster-whisper` (medium, GPU) |
| Acoustic features | `librosa` |
| Text emotion | `transformers` — `j-hartmann/emotion-english-distilroberta-base` (GPU) |
| Conversation / report generation | Ollama, `qwen2.5:7b-instruct` |
| Text-to-speech | `edge-tts` |
| PDF export | `reportlab` |
| Backend | FastAPI (`main.py`), SQLite (`db.py`), core inference in `pipeline.py` |
| Auth | Real accounts — email + password, PBKDF2 hashing (stdlib), bearer token — each user only sees their own sessions |
| Frontend | Static HTML/JS (`index.html`) — login/signup, mic recording, live chat, session sidebar, report modal, TTS playback, PDF export |

**Privacy by design**: raw audio is never persisted. A temp file is written per turn and
deleted immediately after processing — only transcript text, acoustic features, emotion scores,
and the final report are stored in `setu.db`.

**GPU concurrency**: Whisper, the emotion classifier, and Ollama calls are serialized behind a
single `asyncio.Lock` with a bounded queue (`MAX_QUEUE_DEPTH = 3`) — a request arriving while the
queue is full gets an immediate `503` instead of hanging indefinitely.

## Project layout

```
experiment/
├── web_app/
│   ├── backend/
│   │   ├── main.py          FastAPI app, routes
│   │   ├── db.py             SQLite models/queries
│   │   ├── security.py       Password hashing + auth tokens
│   │   ├── pipeline.py        Core inference: transcription, features, LLM calls
│   │   └── requirements.txt
│   └── frontend/
│       └── index.html         Login/signup, mic recording, chat, reports, TTS, PDF export
│
├── speech_to_text/, voice_emotion/, text_emotion/, chat_llm/, voice_chat/
│                             Early prototype scripts, kept for reference — each stage of the
│                             pipeline (transcription, acoustic features, text emotion, report
│                             generation) was validated independently here before being merged
│                             into web_app above.

documentation/                Project write-up: problem statement, architecture, DFDs, use case,
                               algorithms, limitations, future work.
```

## Not yet built / future scope

- **Remote/public access** — currently local-only by design. A public link (for demos, or for
  patients to use somewhere the GPU isn't) would need a tunnel (Cloudflare Tunnel or Tailscale
  Funnel) in front of the local backend, since inference has to stay on GPU hardware.
- **Doctor-facing portal** — deliberately out of scope so far; the patient-generated PDF export
  is the current substitute for getting a summary to a clinician.
- **Token expiry / revocation** — bearer tokens currently don't expire.
- **Emoji-in-PDF rendering** — emoji in the patient message currently render as broken glyphs in
  the exported PDF (Helvetica has no emoji glyphs).
- **Windows `.webm` upload handling** — needs ffmpeg present; current fix calls
  `imageio_ffmpeg.get_ffmpeg_exe()`'s path directly via `subprocess` rather than relying on
  system PATH.
- **Load testing** — GPU concurrency handling hasn't been tested under real concurrent
  multi-user load.
- **UI visual polish.**
- **Data retention/deletion controls** — transcripts, acoustic features, and reports persist
  indefinitely in `setu.db`; no per-session delete endpoint yet.
- Full list of longer-term future work (multilingual support, doctor dashboards, cross-session
  trend view, video-based emotion signal, etc.) is in `documentation/future_work.md`.

## Limitations

Design trade-offs that come with running local/private/offline rather than gaps more time would
close — hardware dependency (CUDA GPU required), a capability ceiling from using a small local
LLM instead of a cloud model, no clinical validation of the distress scoring, English-only
support in practice, and more. Full list in `documentation/limitations.md`.