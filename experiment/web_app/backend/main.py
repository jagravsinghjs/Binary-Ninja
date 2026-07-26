"""
main.py
FastAPI backend for the Setu voice check-in webapp.

v2 changes (moving from "one laptop, one demo user" to a public URL):

1. Real accounts. `user_id` used to be a UUID the frontend picked for itself
   and sent on every request — meaning anyone who guessed or intercepted a
   UUID could read or write that user's sessions. Now every session-touching
   route requires a bearer token (Authorization: Bearer <token>), resolved
   to a user via get_current_user(). Sessions are also ownership-checked:
   a token can only touch sessions belonging to the same user_id.
   THIS is the fix that actually matters for going public — more than which
   tunnel you pick.

2. Bounded queue instead of a bare lock. inference_lock still serializes
   Whisper / emotion classifier / Ollama calls (one shared GPU), but now a
   request that arrives while MAX_QUEUE_DEPTH others are already waiting
   gets an immediate 503 instead of hanging indefinitely. With one demo
   user that never mattered; with a public link and real concurrent
   traffic, an unbounded queue just means everyone times out together
   instead of most people getting a fast, clear "try again in a moment."

3. TTS endpoints call pipeline.synthesize_speech directly, NOT through
   inference_lock — edge-tts is a network call to Microsoft's service, not
   local GPU work, so it doesn't compete with Whisper/Ollama for the GPU
   and serializing it would only add needless latency.

Raw audio: still never persisted — temp file per turn, deleted in a
finally block, unchanged from v1.
"""

import asyncio
import os
import tempfile
import uuid
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Form, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import db
import pipeline
import security

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
FRONTEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "frontend"))

MAX_QUEUE_DEPTH = 3  # requests waiting on GPU inference before we start rejecting

app = FastAPI(title="Setu voice check-in")

# Frontend is now commonly hosted SEPARATELY from this backend (static
# host + tunnel), so this is cross-origin in the real deployment, not just
# local dev. Set SETU_ALLOWED_ORIGIN to your static-host URL (e.g.
# https://setu-demo.pages.dev) before sharing the tunnel link with judges;
# falls back to "*" so local dev / same-origin still works untouched.
_allowed_origin = os.environ.get("SETU_ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_allowed_origin] if _allowed_origin != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

whisper_model = None
emotion_classifier = None
inference_lock = asyncio.Lock()
queue_depth = 0  # plain int is safe here — single-threaded asyncio event loop


@app.on_event("startup")
def _startup():
    global whisper_model, emotion_classifier
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    db.init_db()
    print("Loading models (this can take a minute)...")
    whisper_model, emotion_classifier = pipeline.load_models()
    print("Models loaded.")


def _now():
    return datetime.now(timezone.utc).isoformat()


async def run_gpu_bound(fn, *args):
    """Runs fn(*args) on a worker thread, serialized behind inference_lock,
    with a bounded wait queue so this fails fast under load instead of
    piling up requests indefinitely."""
    global queue_depth
    if queue_depth >= MAX_QUEUE_DEPTH:
        raise HTTPException(
            503, "Setu is busy with another check-in right now — try again in a moment."
        )
    queue_depth += 1
    try:
        async with inference_lock:
            return await asyncio.to_thread(fn, *args)
    finally:
        queue_depth -= 1


# ------------------------------------------------------------------
# Health (no auth — just "is the tunnel/backend up", used by the
# frontend's connect screen and optionally to show queue state)
# ------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "queue_depth": queue_depth,
        "busy": queue_depth >= MAX_QUEUE_DEPTH,
    }


# ------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------

def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing or invalid Authorization header")
    token = authorization[len("Bearer "):]
    user = db.get_user_by_token(token)
    if not user:
        raise HTTPException(401, "invalid or expired token")
    return user


def _owned_session_or_404(session_id: str, current_user: dict) -> dict:
    session = db.get_session(session_id)
    if not session or session["user_id"] != current_user["id"]:
        # 404, not 403 — don't confirm to a stranger that a session id exists
        raise HTTPException(404, "session not found")
    return session


@app.post("/api/auth/signup")
def signup(email: str = Form(...), password: str = Form(...)):
    if len(password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    if db.get_user_by_email(email):
        raise HTTPException(400, "an account with that email already exists")
    user_id = uuid.uuid4().hex
    db.create_account(user_id, email, security.hash_password(password), _now())
    token = security.new_token()
    db.create_token(token, user_id, _now())
    return {"token": token, "user_id": user_id, "email": email.lower().strip()}


@app.post("/api/auth/login")
def login(email: str = Form(...), password: str = Form(...)):
    user = db.get_user_by_email(email)
    if not user or not security.verify_password(password, user["password_hash"]):
        raise HTTPException(401, "incorrect email or password")
    token = security.new_token()
    db.create_token(token, user["id"], _now())
    return {"token": token, "user_id": user["id"], "email": user["email"]}


@app.post("/api/auth/logout")
def logout(authorization: str = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        db.delete_token(authorization[len("Bearer "):])
    return {"ok": True}


# ------------------------------------------------------------------
# Sessions (all require a valid token; all ownership-checked)
# ------------------------------------------------------------------

@app.get("/api/sessions")
def list_sessions(current_user: dict = Depends(get_current_user)):
    return db.list_sessions_for_user(current_user["id"])


@app.post("/api/sessions")
def start_session(current_user: dict = Depends(get_current_user)):
    session_id = uuid.uuid4().hex
    db.create_session(session_id, current_user["id"], _now())
    return {"session_id": session_id, "opener": pipeline.OPENER}


@app.get("/api/sessions/{session_id}")
def get_session_detail(session_id: str, current_user: dict = Depends(get_current_user)):
    session = _owned_session_or_404(session_id, current_user)
    session["turns"] = db.get_turns(session_id)
    return session


@app.post("/api/sessions/{session_id}/turns")
async def submit_turn(
    session_id: str,
    audio: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    session = _owned_session_or_404(session_id, current_user)
    if session["status"] != "active":
        raise HTTPException(400, "session already ended")

    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await audio.read())
        turn = await run_gpu_bound(
            pipeline.process_turn, whisper_model, emotion_classifier, tmp_path
        )
    finally:
        os.remove(tmp_path)  # raw audio never survives past this request

    if turn is None:
        raise HTTPException(422, "No speech detected, try again")

    existing_turns = db.get_turns(session_id)
    turn_index = len(existing_turns)
    started = datetime.fromisoformat(session["started_at"])
    elapsed = round((datetime.now(timezone.utc) - started).total_seconds(), 1)

    messages = pipeline.build_message_history(existing_turns)
    messages.append({"role": "user", "content": pipeline.format_user_turn(turn)})

    reply = await run_gpu_bound(pipeline.call_ollama_chat, messages, pipeline.CONVO_SYSTEM_PROMPT)

    db.insert_turn(
        session_id, turn_index, elapsed, turn["text"], reply,
        turn["acoustic_features"], turn["arousal_label"], turn["text_emotion"], _now(),
    )

    return {
        "turn_index": turn_index,
        "text": turn["text"],
        "assistant_reply": reply,
        "elapsed_seconds": elapsed,
    }


def _audio_path(session_id: str, key: str) -> str:
    return os.path.join(AUDIO_DIR, f"{session_id}_{key}.mp3")


@app.get("/api/sessions/{session_id}/opener-audio")
async def get_opener_audio(session_id: str, current_user: dict = Depends(get_current_user)):
    _owned_session_or_404(session_id, current_user)
    path = _audio_path(session_id, "opener")
    if not os.path.exists(path):
        await pipeline.synthesize_speech(pipeline.OPENER, path)
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/api/sessions/{session_id}/turns/{turn_index}/audio")
async def get_turn_audio(
    session_id: str, turn_index: int, current_user: dict = Depends(get_current_user)
):
    _owned_session_or_404(session_id, current_user)
    turns = db.get_turns(session_id)
    match = next((t for t in turns if t["turn_index"] == turn_index), None)
    if not match or not match["assistant_reply"]:
        raise HTTPException(404, "no reply audio for this turn")
    path = _audio_path(session_id, f"turn{turn_index}")
    if not os.path.exists(path):
        await pipeline.synthesize_speech(match["assistant_reply"], path)
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/api/sessions/{session_id}/report-audio")
async def get_report_audio(session_id: str, current_user: dict = Depends(get_current_user)):
    _owned_session_or_404(session_id, current_user)
    report = db.get_report(session_id)
    if not report:
        raise HTTPException(404, "no report for this session yet")
    path = _audio_path(session_id, "report")
    if not os.path.exists(path):
        await pipeline.synthesize_speech(report["patient_message"], path)
    return FileResponse(path, media_type="audio/mpeg")


@app.post("/api/sessions/{session_id}/end")
async def end_session(session_id: str, current_user: dict = Depends(get_current_user)):
    session = _owned_session_or_404(session_id, current_user)
    if session["status"] != "active":
        report = db.get_report(session_id)
        if report:
            return report
        raise HTTPException(400, "session already ended with no report")

    turns = db.get_turns(session_id)
    if not turns:
        raise HTTPException(400, "no turns recorded yet")

    result = await run_gpu_bound(pipeline.generate_report, turns)

    graph_path = os.path.join(OUTPUT_DIR, f"{session_id}_mental_state.png")
    pipeline.make_graph(result["turns"], graph_path)

    db.save_report(
        session_id, result["clinician_summary"], result["patient_message"], graph_path, _now()
    )
    db.save_report_segments(session_id, result["turns"])
    db.end_session(session_id, _now())

    return db.get_report(session_id)


@app.get("/api/sessions/{session_id}/report")
def get_report(session_id: str, current_user: dict = Depends(get_current_user)):
    _owned_session_or_404(session_id, current_user)
    report = db.get_report(session_id)
    if not report:
        raise HTTPException(404, "no report for this session yet")
    return report


@app.get("/api/sessions/{session_id}/graph")
def get_graph(session_id: str, current_user: dict = Depends(get_current_user)):
    _owned_session_or_404(session_id, current_user)
    report = db.get_report(session_id)
    if not report or not report.get("graph_path") or not os.path.exists(report["graph_path"]):
        raise HTTPException(404, "no graph for this session")
    return FileResponse(report["graph_path"], media_type="image/png")


@app.get("/api/sessions/{session_id}/report-pdf")
def get_report_pdf(session_id: str, current_user: dict = Depends(get_current_user)):
    session = _owned_session_or_404(session_id, current_user)
    report = db.get_report(session_id)
    if not report:
        raise HTTPException(404, "no report for this session yet")

    pdf_path = os.path.join(OUTPUT_DIR, f"{session_id}_report.pdf")
    if not os.path.exists(pdf_path):
        turns = db.get_turns(session_id)
        pipeline.generate_session_pdf(session, current_user["email"], turns, report, pdf_path)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"setu_session_{session_id[:8]}.pdf",
    )


# ------------------------------------------------------------------
# Static frontend (mounted last so /api/* takes priority)
# ------------------------------------------------------------------

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")