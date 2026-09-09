"""
main.py
-------
FastAPI entrypoint. Wires:
  - WebSocket /ws         -- the live event stream React listens to, and
                              the channel the frontend sends transcribed
                              text / control actions over.
  - POST /api/voice-command -- upload audio -> Person 1 transcribes it ->
                              task_manager runs the full pipeline. Also
                              works fine for curl/Postman testing without
                              a websocket client.
  - POST /api/reset        -- clear task state (the "New Task" button).
  - GET  /api/state        -- current state/task snapshot, for a page reload.

Run with:
    python main.py
"""

import os
import sys
import asyncio
import logging
from typing import Set, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from voice import transcribe_audio, voice_manager
from browser.browser import browser_manager
from task_manager import task_manager
import events

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await browser_manager.start()
    except Exception as e:
        logger.warning(f"Browser startup in lifespan failed ({e}); will lazy-load on first search.")
    yield
    await browser_manager.shutdown()


app = FastAPI(title="Interruptible AI Voice Browser Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hackathon simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/health")
async def health_check():
    """Healthcheck endpoint for Render / cloud deployment monitors."""
    return {"status": "ok", "service": "Interruptible AI Voice Browser Agent"}



# ------------------------------------------------------------------------
# WebSocket broadcast hub
# ------------------------------------------------------------------------
class Hub:
    def __init__(self):
        self.connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.add(ws)
        logger.info(f"Client connected. Total: {len(self.connections)}")

    def disconnect(self, ws: WebSocket):
        self.connections.discard(ws)
        logger.info(f"Client disconnected. Remaining: {len(self.connections)}")

    async def broadcast(self, event: dict):
        dead = set()
        for ws in list(self.connections):
            try:
                await ws.send_json(event)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.connections.discard(ws)


hub = Hub()


# ------------------------------------------------------------------------
# REST endpoints
# ------------------------------------------------------------------------
@app.post("/api/voice-command")
async def voice_command(file: UploadFile = File(...)):
    """
    Full pipeline entry point via file upload (used by the frontend's
    recorder, and testable directly with curl). Transcribes the audio
    (Person 1), then runs it through the central task manager end to end.

    Returns:
        transcript: the transcribed text
        language:   ISO 639-1 code detected by STT (e.g. "en", "hi")
        result:     the task manager pipeline result
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    voice_manager.start_transcribing()
    try:
        stt_result = transcribe_audio(content, mime_type=file.content_type or "audio/webm")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    text = stt_result.get("text", "")
    stt_language = stt_result.get("language", "en")

    result = await task_manager.handle_utterance(text, hub.broadcast, stt_language=stt_language)
    return {"transcript": text, "language": stt_language, "result": result}


class TextCommand(BaseModel):
    text: str


@app.post("/api/text-command")
async def text_command(req: TextCommand):
    """
    Bypass audio entirely -- handy for testing/demoing without a mic.
    Language defaults to "en" since typed text has no audio to detect from;
    the LLM parse step will detect the actual language from the text.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    result = await task_manager.handle_utterance(req.text, hub.broadcast, stt_language="und")
    return {"transcript": req.text, "result": result}


@app.post("/api/reset")
async def reset():
    task_manager.reset()
    await hub.broadcast(events.state_changed(None, "IDLE"))
    await hub.broadcast(events.metrics_updated(dict(task_manager._m)))
    return {"status": "reset"}


@app.get("/api/state")
async def get_state():
    current = task_manager.get_current_task()
    return {
        "current_task_id": task_manager.current_task_id,
        "task": current.dict() if current else None,
        "voice_state": voice_manager.state.value,
    }


# ------------------------------------------------------------------------
# WebSocket
# ------------------------------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await hub.connect(websocket)

    current = task_manager.get_current_task()
    await websocket.send_json(events.state_changed(
        task_manager.current_task_id,
        current.state.value if current else "IDLE",
    ))
    await websocket.send_json(events.metrics_updated(dict(task_manager._m)))

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "utterance":
                text = data.get("text", "")
                if text.strip():
                    await task_manager.handle_utterance(text, hub.broadcast)

            elif action == "reset":
                task_manager.reset()
                await hub.broadcast(events.state_changed(None, "IDLE"))
                await hub.broadcast(events.metrics_updated(dict(task_manager._m)))

            elif action == "start_listening":
                voice_manager.start_listening()
                await hub.broadcast(events.state_changed(task_manager.current_task_id, "LISTENING"))

            elif action == "approval":
                # User approved or rejected the agent's recommendation.
                # approved=True  → agent opens product URL
                # approved=False → agent acknowledges and waits for refinement
                approved = bool(data.get("approved", False))
                await task_manager.handle_approval(approved, hub.broadcast)

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        hub.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        hub.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    # Windows + Playwright needs ProactorEventLoop for subprocess support;
    # uvicorn 0.36+ ignores asyncio.set_event_loop_policy(), so we pass the
    # loop directly via its own --loop mechanism instead.
    loop_arg = "asyncio:ProactorEventLoop" if sys.platform == "win32" else "auto"
    uvicorn.run("main:app", host=host, port=port, reload=False, loop=loop_arg)
