"""
FastAPI WebSocket server for real-time call sentiment analysis.

Endpoints:
    GET  /health              Health check
    WS   /ws/stream           Real-time audio stream → sentiment results
    WS   /ws/dashboard        Dashboard clients subscribe to live results
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from server.transcriber import Transcriber
from server.sentiment import SentimentModel
from server.emotion import EmotionModel
from server.pipeline import RealtimePipeline

# ---------------------------------------------------------------------------
# Globals (initialised on startup)
# ---------------------------------------------------------------------------
transcriber: Transcriber | None = None
sentiment_model: SentimentModel | None = None
emotion_model: EmotionModel | None = None

# Dashboard clients that want to receive live updates
dashboard_clients: list[WebSocket] = []


# ---------------------------------------------------------------------------
# Lifespan — load models once
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcriber, sentiment_model, emotion_model
    print("=" * 60)
    print("  Loading models (this takes ~30 s on first run) ...")
    print("=" * 60)

    model_size = os.getenv("WHISPER_MODEL", "base.en")
    device = os.getenv("DEVICE", "cpu")
    compute_type = os.getenv("COMPUTE_TYPE", "int8")

    transcriber = Transcriber(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )
    sentiment_model = SentimentModel(device=-1)
    emotion_model = EmotionModel(device=-1)

    print("=" * 60)
    print("  All models loaded — server ready!")
    print("=" * 60)
    yield
    print("[Server] Shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Real-Time Call Sentiment Analysis",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return JSONResponse(
        {
            "status": "ok",
            "models_loaded": transcriber is not None
            and sentiment_model is not None
            and emotion_model is not None,
        }
    )


# ---------------------------------------------------------------------------
# WebSocket: audio stream
# ---------------------------------------------------------------------------
@app.websocket("/ws/stream")
async def audio_stream(websocket: WebSocket):
    """
    Receive real-time PCM audio (int16, 16 kHz, mono) and return
    JSON sentiment results for each detected utterance.
    """
    await websocket.accept()
    print("[WS] Audio client connected")

    assert transcriber is not None and sentiment_model is not None

    # Each connection gets its own pipeline (isolated VAD + state)
    # but shares the heavy models (Whisper + DistilBERT + GoEmotions)
    pipeline = RealtimePipeline(
        transcriber=transcriber,
        sentiment_model=sentiment_model,
        emotion_model=emotion_model,
    )

    try:
        while True:
            message = await websocket.receive()

            # Check for end-of-stream text signal
            if "text" in message:
                txt = message["text"]
                if txt.strip().upper() == "END":
                    print("[WS] End-of-stream signal received — flushing pipeline")
                    final_results = await asyncio.to_thread(pipeline.flush)
                    for result in final_results:
                        await websocket.send_json(result)
                        await _broadcast_to_dashboards(result)
                    print(f"[WS] Flushed {len(final_results)} final utterance(s)")
                    continue

            # Binary audio data
            if "bytes" in message:
                data = message["bytes"]
                results = await asyncio.to_thread(pipeline.process_audio_chunk, data)

                for result in results:
                    await websocket.send_json(result)
                    await _broadcast_to_dashboards(result)

    except WebSocketDisconnect:
        print("[WS] Audio client disconnected")
        # Attempt flush in case client disconnected without END signal
        try:
            final_results = await asyncio.to_thread(pipeline.flush)
            for result in final_results:
                await websocket.send_json(result)
                await _broadcast_to_dashboards(result)
        except Exception:
            pass  # client already gone
    except Exception as exc:
        print(f"[WS] Error: {exc}")


# ---------------------------------------------------------------------------
# WebSocket: dashboard (read-only subscriber)
# ---------------------------------------------------------------------------
@app.websocket("/ws/dashboard")
async def dashboard_stream(websocket: WebSocket):
    """Dashboard clients subscribe here for live updates."""
    await websocket.accept()
    dashboard_clients.append(websocket)
    print(f"[WS] Dashboard client connected ({len(dashboard_clients)} total)")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        dashboard_clients.remove(websocket)
        print(f"[WS] Dashboard client disconnected ({len(dashboard_clients)} total)")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
async def _broadcast_to_dashboards(message: dict[str, Any]) -> None:
    """Push a result dict to all connected dashboard clients."""
    dead: list[WebSocket] = []
    for client in dashboard_clients:
        try:
            await client.send_json(message)
        except Exception:
            dead.append(client)
    for client in dead:
        dashboard_clients.remove(client)
