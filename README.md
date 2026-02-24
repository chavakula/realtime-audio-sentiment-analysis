# Real-Time Call Sentiment Analysis PoC

Analyse the sentiment and emotions of audio calls in real time. Drop an audio
file into the Streamlit dashboard and watch a live transcript with per-utterance
sentiment colour-coding, 28-emotion detection, escalation alerts, an integrated
audio player, and per-speaker summaries.

## Architecture

```
┌──────────────────┐  audio file   ┌───────────────────┐  PCM chunks   ┌─────────────────┐
│  Streamlit UI    │──────────────▶│  WebSocket Client │──────────────▶│  FastAPI Server  │
│  (frontend/)     │               │  (in Streamlit)   │               │  /ws/stream      │
└──────────────────┘               └───────────────────┘               └────────┬─────────┘
        ▲                                                                       │
        │  JSON results (per utterance)                                         ▼
        └──────────────────────────────────────────────────────┐  ┌─────────────────────────┐
                                                               │  │  Pipeline               │
                                                               │  │  1. Silero VAD          │
                                                               │  │  2. faster-whisper STT  │
                                                               │  │  3. RoBERTa Sentiment   │
                                                               │  │  4. GoEmotions (28 emo) │
                                                               │  └─────────────────────────┘
                                                               │
                                                          WebSocket
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| VAD | Silero VAD (local, via torch.hub) |
| Speech-to-Text | faster-whisper `base.en` (CTranslate2, CPU) |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` — 3-class: positive / negative / neutral |
| Emotion Detection | `SamLowe/roberta-base-go_emotions` — 28 emotions, multi-label |
| Backend | FastAPI + WebSocket |
| Frontend | Streamlit + Plotly |
| Audio processing | pydub + numpy |
| Containerisation | Docker + Docker Compose |

**Zero cloud APIs. Everything runs locally.**

## Features

- **Real-time streaming** — audio is streamed in 256 ms chunks at configurable speed (0.5x–4x)
- **3-class sentiment** — positive, negative, neutral with confidence scores
- **28 granular emotions** — admiration, anger, annoyance, joy, gratitude, etc. (colour-coded badges)
- **Escalation detection** — flags speakers with 3+ consecutive negative utterances
- **Integrated audio player** — styled HTML5 player with speed sync and live/ready indicator
- **Live dashboard** — transcript, stats cards, sentiment timeline, emotion distribution chart, per-speaker pie charts
- **End-of-stream flush** — ensures the last utterance is always processed (explicit `END` signal protocol)

## Quickstart (Local)

### Prerequisites

- Python 3.11+ (tested on 3.13)
- ffmpeg (`brew install ffmpeg`)

### 1. Install dependencies

```bash
cd sentiment-poc
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate sample audio (optional, macOS only)

```bash
python generate_samples.py
```

Creates `sample_calls/positive_call.wav`, `negative_call.wav`, `escalating_call.wav`.

### 3. Start the FastAPI server

```bash
cd sentiment-poc
source .venv/bin/activate
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Models download on first run (~1.5 GB total for Whisper + RoBERTa Sentiment + GoEmotions). Takes ~30–60 seconds.

### 4. Start the Streamlit dashboard

In a second terminal:

```bash
cd sentiment-poc
source .venv/bin/activate
streamlit run frontend/app.py --server.port 8501
```

### 5. Use it

1. Open http://localhost:8501
2. Upload an audio file (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`)
3. Pick a speed (0.5x – 4x; 1x = real-time)
4. Click **Start**
5. Watch the live transcript, sentiment scores, emotion badges, timeline chart, and listen via the audio player

### 6. Shutdown

```bash
lsof -ti :8000 :8501 | xargs kill -9
```

Or press `Ctrl+C` in each terminal.

## Quickstart (Docker)

```bash
cd sentiment-poc
docker compose up --build
```

Then open http://localhost:8501.

## Project Structure

```
sentiment-poc/
├── server/
│   ├── __init__.py
│   ├── main.py           # FastAPI app + WebSocket endpoints
│   ├── vad.py            # Silero VAD wrapper
│   ├── transcriber.py    # faster-whisper STT wrapper
│   ├── sentiment.py      # RoBERTa 3-class sentiment (positive/negative/neutral)
│   ├── emotion.py        # GoEmotions 28-emotion detector
│   └── pipeline.py       # Orchestrator: VAD → STT → Sentiment → Emotion
├── frontend/
│   └── app.py            # Streamlit dashboard + audio player
├── sample_calls/         # Generated test audio files
├── generate_samples.py   # macOS TTS sample generator
├── test_e2e.py           # End-to-end WebSocket test
├── test_models.py        # Model accuracy test (single model)
├── test_compare_models.py # Side-by-side sentiment model comparison
├── test_updated_models.py # Verify updated models
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
└── README.md
```

## How It Works

1. **Audio Upload** — Streamlit accepts a file upload (drag-and-drop).
2. **Simulated Stream** — The file is decoded (pydub → 16 kHz mono int16 PCM),
   then sent in 256 ms chunks over a WebSocket to the FastAPI server at the
   selected playback speed.
3. **VAD** — Silero VAD detects when someone is speaking. When speech ends
   (>320 ms of silence), the complete utterance is emitted.
4. **Transcription** — faster-whisper transcribes the utterance to text
   (~200–500 ms on CPU).
5. **Sentiment** — `cardiffnlp/twitter-roberta-base-sentiment-latest` classifies
   the text as positive / negative / neutral (~15 ms on CPU).
6. **Emotion** — `SamLowe/roberta-base-go_emotions` detects up to 28 granular
   emotions with multi-label classification (~15 ms on CPU). Emotions above a
   0.15 confidence threshold are returned.
7. **Result** — JSON with speaker, text, sentiment, score, emotions, escalation
   flag, latency breakdown, and running summary is sent back via WebSocket.
8. **End-of-Stream** — Client sends an `END` text message. Server flushes the
   VAD buffer (including partial frames) and processes any remaining speech
   before closing.
9. **Dashboard** — Live transcript with colour-coded sentiment, emotion badges,
   stats cards, sentiment timeline, emotion distribution bar chart, and
   per-speaker pie charts. Audio player syncs with streaming speed.

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `WHISPER_MODEL` | `base.en` | Whisper model size (`tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v3`) |
| `DEVICE` | `cpu` | Inference device (`cpu` or `cuda`) |
| `COMPUTE_TYPE` | `int8` | CTranslate2 compute type (`int8`, `float16`, `float32`) |

## Latency (CPU, base.en)

| Stage | Typical |
|-------|---------|
| VAD | ~5 ms per frame |
| Transcription | 200–500 ms per utterance |
| Sentiment | 10–15 ms per utterance |
| Emotion | 14–18 ms per utterance |
| **Total** | **~300–600 ms** per utterance |
