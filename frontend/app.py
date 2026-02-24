"""
Streamlit dashboard for real-time call sentiment analysis.

Features:
- Drag-and-drop audio file upload
- Speed control (0.5x – 4x)
- Live transcript with colour-coded sentiment
- Real-time stats cards
- Sentiment timeline chart
- Call summary at the end
"""

from __future__ import annotations

import base64
import json
import threading
import time
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pydub import AudioSegment

try:
    import websocket  # websocket-client (sync)
except ImportError:
    raise ImportError("pip install websocket-client")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SERVER_WS_URL = "ws://localhost:8000/ws/stream"
CHUNK_SAMPLES = 4096  # ~256 ms at 16 kHz

# Emotion → broad category mapping (mirrors server/emotion.py)
EMOTION_CATEGORY: dict[str, str] = {
    "admiration": "positive", "amusement": "positive", "approval": "positive",
    "caring": "positive", "excitement": "positive", "gratitude": "positive",
    "joy": "positive", "love": "positive", "optimism": "positive",
    "pride": "positive", "relief": "positive",
    "curiosity": "neutral", "desire": "neutral", "surprise": "neutral",
    "realization": "neutral", "neutral": "neutral", "confusion": "neutral",
    "nervousness": "negative", "anger": "negative", "annoyance": "negative",
    "disappointment": "negative", "disapproval": "negative", "disgust": "negative",
    "embarrassment": "negative", "fear": "negative", "grief": "negative",
    "remorse": "negative", "sadness": "negative",
}

st.set_page_config(
    page_title="Call Sentiment Analysis",
    page_icon="📞",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Thread-safe shared state (plain dict — no ScriptRunContext needed)
# ---------------------------------------------------------------------------
class SharedState:
    """
    A plain-Python container that the background thread writes to
    and the Streamlit main loop reads from.  Not tied to
    ScriptRunContext so it works from any thread.
    """

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []
        self.is_streaming: bool = False
        self.stream_done: bool = False
        self.error: str | None = None
        self.progress: float = 0.0

    def reset(self) -> None:
        self.results = []
        self.is_streaming = False
        self.stream_done = False
        self.error = None
        self.progress = 0.0


def _get_shared() -> SharedState:
    """Return (or create) the SharedState stored in session_state."""
    if "shared" not in st.session_state:
        st.session_state["shared"] = SharedState()
    return st.session_state["shared"]


shared = _get_shared()


# ---------------------------------------------------------------------------
# Audio streamer (runs in background thread)
# ---------------------------------------------------------------------------
def _stream_audio(
    state: SharedState,
    audio_bytes: bytes,
    file_format: str,
    speed: float,
    ws_url: str,
) -> None:
    """
    Decode the uploaded audio, convert to 16 kHz mono int16 PCM,
    and stream it over a WebSocket at the selected speed.
    Results are appended to state.results (plain list).
    """
    try:
        state.reset()
        state.is_streaming = True

        # Decode audio
        audio = AudioSegment.from_file(BytesIO(audio_bytes), format=file_format)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

        raw = np.frombuffer(audio.raw_data, dtype=np.int16)
        total_samples = len(raw)

        # Connect WebSocket (sync)
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        ws.settimeout(0.05)  # non-blocking recv

        chunk_duration_s = CHUNK_SAMPLES / 16000
        sleep_time = chunk_duration_s / speed

        sent_samples = 0
        for i in range(0, total_samples, CHUNK_SAMPLES):
            if not state.is_streaming:
                break  # user stopped

            chunk = raw[i : i + CHUNK_SAMPLES]
            ws.send_binary(chunk.tobytes())
            sent_samples += len(chunk)
            state.progress = min(1.0, sent_samples / total_samples)

            # Non-blocking: read any available results
            _drain_ws(ws, state)

            time.sleep(sleep_time)

        # Signal end-of-stream so the server flushes remaining audio
        try:
            ws.send("END")
        except Exception:
            pass

        # Wait for final flushed results — break early once no new data arrives
        ws.settimeout(0.3)
        idle_rounds = 0
        prev_count = len(state.results)
        for _ in range(50):  # max ~10s safety cap
            _drain_ws(ws, state)
            cur_count = len(state.results)
            if cur_count > prev_count:
                idle_rounds = 0
                prev_count = cur_count
            else:
                idle_rounds += 1
            if idle_rounds >= 5:  # ~1.5s with no new results → done
                break
            time.sleep(0.3)

        ws.close()

    except Exception as exc:
        state.error = str(exc)
    finally:
        state.is_streaming = False
        state.stream_done = True
        state.progress = 1.0


def _drain_ws(ws, state: SharedState) -> None:
    """Read all available messages from the WebSocket."""
    while True:
        try:
            msg = ws.recv()
            if msg:
                data = json.loads(msg)
                state.results.append(data)
        except websocket.WebSocketTimeoutException:
            break
        except Exception:
            break


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
st.title("📞 Real-Time Call Sentiment Analysis")
st.caption("Drop an audio file, hit **Start**, and watch the live transcript + sentiment.")

# -- Sidebar controls --
with st.sidebar:
    st.header("⚙️ Controls")

    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg", "webm"],
    )
    
    # Detect new file upload and cache it for audio player
    if uploaded_file is not None:
        current_filename = uploaded_file.name
        if ("audio_filename" not in st.session_state or 
            st.session_state["audio_filename"] != current_filename):
            # New file uploaded - cache it
            audio_bytes_preview = uploaded_file.read()
            uploaded_file.seek(0)  # Reset for later use
            st.session_state["audio_data"] = audio_bytes_preview
            st.session_state["audio_filename"] = current_filename
            name = uploaded_file.name.lower()
            fmt = name.rsplit(".", 1)[-1] if "." in name else "wav"
            mime_map = {
                "wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4",
                "flac": "audio/flac", "ogg": "audio/ogg", "webm": "audio/webm"
            }
            st.session_state["audio_mime"] = mime_map.get(fmt, "audio/wav")
    elif "audio_data" in st.session_state:
        # File removed - clear cache
        st.session_state.pop("audio_data", None)
        st.session_state.pop("audio_filename", None)
        st.session_state.pop("audio_mime", None)

    speed = st.select_slider(
        "Playback speed",
        options=[0.5, 1.0, 1.5, 2.0, 4.0],
        value=1.0,
        format_func=lambda x: f"{x}x",
    )
    
    # Update playback speed in session state
    if "playback_speed" not in st.session_state or st.session_state["playback_speed"] != speed:
        st.session_state["playback_speed"] = speed

    col1, col2 = st.columns(2)
    start_btn = col1.button(
        "▶  Start",
        disabled=uploaded_file is None or shared.is_streaming,
        use_container_width=True,
    )
    stop_btn = col2.button(
        "⏹ Stop",
        disabled=not shared.is_streaming,
        use_container_width=True,
    )

    st.divider()
    st.markdown(
        f"**Server:** `{SERVER_WS_URL}`\n\n"
        f"**Status:** {'🟢 Streaming...' if shared.is_streaming else '⚪ Idle'}"
    )


# -- Start / Stop logic --
if start_btn and uploaded_file is not None:
    # Determine format from extension
    name = uploaded_file.name.lower()
    fmt = name.rsplit(".", 1)[-1] if "." in name else "wav"
    fmt_map = {"m4a": "m4a", "mp3": "mp3", "flac": "flac", "ogg": "ogg", "webm": "webm"}
    file_format = fmt_map.get(fmt, "wav")

    audio_bytes = uploaded_file.read()
    
    # Update playback speed for audio player
    st.session_state["playback_speed"] = speed

    thread = threading.Thread(
        target=_stream_audio,
        args=(shared, audio_bytes, file_format, speed, SERVER_WS_URL),
        daemon=True,
    )
    thread.start()
    # Brief pause so the thread sets is_streaming before we rerun
    time.sleep(0.2)
    st.rerun()

if stop_btn:
    shared.is_streaming = False
    st.rerun()


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------
if shared.is_streaming:
    st.progress(shared.progress, text="Streaming audio...")


# ---------------------------------------------------------------------------
# Audio Player
# ---------------------------------------------------------------------------
if "audio_data" in st.session_state and st.session_state["audio_data"] is not None:
    audio_b64 = base64.b64encode(st.session_state["audio_data"]).decode()
    mime_type = st.session_state.get("audio_mime", "audio/wav")
    playback_speed = st.session_state.get("playback_speed", 1.0)
    auto_play = "true" if shared.is_streaming else "false"
    is_playing = "true" if shared.is_streaming else "false"

    audio_html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        .audio-container {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: linear-gradient(135deg, #1e1e2e 0%, #2d2b55 100%);
            border-radius: 16px;
            padding: 20px 24px;
            margin: 16px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
            position: relative;
            overflow: hidden;
        }}
        .audio-container::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.15) 0%, transparent 60%);
            pointer-events: none;
        }}
        .audio-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }}
        .audio-title {{
            color: #e2e8f0;
            font-size: 15px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .audio-badge {{
            font-size: 11px;
            font-weight: 500;
            padding: 3px 10px;
            border-radius: 20px;
            letter-spacing: 0.5px;
        }}
        .badge-streaming {{
            background: rgba(34, 197, 94, 0.2);
            color: #4ade80;
            animation: pulse 1.5s ease-in-out infinite;
        }}
        .badge-ready {{
            background: rgba(148, 163, 184, 0.2);
            color: #94a3b8;
        }}
        .speed-tag {{
            font-size: 11px;
            font-weight: 500;
            padding: 3px 8px;
            border-radius: 6px;
            background: rgba(99, 102, 241, 0.25);
            color: #a5b4fc;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
        .audio-container audio {{
            width: 100%;
            height: 44px;
            border-radius: 8px;
            outline: none;
        }}
        /* Webkit (Chrome/Edge/Safari) audio styling */
        .audio-container audio::-webkit-media-controls-panel {{
            background: rgba(255, 255, 255, 0.08);
            border-radius: 8px;
        }}
    </style>
    <div class="audio-container">
        <div class="audio-header">
            <span class="audio-title">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                    <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
                    <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
                </svg>
                Audio Playback
                <span class="speed-tag">{playback_speed}x</span>
            </span>
            <span class="audio-badge {'badge-streaming' if {is_playing} else 'badge-ready'}">
                {'● LIVE' if {is_playing} else '● READY'}
            </span>
        </div>
        <audio id="audioPlayer" controls>
            <source src="data:{mime_type};base64,{audio_b64}" type="{mime_type}">
        </audio>
    </div>
    <script>
        const audio = document.getElementById('audioPlayer');
        if (audio) {{
            audio.playbackRate = {playback_speed};
            if ({auto_play} && audio.paused) {{
                audio.play().catch(e => console.log('Auto-play blocked:', e));
            }}
        }}
    </script>
    """
    st.components.v1.html(audio_html, height=130)


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------
if shared.error:
    st.error(f"Error: {shared.error}")


# ---------------------------------------------------------------------------
# Live stats
# ---------------------------------------------------------------------------
results: list[dict] = list(shared.results)  # snapshot

if results:
    pos = sum(1 for r in results if r.get("sentiment") == "positive")
    neg = sum(1 for r in results if r.get("sentiment") == "negative")
    neu = sum(1 for r in results if r.get("sentiment") == "neutral")
    esc = sum(1 for r in results if r.get("escalation"))
    total = len(results)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Utterances", total)
    c2.metric("😊 Positive", pos)
    c3.metric("😐 Neutral", neu)
    c4.metric("😠 Negative", neg)
    c5.metric("⚠️ Escalations", esc)
    avg_latency = round(
        sum(r.get("latency", {}).get("total_ms", 0) for r in results) / total, 0
    )
    c6.metric("Avg Latency", f"{avg_latency}ms")

    # -- Top Emotions bar --
    all_emotions: list[str] = []
    for r in results:
        for e in r.get("emotions", []):
            all_emotions.append(e["label"])

    if all_emotions:
        from collections import Counter as _Counter
        emo_counts = _Counter(all_emotions).most_common(8)
        emo_labels, emo_vals = zip(*emo_counts)
        c_emo1, c_emo2 = st.columns([3, 2])
        with c_emo1:
            st.caption("**Top Emotions Detected**")
            emo_bar_html = ""
            _emo_colors = {
                "positive": "#4caf50", "negative": "#f44336", "neutral": "#ff9800",
            }
            for lbl, cnt in emo_counts:
                cat = EMOTION_CATEGORY.get(lbl, "neutral")
                clr = _emo_colors.get(cat, "#9e9e9e")
                emo_bar_html += (
                    f'<span style="display:inline-block;margin:2px 4px;padding:3px 10px;'
                    f'border-radius:12px;background:{clr};color:#fff;font-size:13px;">'
                    f'{lbl} ({cnt})</span>'
                )
            st.markdown(emo_bar_html, unsafe_allow_html=True)

    st.divider()

    # -- Two-column layout: transcript + chart --
    left, right = st.columns([3, 2])

    # -- Live Transcript --
    with left:
        st.subheader("📝 Live Transcript")
        for r in reversed(results):
            sentiment = r.get("sentiment", "unknown")
            icon = {"positive": "😊", "negative": "😠", "neutral": "😐"}.get(sentiment, "❓")
            color = {"positive": "#e8f5e9", "negative": "#ffebee", "neutral": "#fff8e1"}.get(sentiment, "#f5f5f5")
            border = {"positive": "#4caf50", "negative": "#f44336", "neutral": "#ff9800"}.get(sentiment, "#9e9e9e")
            esc_badge = " ⚠️ **ESCALATION**" if r.get("escalation") else ""

            # Emotion badges
            emo_html = ""
            for e in r.get("emotions", []):
                emo_lbl = e["label"]
                emo_score = e["score"]
                _cat_colors = {"positive": "#4caf50", "negative": "#f44336", "neutral": "#ff9800"}
                ecat = EMOTION_CATEGORY.get(emo_lbl, "neutral")
                eclr = _cat_colors.get(ecat, "#9e9e9e")
                emo_html += (
                    f'<span style="display:inline-block;margin:1px 2px;padding:1px 6px;'
                    f'border-radius:8px;background:{eclr};color:#fff;font-size:11px;">'
                    f'{emo_lbl} {emo_score*100:.0f}%</span>'
                )

            emo_ms = r.get('latency', {}).get('emotion_ms', 0)

            st.markdown(
                f"""<div style="padding:10px;margin:5px 0;border-radius:6px;
                border-left:4px solid {border};background:{color}">
                <b>{icon} [{r.get('call_time_seconds',0):.1f}s] {r.get('speaker','?')}:</b>
                {r.get('text','')}<br>
                {emo_html}
                <br><small style="color:#666">{sentiment} ({r.get('score',0)*100:.0f}%)
                &nbsp;|&nbsp; STT {r.get('latency',{}).get('stt_ms',0)}ms
                &nbsp;|&nbsp; Sent {r.get('latency',{}).get('sentiment_ms',0)}ms
                &nbsp;|&nbsp; Emo {emo_ms}ms
                {esc_badge}</small>
                </div>""",
                unsafe_allow_html=True,
            )

    # -- Sentiment Timeline --
    with right:
        st.subheader("📈 Sentiment Timeline")

        df = pd.DataFrame(results)
        if not df.empty and "call_time_seconds" in df.columns:
            def _sign_score(row):
                if row["sentiment"] == "positive":
                    return row["score"]
                elif row["sentiment"] == "negative":
                    return -row["score"]
                else:  # neutral
                    return 0.0
            df["score_signed"] = df.apply(_sign_score, axis=1)
            fig = px.scatter(
                df,
                x="call_time_seconds",
                y="score_signed",
                color="speaker",
                hover_data=["text", "sentiment"],
                labels={
                    "call_time_seconds": "Call Time (s)",
                    "score_signed": "Sentiment Score",
                },
                color_discrete_sequence=["#2196f3", "#ff9800"],
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # -- Per-Speaker Summary --
        st.subheader("📊 Speaker Summary")
        summary = results[-1].get("summary", {}) if results else {}
        per_speaker = summary.get("per_speaker", {})

        for speaker, data in per_speaker.items():
            pos_pct = data.get("positive_pct", 0)
            neg_pct = data.get("negative_pct", 0)
            neu_pct = data.get("neutral_pct", 0)
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=["Positive", "Neutral", "Negative"],
                        values=[max(pos_pct, 0.1), max(neu_pct, 0.1), max(neg_pct, 0.1)],
                        marker_colors=["#4caf50", "#ff9800", "#f44336"],
                        hole=0.4,
                    )
                ]
            )
            fig_pie.update_layout(
                title=f"{speaker} ({data.get('total',0)} utterances)",
                height=250,
                margin=dict(l=0, r=0, t=40, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # -- Emotion Distribution --
        _all_emo_labels: list[str] = []
        for r in results:
            dom = r.get("dominant_emotion")
            if dom:
                _all_emo_labels.append(dom)
        if _all_emo_labels:
            from collections import Counter as _Counter2
            st.subheader("🎭 Emotion Distribution")
            _emo_c = _Counter2(_all_emo_labels).most_common(12)
            _e_labels, _e_counts = zip(*_emo_c)
            _e_colors = [
                {"positive": "#4caf50", "negative": "#f44336", "neutral": "#ff9800"}.get(
                    EMOTION_CATEGORY.get(l, "neutral"), "#9e9e9e"
                )
                for l in _e_labels
            ]
            fig_emo = go.Figure(
                data=[
                    go.Bar(
                        x=list(_e_labels),
                        y=list(_e_counts),
                        marker_color=_e_colors,
                    )
                ]
            )
            fig_emo.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Emotion",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_emo, use_container_width=True)
else:
    st.info("No results yet. Upload an audio file and click **Start** to begin analysis.")


# ---------------------------------------------------------------------------
# Auto-refresh while streaming / detect completion
# ---------------------------------------------------------------------------
if shared.is_streaming:
    time.sleep(1)
    st.rerun()
elif shared.stream_done:
    # Stream just finished — rerun once to update UI to idle state
    shared.stream_done = False
    st.rerun()
