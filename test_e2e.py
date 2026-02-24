#!/usr/bin/env python3
"""Quick end-to-end test: stream sample audio via WebSocket."""

import json
import time

import numpy as np
import websocket
from pydub import AudioSegment

# Load sample audio
audio = AudioSegment.from_wav("sample_calls/negative_call.wav")
audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
raw = np.frombuffer(audio.raw_data, dtype=np.int16)

print(f"Audio: {len(audio)/1000:.1f}s, {len(raw)} samples")

# Connect and stream
ws = websocket.WebSocket()
ws.connect("ws://localhost:8000/ws/stream")
ws.settimeout(0.1)

chunk_size = 4096
results = []
for i in range(0, len(raw), chunk_size):
    chunk = raw[i : i + chunk_size]
    ws.send_binary(chunk.tobytes())
    try:
        msg = ws.recv()
        if msg:
            data = json.loads(msg)
            results.append(data)
            t = data["call_time_seconds"]
            spk = data["speaker"]
            txt = data["text"][:60]
            sent = data["sentiment"]
            score = data["score"]
            print(f"  [{t:.1f}s] {spk}: '{txt}' -> {sent} ({score:.2f})")
    except websocket.WebSocketTimeoutException:
        pass
    time.sleep(0.016)  # ~4x speed

# Wait for remaining results
for _ in range(100):
    try:
        msg = ws.recv()
        if msg:
            data = json.loads(msg)
            results.append(data)
            t = data["call_time_seconds"]
            spk = data["speaker"]
            txt = data["text"][:60]
            sent = data["sentiment"]
            score = data["score"]
            print(f"  [{t:.1f}s] {spk}: '{txt}' -> {sent} ({score:.2f})")
    except Exception:
        break
    time.sleep(0.1)

ws.close()
print(f"\nTotal results: {len(results)}")
if results:
    summary = results[-1].get("summary", {})
    print(f"Summary: {json.dumps(summary, indent=2)}")
    print("\n--- Emotion Breakdown ---")
    for r in results:
        emo = r.get("dominant_emotion", "n/a")
        emo_s = r.get("dominant_emotion_score", 0)
        cat = r.get("emotion_category", "?")
        all_emo = ", ".join(f"{e['label']}({e['score']:.2f})" for e in r.get("emotions", []))
        print(f"  [{r['call_time_seconds']:.1f}s] {r['speaker']}: {emo} ({emo_s:.2f}) [{cat}] | {all_emo}")
