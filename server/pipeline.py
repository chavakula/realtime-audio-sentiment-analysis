"""
Real-time processing pipeline: VAD → STT → Sentiment.

Orchestrates the three components and produces JSON-serialisable
results for each detected utterance.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import numpy as np

from server.vad import VoiceActivityDetector
from server.transcriber import Transcriber
from server.sentiment import SentimentModel, SentimentSession, SentimentResult
from server.emotion import EmotionModel, EmotionResult


class RealtimePipeline:
    """
    Per-session pipeline that processes raw audio and emits results.

    Each WebSocket connection should instantiate its own ``RealtimePipeline``
    so that VAD state, speaker counters, and sentiment history are isolated.

    The heavy models (Whisper, DistilBERT) are **shared** across sessions
    via dependency injection — they are thread-safe for inference.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        sentiment_model: SentimentModel,
        emotion_model: EmotionModel | None = None,
        vad_threshold: float = 0.5,
    ):
        self.transcriber = transcriber
        self.sentiment = SentimentSession(sentiment_model)
        self.emotion_model = emotion_model

        # Per-session VAD instance (maintains its own buffers)
        self.vad = VoiceActivityDetector(threshold=vad_threshold)

        # Per-session state
        self._call_start = time.time()
        self._utterance_count = 0
        self._results: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Reset for a new call."""
        self.vad.reset()
        self.sentiment.reset()
        self._call_start = time.time()
        self._utterance_count = 0
        self._results.clear()

    # ------------------------------------------------------------------
    # Main entry point — called for each chunk from the WebSocket
    # ------------------------------------------------------------------

    def process_audio_chunk(self, pcm_int16: bytes) -> list[dict[str, Any]]:
        """
        Process a chunk of raw PCM audio.

        Args:
            pcm_int16: Raw bytes of int16 PCM audio at 16 kHz mono.

        Returns:
            List of result dicts (one per completed utterance in this chunk).
            Most chunks return an empty list; results appear when the
            speaker pauses.
        """
        # Convert int16 bytes → float32 numpy
        audio = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0

        # VAD: may return 0 or more complete utterances
        utterances = self.vad.process_chunk(audio)

        results: list[dict[str, Any]] = []
        for utterance_audio in utterances:
            result = self._process_utterance(utterance_audio)
            if result is not None:
                results.append(result)

        return results

    def flush(self) -> list[dict[str, Any]]:
        """Flush remaining audio at end of stream."""
        results: list[dict[str, Any]] = []
        for utterance_audio in self.vad.flush():
            result = self._process_utterance(utterance_audio)
            if result is not None:
                results.append(result)
        return results

    def get_all_results(self) -> list[dict[str, Any]]:
        """Return all results collected so far."""
        return list(self._results)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_utterance(self, audio: np.ndarray) -> dict[str, Any] | None:
        """Run STT + Sentiment on a single utterance."""
        t0 = time.time()

        # 1) Transcribe
        text = self.transcriber.transcribe(audio)
        t_stt = time.time()

        if not text.strip():
            return None

        # 2) Assign speaker (simple alternation for PoC)
        self._utterance_count += 1
        speaker = f"Speaker_{(self._utterance_count % 2) + 1}"

        # 3) Sentiment
        sentiment: SentimentResult = self.sentiment.analyze(text, speaker)
        t_sent = time.time()

        # 4) Emotion detection (optional)
        emotion_data: dict[str, Any] = {}
        if self.emotion_model is not None:
            emo: EmotionResult = self.emotion_model.predict(text)
            emotion_data = {
                "emotions": emo.emotions,        # [{label, score}, ...]
                "dominant_emotion": emo.dominant,
                "dominant_emotion_score": emo.dominant_score,
                "emotion_category": emo.category,
            }
        t_emo = time.time()

        # 5) Build result
        call_time = time.time() - self._call_start
        result: dict[str, Any] = {
            "type": "utterance",
            "utterance_id": self._utterance_count,
            "speaker": sentiment.speaker,
            "text": sentiment.text,
            "sentiment": sentiment.label,
            "score": sentiment.score,
            "escalation": sentiment.escalation,
            **emotion_data,
            "call_time_seconds": round(call_time, 1),
            "audio_duration_seconds": round(len(audio) / 16000, 2),
            "latency": {
                "stt_ms": round((t_stt - t0) * 1000, 1),
                "sentiment_ms": round((t_sent - t_stt) * 1000, 1),
                "emotion_ms": round((t_emo - t_sent) * 1000, 1),
                "total_ms": round((t_emo - t0) * 1000, 1),
            },
            "summary": self.sentiment.get_summary(),
        }

        self._results.append(result)

        emo_tag = f" | Emo: {emotion_data.get('dominant_emotion', 'n/a')}" if emotion_data else ""
        print(
            f"[Pipeline] #{self._utterance_count} | {speaker} | "
            f"{sentiment.label} ({sentiment.score:.2f}){emo_tag} | "
            f"STT {result['latency']['stt_ms']}ms | "
            f"Sent {result['latency']['sentiment_ms']}ms | "
            f"Emo {result['latency']['emotion_ms']}ms | "
            f"'{text[:60]}...'" if len(text) > 60 else
            f"[Pipeline] #{self._utterance_count} | {speaker} | "
            f"{sentiment.label} ({sentiment.score:.2f}){emo_tag} | "
            f"STT {result['latency']['stt_ms']}ms | "
            f"Sent {result['latency']['sentiment_ms']}ms | "
            f"Emo {result['latency']['emotion_ms']}ms | "
            f"'{text}'"
        )

        return result
