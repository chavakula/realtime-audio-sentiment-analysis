"""
Speech-to-Text using faster-whisper.

Uses the CTranslate2 backend for fast CPU inference.
Model is loaded once and reused across all requests.
"""

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    """
    Real-time speech-to-text using faster-whisper (base.en model).

    Optimised for low-latency single-utterance transcription on CPU.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """
        Args:
            model_size: Whisper model variant. "base.en" is a good
                        speed/accuracy trade-off for English.
            device: "cpu" or "cuda".
            compute_type: "int8" (fastest CPU), "float16" (GPU), etc.
        """
        print(f"[Transcriber] Loading faster-whisper model '{model_size}' on {device} ({compute_type})...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        print("[Transcriber] Model loaded ✓")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a single utterance.

        Args:
            audio: float32 numpy array, 16 kHz mono.

        Returns:
            Transcribed text (stripped). Empty string if nothing detected.
        """
        segments, _info = self.model.transcribe(
            audio,
            beam_size=1,
            best_of=1,
            language="en",
            without_timestamps=True,
            vad_filter=False,  # We already ran VAD upstream
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()

    def transcribe_with_timestamps(self, audio: np.ndarray) -> list[dict]:
        """
        Transcribe with word-level timestamps.

        Returns:
            List of dicts with keys: text, start, end, words.
        """
        segments, _ = self.model.transcribe(
            audio,
            beam_size=1,
            word_timestamps=True,
            language="en",
        )

        results = []
        for seg in segments:
            results.append(
                {
                    "text": seg.text.strip(),
                    "start": seg.start,
                    "end": seg.end,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in (seg.words or [])
                    ],
                }
            )
        return results
