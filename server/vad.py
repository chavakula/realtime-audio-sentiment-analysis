"""
Voice Activity Detection using Silero VAD.

Buffers incoming audio and emits complete utterances
when a speaker pauses (silence > threshold).
"""

import numpy as np
import torch


class VoiceActivityDetector:
    """
    Streaming VAD using Silero VAD model.

    Accepts raw float32 PCM audio at 16kHz, detects speech segments,
    and yields complete utterances when the speaker pauses.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_frames: int = 10,   # ~320ms silence to end utterance
        min_utterance_samples: int = 4800,  # minimum 300ms utterance
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.min_silence_frames = min_silence_frames
        self.min_utterance_samples = min_utterance_samples
        self.sample_rate = sample_rate

        # Load Silero VAD
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()

        # State
        self._buffer = np.array([], dtype=np.float32)
        self._speech_buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._silence_count = 0

    def reset(self) -> None:
        """Reset internal state for a new call."""
        self._buffer = np.array([], dtype=np.float32)
        self._speech_buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._silence_count = 0
        self.model.reset_states()

    def process_chunk(self, audio_chunk: np.ndarray) -> list[np.ndarray]:
        """
        Feed an audio chunk and return any completed utterances.

        Args:
            audio_chunk: float32 numpy array, 16kHz mono audio.

        Returns:
            List of numpy arrays, each a complete utterance.
        """
        completed: list[np.ndarray] = []

        self._buffer = np.concatenate([self._buffer, audio_chunk])

        # Silero VAD expects 512-sample frames at 16kHz (~32ms per frame)
        frame_size = 512
        while len(self._buffer) >= frame_size:
            frame = self._buffer[:frame_size]
            self._buffer = self._buffer[frame_size:]

            # Run VAD inference
            tensor = torch.from_numpy(frame).float()
            speech_prob = self.model(tensor, self.sample_rate).item()

            if speech_prob >= self.threshold:
                # Speech detected
                self._is_speaking = True
                self._silence_count = 0
                self._speech_buffer = np.concatenate([self._speech_buffer, frame])
            else:
                if self._is_speaking:
                    # Silence while we were speaking
                    self._silence_count += 1
                    self._speech_buffer = np.concatenate([self._speech_buffer, frame])

                    if self._silence_count >= self.min_silence_frames:
                        # Utterance complete
                        if len(self._speech_buffer) >= self.min_utterance_samples:
                            completed.append(self._speech_buffer.copy())
                        self._speech_buffer = np.array([], dtype=np.float32)
                        self._is_speaking = False
                        self._silence_count = 0

        return completed

    def flush(self) -> list[np.ndarray]:
        """Flush any remaining audio (buffer + speech) at end of stream."""
        # First, process any leftover samples still sitting in the input buffer
        # (frames smaller than 512 that weren't processed yet).
        if len(self._buffer) > 0:
            # Pad to a full frame so Silero can evaluate it
            padded = np.zeros(512, dtype=np.float32)
            padded[: len(self._buffer)] = self._buffer
            self._buffer = np.array([], dtype=np.float32)

            tensor = torch.from_numpy(padded).float()
            speech_prob = self.model(tensor, self.sample_rate).item()

            if speech_prob >= self.threshold or self._is_speaking:
                self._speech_buffer = np.concatenate([self._speech_buffer, padded])

        result: list[np.ndarray] = []
        if len(self._speech_buffer) >= self.min_utterance_samples:
            result.append(self._speech_buffer.copy())
        self._speech_buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._silence_count = 0
        return result
