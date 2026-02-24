"""
Sentiment analysis using cardiffnlp/twitter-roberta-base-sentiment-latest.

3-class model: positive / negative / neutral.
Separates the **model** (loaded once, shared across sessions) from
the **session state** (per-call speaker history, escalation tracking).
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass

from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]


@dataclass
class SentimentResult:
    """Single utterance sentiment result."""

    speaker: str
    text: str
    label: str      # "positive", "negative", or "neutral"
    score: float    # confidence 0-1
    escalation: bool  # True if speaker has 3+ consecutive negatives


# ======================================================================
# Model — loaded once, shared across all WebSocket sessions
# ======================================================================
class SentimentModel:
    """
    Thin wrapper around the HuggingFace sentiment pipeline.

    Load once at server startup and pass to every ``SentimentSession``.
    Thread-safe for inference.
    """

    # cardiffnlp label mapping
    _LABEL_MAP = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        # Fallbacks for numeric label keys some versions emit
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
    }

    def __init__(self, device: int = -1):
        print("[Sentiment] Loading cardiffnlp/twitter-roberta-base-sentiment-latest...")
        self.classifier = hf_pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device,
        )
        print("[Sentiment] Model loaded ✓")

    def predict(self, text: str) -> dict:
        """Return raw model output with normalized label."""
        raw = self.classifier(text[:512])[0]
        raw["label"] = self._LABEL_MAP.get(raw["label"], raw["label"].lower())
        return raw


# ======================================================================
# Session — one per WebSocket connection (per call)
# ======================================================================
class SentimentSession:
    """
    Per-call sentiment state.

    Tracks speaker history and escalation.  Uses a **shared**
    ``SentimentModel`` for inference so the heavy weights are
    loaded only once.
    """

    ESCALATION_WINDOW = 3  # consecutive negatives to flag

    def __init__(self, model: SentimentModel):
        self._model = model
        self._history: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=20))

    def reset(self) -> None:
        """Clear all speaker history (new call)."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str, speaker: str = "Unknown") -> SentimentResult:
        """Classify sentiment for a single utterance."""
        result = self._model.predict(text)

        label: str = result["label"].lower()
        score: float = round(result["score"], 4)

        self._history[speaker].append(label)
        escalation = self._check_escalation(speaker)

        return SentimentResult(
            speaker=speaker,
            text=text,
            label=label,
            score=score,
            escalation=escalation,
        )

    def get_summary(self) -> dict:
        """
        Running summary of the call so far.

        Returns dict with ``per_speaker`` and ``overall`` keys.
        """
        per_speaker: dict[str, dict] = {}
        total_all = 0

        for speaker, history in self._history.items():
            labels = list(history)
            total = len(labels)
            total_all += total
            pos = labels.count("positive")
            neg = labels.count("negative")
            neu = labels.count("neutral")

            per_speaker[speaker] = {
                "total": total,
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "positive_pct": round(pos / total * 100, 1) if total else 0,
                "negative_pct": round(neg / total * 100, 1) if total else 0,
                "neutral_pct": round(neu / total * 100, 1) if total else 0,
                "trend": labels[-1] if labels else "unknown",
                "escalation": self._check_escalation(speaker),
            }

        all_labels = [lbl for h in self._history.values() for lbl in h]
        counts = Counter(all_labels)
        dominant = counts.most_common(1)[0][0] if counts else "unknown"

        return {
            "per_speaker": per_speaker,
            "overall": {"total": total_all, "dominant": dominant},
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_escalation(self, speaker: str) -> bool:
        history = self._history.get(speaker)
        if not history or len(history) < self.ESCALATION_WINDOW:
            return False
        recent = list(history)[-self.ESCALATION_WINDOW:]
        return all(s == "negative" for s in recent)
