"""
Emotion detection using SamLowe/roberta-base-go_emotions (28 emotions).

Multi-label classifier — each emotion has an independent probability.
Returns top emotions above a confidence threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]


# The 28 GoEmotions labels for reference:
GO_EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# Map each emotion to a broad category for UI colouring
EMOTION_CATEGORY: dict[str, str] = {
    "admiration": "positive",
    "amusement": "positive",
    "approval": "positive",
    "caring": "positive",
    "curiosity": "neutral",
    "desire": "neutral",
    "excitement": "positive",
    "gratitude": "positive",
    "joy": "positive",
    "love": "positive",
    "optimism": "positive",
    "pride": "positive",
    "relief": "positive",
    "surprise": "neutral",
    "realization": "neutral",
    "neutral": "neutral",
    "confusion": "neutral",
    "nervousness": "negative",
    "anger": "negative",
    "annoyance": "negative",
    "disappointment": "negative",
    "disapproval": "negative",
    "disgust": "negative",
    "embarrassment": "negative",
    "fear": "negative",
    "grief": "negative",
    "remorse": "negative",
    "sadness": "negative",
}


@dataclass
class EmotionResult:
    """Detected emotions for a single utterance."""

    emotions: list[dict[str, float]]  # [{"label": "joy", "score": 0.92}, ...]
    dominant: str                      # top-1 emotion label
    dominant_score: float              # top-1 confidence
    category: str                      # "positive" / "negative" / "neutral"


# ======================================================================
# Model — loaded once, shared across all sessions
# ======================================================================
class EmotionModel:
    """
    Wrapper around ``SamLowe/roberta-base-go_emotions`` multi-label
    classifier.

    Load once at server startup.  Thread-safe for inference (single
    forward pass per call, no mutable state).
    """

    DEFAULT_THRESHOLD = 0.15  # minimum score to include an emotion

    def __init__(self, device: int = -1, threshold: float | None = None):
        self.threshold = threshold or self.DEFAULT_THRESHOLD

        print("[Emotion] Loading SamLowe/roberta-base-go_emotions model...")
        self.classifier = hf_pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,  # return all 28 labels
            device=device,
        )
        print("[Emotion] Model loaded ✓")

    def predict(self, text: str) -> EmotionResult:
        """
        Classify text and return emotions above the threshold.

        Args:
            text: Input utterance (truncated to 512 chars).

        Returns:
            EmotionResult with filtered emotions sorted by score.
        """
        raw: list[dict] = self.classifier(text[:512])[0]  # type: ignore[index]

        # Filter to emotions above threshold, sort descending
        filtered = sorted(
            [e for e in raw if e["score"] >= self.threshold],
            key=lambda e: e["score"],
            reverse=True,
        )

        # Always include at least one (the top prediction)
        if not filtered and raw:
            top = max(raw, key=lambda e: e["score"])
            filtered = [top]

        # Round scores for cleanliness
        emotions = [
            {"label": e["label"], "score": round(e["score"], 4)}
            for e in filtered
        ]

        dominant = emotions[0]["label"] if emotions else "neutral"
        dominant_score = emotions[0]["score"] if emotions else 0.0
        category = EMOTION_CATEGORY.get(dominant, "neutral")

        return EmotionResult(
            emotions=emotions,
            dominant=dominant,
            dominant_score=dominant_score,
            category=category,
        )
