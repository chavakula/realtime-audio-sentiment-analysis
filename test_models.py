#!/usr/bin/env python3
"""Test sentiment + emotion model accuracy on known inputs."""

from transformers import pipeline

# Test sentences with expected labels
tests = [
    ("I'm seriously considering never buying from you again.", "negative"),
    ("I hate this product, it is terrible", "negative"),
    ("This is the worst service I have ever experienced", "negative"),
    ("I want a full refund", "negative"),
    ("Never buying from you again", "negative"),
    ("I am very happy with the service", "positive"),
    ("Thank you so much for your help", "positive"),
    ("You people are completely useless", "negative"),
    ("I'm done with this company", "negative"),
]

# --- Sentiment ---
print("=== SENTIMENT (DistilBERT SST-2) ===")
sent = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,
)
wrong = 0
for text, expected in tests:
    r = sent(text)[0]
    label = r["label"].lower()
    ok = "✓" if label == expected else "✗ WRONG"
    if label != expected:
        wrong += 1
    print(f"  {ok:10s} {label:8s} ({r['score']:.2f}) | {text}")

print(f"\nSentiment accuracy: {len(tests) - wrong}/{len(tests)}")

# --- Emotion ---
print("\n=== EMOTION (GoEmotions top-5) ===")
emo = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=5,
    device=-1,
)
for text, _ in tests:
    r = emo(text)[0]
    top = ", ".join(f"{e['label']}({e['score']:.2f})" for e in r[:5])
    print(f"  {top}")
    print(f"    -> {text}")
