#!/usr/bin/env python3
"""Verify updated sentiment + emotion models on tricky inputs."""

from server.sentiment import SentimentModel
from server.emotion import EmotionModel

tests = [
    ("I'm seriously considering never buying from you again.", "negative"),
    ("I hate this product, it is terrible", "negative"),
    ("This is the worst service I have ever experienced", "negative"),
    ("I want a full refund", "negative"),
    ("Never buying from you again", "negative"),
    ("I'm done with this company", "negative"),
    ("You people are completely useless", "negative"),
    ("I am very happy with the service", "positive"),
    ("Thank you so much for your help", "positive"),
    ("I completely understand, I will process your refund immediately", "positive"),
]

print("=== SENTIMENT (cardiffnlp RoBERTa 3-class) ===")
sent = SentimentModel(device=-1)
wrong = 0
for text, expected in tests:
    r = sent.predict(text)
    label = r["label"]
    score = r["score"]
    # neutral is acceptable for ambiguous cases, but positive for negative text is wrong
    is_correct = label == expected or (label == "neutral" and expected in ("positive", "negative"))
    ok = "✓" if label == expected else ("~ neutral" if label == "neutral" else "✗ WRONG")
    if label != expected and label != "neutral":
        wrong += 1
    print(f"  {ok:12s} {label:8s} ({score:.2f}) | {text}")
print(f"\n  Hard errors: {wrong}/{len(tests)}")

print("\n=== EMOTION (GoEmotions, threshold=0.15) ===")
emo = EmotionModel(device=-1)
for text, _ in tests:
    r = emo.predict(text)
    top = ", ".join(f"{e['label']}({e['score']:.2f})" for e in r.emotions[:5])
    print(f"  [{r.category:8s}] {top}")
    print(f"            -> {text}")
