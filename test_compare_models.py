#!/usr/bin/env python3
"""Compare sentiment models on tricky inputs."""

from transformers import pipeline

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

models = [
    ("distilbert-base-uncased-finetuned-sst-2-english", {"POSITIVE": "positive", "NEGATIVE": "negative"}),
    ("cardiffnlp/twitter-roberta-base-sentiment-latest", {"positive": "positive", "negative": "negative", "neutral": "neutral"}),
]

for model_name, label_map in models:
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")
    clf = pipeline("sentiment-analysis", model=model_name, device=-1)
    wrong = 0
    for text, expected in tests:
        r = clf(text)[0]
        raw_label = r["label"]
        label = label_map.get(raw_label, raw_label).lower()
        # For 3-class model, treat neutral as wrong for both pos/neg expected
        ok = "✓" if label == expected else "✗ WRONG"
        if label != expected:
            wrong += 1
        print(f"  {ok:10s} {label:8s} ({r['score']:.2f}) | {text}")
    print(f"\n  Accuracy: {len(tests) - wrong}/{len(tests)}")
