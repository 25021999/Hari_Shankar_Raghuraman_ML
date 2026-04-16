"""
predict.py
NLP Sentiment Analysis — Single and Batch Prediction
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import pandas as pd
import numpy as np
from preprocessing.text_processor import clean_text, generate_nlp_dataset


def load_models():
    with open("outputs/sentiment_model.pkl", "rb") as f:
        sentiment_model = pickle.load(f)
    with open("outputs/category_model.pkl", "rb") as f:
        category_model = pickle.load(f)
    with open("outputs/label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return sentiment_model, category_model, encoders


def predict_single(text: str) -> dict:
    """Predict sentiment and category for a single text input."""
    sentiment_model, category_model, encoders = load_models()

    cleaned = clean_text(text)

    # Sentiment prediction with confidence
    sent_probs  = sentiment_model.predict_proba([cleaned])[0]
    sent_label  = sentiment_model.predict([cleaned])[0]
    sentiment   = encoders["sentiment"].inverse_transform([sent_label])[0]
    confidence  = round(float(sent_probs.max()), 4)

    # Category prediction
    cat_label   = category_model.predict([cleaned])[0]
    category    = encoders["category"].inverse_transform([cat_label])[0]

    priority    = "HIGH" if sentiment == "negative" else \
                  "MEDIUM" if sentiment == "neutral" else "LOW"

    return {
        "original_text":  text[:100] + "..." if len(text) > 100 else text,
        "sentiment":      sentiment,
        "confidence":     confidence,
        "category":       category,
        "priority":       priority,
        "action":         _get_action(sentiment, category)
    }


def predict_batch(texts: list) -> pd.DataFrame:
    """Predict sentiment and category for a list of texts."""
    sentiment_model, category_model, encoders = load_models()
    cleaned = [clean_text(t) for t in texts]

    sent_labels = sentiment_model.predict(cleaned)
    cat_labels  = category_model.predict(cleaned)
    sent_probs  = sentiment_model.predict_proba(cleaned)

    results = pd.DataFrame({
        "text":       [t[:60] + "..." if len(t) > 60 else t for t in texts],
        "sentiment":  encoders["sentiment"].inverse_transform(sent_labels),
        "confidence": np.round(sent_probs.max(axis=1), 3),
        "category":   encoders["category"].inverse_transform(cat_labels),
    })
    results["priority"] = results["sentiment"].map(
        {"negative": "HIGH", "neutral": "MEDIUM", "positive": "LOW"}
    )
    return results


def _get_action(sentiment, category):
    actions = {
        ("negative", "billing"):           "Immediate billing review. Escalate to finance team.",
        ("negative", "technical_support"): "Create urgent tech ticket. Assign senior engineer.",
        ("negative", "cancellation"):      "Route to retention team. Offer discount immediately.",
        ("negative", "fraud_report"):      "Flag account. Alert fraud team within 1 hour.",
        ("negative", "escalation"):        "Escalate to manager. Response required within 2 hours.",
        ("neutral",  "general_inquiry"):   "Route to support queue. Standard SLA applies.",
        ("positive", "product_feedback"):  "Tag as positive review. Share with product team.",
    }
    return actions.get((sentiment, category),
                       "Route to appropriate team based on category.")


if __name__ == "__main__":
    print("=" * 62)
    print("SINGLE TEXT PREDICTION DEMO")
    print("=" * 62)

    test_inputs = [
        "I have been trying to get a refund for three weeks and nobody is responding to my emails. This is completely unacceptable.",
        "Just received my order and everything looks great. Very happy with the quality and fast delivery.",
        "I need to update my payment method but cannot find the option in my account settings.",
        "The app keeps crashing every time I try to open the dashboard. I have reinstalled it twice.",
        "I would really love to see a dark mode feature added. It would make the interface much easier to use at night.",
    ]

    for text in test_inputs:
        result = predict_single(text)
        print(f"\n  Input     : {result['original_text']}")
        print(f"  Sentiment : {result['sentiment'].upper()} (confidence: {result['confidence']:.1%})")
        print(f"  Category  : {result['category']}")
        print(f"  Priority  : {result['priority']}")
        print(f"  Action    : {result['action']}")
        print(f"  {'-'*58}")

    print("\n\n" + "=" * 62)
    print("BATCH PREDICTION DEMO")
    print("=" * 62)
    df = generate_nlp_dataset(10)
    results = predict_batch(df["text"].tolist())
    print(results.to_string(index=False))
