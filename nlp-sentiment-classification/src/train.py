"""
train.py
NLP Sentiment Analysis and Text Classification — Model Training

Baseline  : TF-IDF + Logistic Regression / LinearSVC
Production: Fine-tuned BERT via HuggingFace Transformers (see comments)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pickle, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics         import (classification_report, confusion_matrix,
                                      accuracy_score, f1_score)
from sklearn.preprocessing   import LabelEncoder

from preprocessing.text_processor import generate_nlp_dataset, clean_text

os.makedirs("outputs/plots", exist_ok=True)


def train():
    print("=" * 62)
    print("NLP SENTIMENT ANALYSIS AND TEXT CLASSIFICATION")
    print("=" * 62)

    # ── 1. DATA ──────────────────────────────────────────────
    print("\n[1/5] Generating and preprocessing data...")
    df = generate_nlp_dataset(3000)
    df["clean_text"] = df["text"].apply(clean_text)

    le_sentiment = LabelEncoder()
    le_category  = LabelEncoder()
    df["sentiment_label"] = le_sentiment.fit_transform(df["sentiment"])
    df["category_label"]  = le_category.fit_transform(df["category"])

    print(f"      Total samples    : {len(df)}")
    print(f"      Sentiment classes: {list(le_sentiment.classes_)}")
    print(f"      Category classes : {list(le_category.classes_)}")

    # ── 2. SPLIT ─────────────────────────────────────────────
    X_tr, X_te, ys_tr, ys_te, yc_tr, yc_te = train_test_split(
        df["clean_text"],
        df["sentiment_label"],
        df["category_label"],
        test_size=0.20, random_state=42
    )
    print(f"      Train: {len(X_tr)} | Test: {len(X_te)}")

    # ── 3. SENTIMENT MODEL ───────────────────────────────────
    print("\n[2/5] Training sentiment classifier (TF-IDF + LogReg)...")
    sentiment_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1,2),
                                   sublinear_tf=True, min_df=2)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0,
                                      solver="lbfgs"))
    ])
    sentiment_pipeline.fit(X_tr, ys_tr)
    ys_pred  = sentiment_pipeline.predict(X_te)
    ys_prob  = sentiment_pipeline.predict_proba(X_te)
    sent_acc = accuracy_score(ys_te, ys_pred)
    sent_f1  = f1_score(ys_te, ys_pred, average="weighted")
    cv_sent  = cross_val_score(sentiment_pipeline, df["clean_text"],
                                df["sentiment_label"], cv=5, scoring="f1_weighted")
    print(f"      Sentiment Accuracy : {sent_acc:.4f}")
    print(f"      Sentiment F1       : {sent_f1:.4f}")
    print(f"      CV F1 (mean±std)   : {cv_sent.mean():.4f} ± {cv_sent.std():.4f}")

    # ── 4. CATEGORY CLASSIFIER ───────────────────────────────
    print("\n[3/5] Training category classifier (TF-IDF + LinearSVC)...")
    category_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1,2),
                                   sublinear_tf=True, min_df=2)),
        ("clf",   LinearSVC(max_iter=2000, C=1.0))
    ])
    category_pipeline.fit(X_tr, yc_tr)
    yc_pred  = category_pipeline.predict(X_te)
    cat_acc  = accuracy_score(yc_te, yc_pred)
    cat_f1   = f1_score(yc_te, yc_pred, average="weighted")
    cv_cat   = cross_val_score(category_pipeline, df["clean_text"],
                                df["category_label"], cv=5, scoring="f1_weighted")
    print(f"      Category Accuracy  : {cat_acc:.4f}")
    print(f"      Category F1        : {cat_f1:.4f}")
    print(f"      CV F1 (mean±std)   : {cv_cat.mean():.4f} ± {cv_cat.std():.4f}")

    # ── 5. EVALUATION REPORT ─────────────────────────────────
    print("\n[4/5] Full evaluation reports...")
    print("\n  --- SENTIMENT CLASSIFICATION REPORT ---")
    print(classification_report(ys_te, ys_pred,
          target_names=le_sentiment.classes_))
    print("  --- CATEGORY CLASSIFICATION REPORT ---")
    print(classification_report(yc_te, yc_pred,
          target_names=le_category.classes_))

    # ── PLOTS ────────────────────────────────────────────────
    _plot_sentiment_confusion(ys_te, ys_pred, le_sentiment.classes_)
    _plot_category_confusion(yc_te, yc_pred, le_category.classes_)
    _plot_top_tfidf_features(sentiment_pipeline, le_sentiment.classes_)
    _plot_sentiment_distribution(df)

    # ── 6. SAVE ──────────────────────────────────────────────
    print("\n[5/5] Saving models and artifacts...")
    with open("outputs/sentiment_model.pkl", "wb") as f:
        pickle.dump(sentiment_pipeline, f)
    with open("outputs/category_model.pkl", "wb") as f:
        pickle.dump(category_pipeline, f)
    with open("outputs/label_encoders.pkl", "wb") as f:
        pickle.dump({"sentiment": le_sentiment, "category": le_category}, f)

    metrics = {
        "sentiment_accuracy":  round(sent_acc, 4),
        "sentiment_f1":        round(sent_f1, 4),
        "category_accuracy":   round(cat_acc, 4),
        "category_f1":         round(cat_f1, 4),
        "cv_sentiment_f1":     round(cv_sent.mean(), 4),
        "cv_category_f1":      round(cv_cat.mean(), 4),
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 62)
    print("TRAINING COMPLETE")
    print(f"  Sentiment Accuracy : {sent_acc:.4f}")
    print(f"  Category F1        : {cat_f1:.4f}")
    print("  Models saved to outputs/")
    print("=" * 62)
    return sentiment_pipeline, category_pipeline, le_sentiment, le_category


def _plot_sentiment_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Sentiment Classification — Confusion Matrix",
              fontsize=13, fontweight="bold")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("outputs/plots/sentiment_confusion.png", dpi=150)
    plt.close()
    print("      Saved: outputs/plots/sentiment_confusion.png")


def _plot_category_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 8})
    plt.title("Category Classification — Confusion Matrix",
              fontsize=13, fontweight="bold")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/plots/category_confusion.png", dpi=150)
    plt.close()
    print("      Saved: outputs/plots/category_confusion.png")


def _plot_top_tfidf_features(pipeline, classes):
    tfidf = pipeline.named_steps["tfidf"]
    clf   = pipeline.named_steps["clf"]
    feature_names = np.array(tfidf.get_feature_names_out())

    fig, axes = plt.subplots(1, len(classes), figsize=(16, 5))
    fig.suptitle("Top TF-IDF Features per Sentiment Class",
                 fontsize=13, fontweight="bold")
    for i, (ax, cls) in enumerate(zip(axes, classes)):
        coef  = clf.coef_[i]
        top15 = np.argsort(coef)[-15:]
        ax.barh(feature_names[top15], coef[top15], color="steelblue", edgecolor="white")
        ax.set_title(f"Class: {cls}", fontweight="bold")
        ax.set_xlabel("Coefficient")
        ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig("outputs/plots/top_features.png", dpi=150)
    plt.close()
    print("      Saved: outputs/plots/top_features.png")


def _plot_sentiment_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Dataset Distribution", fontsize=13, fontweight="bold")

    sent_counts = df["sentiment"].value_counts()
    colors = {"positive": "steelblue", "negative": "tomato", "neutral": "goldenrod"}
    axes[0].bar(sent_counts.index,
                sent_counts.values,
                color=[colors[s] for s in sent_counts.index],
                edgecolor="white")
    axes[0].set_title("Sentiment Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(sent_counts.values):
        axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold")

    cat_counts = df["category"].value_counts()
    axes[1].barh(cat_counts.index, cat_counts.values, color="steelblue", edgecolor="white")
    axes[1].set_title("Category Distribution", fontweight="bold")
    axes[1].set_xlabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/plots/distributions.png", dpi=150)
    plt.close()
    print("      Saved: outputs/plots/distributions.png")


if __name__ == "__main__":
    train()
