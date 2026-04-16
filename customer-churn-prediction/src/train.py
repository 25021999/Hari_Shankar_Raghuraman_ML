"""
train.py
Customer Churn Prediction — Model Training
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pickle

from data_preprocessing import generate_churn_data, preprocess


def train_model():
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION — MODEL TRAINING")
    print("=" * 60)

    # ── 1. LOAD AND PREPROCESS DATA ──────────────────────────
    print("\n[1/5] Generating and preprocessing data...")
    df = generate_churn_data(5000)
    X, y, feature_names, scaler = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"      Churn rate — Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    # ── 2. BASELINE MODEL ────────────────────────────────────
    print("\n[2/5] Training baseline (Random Forest)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"      Baseline AUC-ROC: {rf_auc:.4f}")

    # ── 3. MAIN MODEL WITH HYPERPARAMETER TUNING ─────────────
    print("\n[3/5] Training Gradient Boosting with hyperparameter tuning...")
    param_grid = {
        "n_estimators":   [100, 200],
        "learning_rate":  [0.05, 0.10],
        "max_depth":      [3, 5],
        "subsample":      [0.8, 1.0],
    }
    gb_base = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(
        gb_base, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    print(f"      Best params: {grid_search.best_params_}")
    print(f"      Best CV AUC: {grid_search.best_score_:.4f}")

    # ── 4. EVALUATE ──────────────────────────────────────────
    print("\n[4/5] Evaluating on test set...")
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    auc         = roc_auc_score(y_test, y_prob)
    cv_scores   = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

    print(f"\n      {'Metric':<30} {'Value'}")
    print(f"      {'-'*45}")
    print(f"      {'AUC-ROC':<30} {auc:.4f}")
    print(f"      {'CV AUC (mean ± std)':<30} {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n      Classification Report:")
    print("      " + classification_report(y_test, y_pred,
          target_names=["No Churn", "Churn"]).replace("\n", "\n      "))

    # ── 5. SAVE ARTIFACTS ────────────────────────────────────
    print("\n[5/5] Saving model and artifacts...")
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    with open("outputs/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("outputs/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("outputs/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    metrics = {
        "auc_roc":       round(auc, 4),
        "cv_auc_mean":   round(cv_scores.mean(), 4),
        "cv_auc_std":    round(cv_scores.std(), 4),
        "baseline_auc":  round(rf_auc, 4),
        "best_params":   grid_search.best_params_
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── PLOTS ────────────────────────────────────────────────
    _plot_confusion_matrix(y_test, y_pred)
    _plot_roc_curve(y_test, y_prob, rf, X_test, rf_auc, auc)
    _plot_feature_importance(model, feature_names)
    _plot_churn_probability_dist(y_prob, y_test)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Final AUC-ROC : {auc:.4f}")
    print(f"  Baseline AUC  : {rf_auc:.4f}")
    print(f"  Improvement   : +{(auc - rf_auc):.4f}")
    print("  Artifacts saved to outputs/")
    print("=" * 60)

    return model, scaler, feature_names, metrics


def _plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title("Confusion Matrix — Churn Prediction", fontsize=14, fontweight="bold")
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/plots/confusion_matrix.png", dpi=150)
    plt.close()
    print("      Saved: outputs/plots/confusion_matrix.png")


def _plot_roc_curve(y_test, y_prob, rf, X_test, rf_auc, gb_auc):
    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])

    plt.figure(figsize=(7, 5))
    plt.plot(fpr_gb, tpr_gb, color="steelblue", lw=2,
             label=f"Gradient Boosting (AUC = {gb_auc:.3f})")
    plt.plot(fpr_rf, tpr_rf, color="darkorange", lw=2, linestyle="--",
             label=f"Random Forest baseline (AUC = {rf_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1, label="Random classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig("outputs/plots/roc_curve.png", dpi=150)
    plt.close()
    print("      Saved: outputs/plots/roc_curve.png")


def _plot_feature_importance(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(12)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="steelblue", edgecolor="white")
    plt.title("Top Feature Importances — Gradient Boosting", fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/plots/feature_importance.png", dpi=150)
    plt.close()
    print("      Saved: outputs/plots/feature_importance.png")


def _plot_churn_probability_dist(y_prob, y_test):
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob[y_test == 0], bins=40, alpha=0.6,
             color="steelblue", label="No Churn", density=True)
    plt.hist(y_prob[y_test == 1], bins=40, alpha=0.6,
             color="tomato", label="Churn", density=True)
    plt.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Decision threshold (0.5)")
    plt.xlabel("Predicted Churn Probability", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Predicted Churn Probability Distribution", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("outputs/plots/probability_distribution.png", dpi=150)
    plt.close()
    print("      Saved: outputs/plots/probability_distribution.png")


if __name__ == "__main__":
    train_model()
