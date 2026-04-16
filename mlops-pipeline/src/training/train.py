"""
train.py
End-to-End MLOps Pipeline — Model Training with Experiment Tracking

Simulates the full MLOps lifecycle:
  - Data versioning
  - Experiment tracking (MLflow-style logging to JSON)
  - Model registry with staging and production stages
  - Automated evaluation gating
  - Drift detection simulation

In production: uses MLflow, Docker, Kubernetes, Airflow, AWS SageMaker
"""

import os, sys, json, pickle, hashlib, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble        import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (roc_auc_score, f1_score, accuracy_score,
                                      classification_report, confusion_matrix)

np.random.seed(42)
os.makedirs("outputs/plots",    exist_ok=True)
os.makedirs("outputs/registry", exist_ok=True)
os.makedirs("outputs/runs",     exist_ok=True)


# ════════════════════════════════════════════════════════════
# EXPERIMENT TRACKER  (MLflow-compatible interface, JSON backend)
# In production: import mlflow; mlflow.set_experiment(...)
# ════════════════════════════════════════════════════════════

class ExperimentTracker:
    """
    Lightweight experiment tracker that mirrors the MLflow API.
    Logs parameters, metrics and artifacts to JSON files.
    In production replace every method with its mlflow.* equivalent.
    """

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.run_id          = None
        self.run_data        = {}
        self.runs_dir        = "outputs/runs"

    def start_run(self, run_name=None):
        ts           = datetime.datetime.utcnow().isoformat()
        self.run_id  = hashlib.md5(ts.encode()).hexdigest()[:8]
        self.run_data = {
            "run_id":          self.run_id,
            "run_name":        run_name or f"run_{self.run_id}",
            "experiment":      self.experiment_name,
            "start_time":      ts,
            "status":          "RUNNING",
            "params":          {},
            "metrics":         {},
            "artifacts":       [],
        }
        print(f"  [Tracker] Run started  : {self.run_id} ({run_name})")
        return self

    def log_param(self, key, value):
        self.run_data["params"][key] = value

    def log_params(self, params: dict):
        self.run_data["params"].update(params)

    def log_metric(self, key, value):
        self.run_data["metrics"][key] = round(float(value), 6)
        print(f"  [Tracker] Metric logged: {key} = {value:.4f}")

    def log_artifact(self, path):
        self.run_data["artifacts"].append(path)

    def end_run(self, status="FINISHED"):
        self.run_data["status"]   = status
        self.run_data["end_time"] = datetime.datetime.utcnow().isoformat()
        path = os.path.join(self.runs_dir, f"run_{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(self.run_data, f, indent=2)
        print(f"  [Tracker] Run finished : {self.run_id} -> {path}")
        return self.run_data

    def __enter__(self):  return self
    def __exit__(self, *_): self.end_run()


# ════════════════════════════════════════════════════════════
# MODEL REGISTRY  (mirrors MLflow Model Registry)
# ════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    Simple model registry with staging/production promotion logic.
    In production: use mlflow.register_model() and MLflow Model Registry UI.
    """

    REGISTRY_FILE = "outputs/registry/registry.json"

    def __init__(self):
        if os.path.exists(self.REGISTRY_FILE):
            with open(self.REGISTRY_FILE) as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": []}

    def register(self, model_name, run_id, metrics, stage="Staging"):
        entry = {
            "model_name":   model_name,
            "version":      len(self.registry["models"]) + 1,
            "run_id":       run_id,
            "stage":        stage,
            "metrics":      metrics,
            "registered_at":datetime.datetime.utcnow().isoformat(),
        }
        self.registry["models"].append(entry)
        self._save()
        print(f"  [Registry] Registered: {model_name} v{entry['version']} -> {stage}")
        return entry

    def promote(self, model_name, version, to_stage="Production"):
        for m in self.registry["models"]:
            if m["model_name"] == model_name and m["version"] == version:
                m["stage"] = to_stage
                m["promoted_at"] = datetime.datetime.utcnow().isoformat()
        self._save()
        print(f"  [Registry] Promoted  : {model_name} v{version} -> {to_stage}")

    def get_production_model(self, model_name):
        prod = [m for m in self.registry["models"]
                if m["model_name"] == model_name and m["stage"] == "Production"]
        return prod[-1] if prod else None

    def _save(self):
        with open(self.REGISTRY_FILE, "w") as f:
            json.dump(self.registry, f, indent=2)

    def print_registry(self):
        print(f"\n  {'Model':<25} {'Version':<8} {'Stage':<12} {'AUC-ROC':<10}")
        print(f"  {'-'*60}")
        for m in self.registry["models"]:
            print(f"  {m['model_name']:<25} {m['version']:<8} "
                  f"{m['stage']:<12} {m['metrics'].get('auc_roc','N/A')}")


# ════════════════════════════════════════════════════════════
# DATA PIPELINE
# ════════════════════════════════════════════════════════════

def generate_data(n=4000, version="v1"):
    """Simulates versioned dataset loaded from data warehouse."""
    tenure          = np.random.randint(1, 72, n)
    monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
    support_calls   = np.random.poisson(2, n)
    last_login      = np.random.randint(0, 90, n)
    num_products    = np.random.randint(1, 6, n)
    contract        = np.random.choice([0, 1, 2], n, p=[0.55, 0.25, 0.20])
    senior          = np.random.choice([0, 1], n, p=[0.84, 0.16])
    total_charges   = np.round(monthly_charges * tenure, 2)

    churn_prob = np.clip(
        0.05 + 0.25*(contract==0) + 0.10*(monthly_charges>80)
        - 0.15*(tenure>24)        + 0.10*(support_calls>3)
        + 0.08*(last_login>60)    - 0.08*(num_products>3),
        0.02, 0.95
    )
    y = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "tenure": tenure, "monthly_charges": monthly_charges,
        "total_charges": total_charges, "support_calls": support_calls,
        "last_login_days": last_login, "num_products": num_products,
        "contract_type": contract, "senior_citizen": senior,
        "charges_per_product": monthly_charges / num_products,
        "inactive": (last_login > 60).astype(int),
        "high_support": (support_calls > 3).astype(int),
    })
    data_hash = hashlib.md5(df.to_csv().encode()).hexdigest()[:8]
    return df, y, {"version": version, "hash": data_hash, "rows": n}


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    return (pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns),
            pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns),
            scaler)


# ════════════════════════════════════════════════════════════
# DRIFT DETECTOR
# ════════════════════════════════════════════════════════════

def check_drift(reference_metrics, current_metrics, threshold=0.05):
    """
    Production drift detection: compares live model metrics against baseline.
    Triggers retraining when performance drops beyond threshold.
    """
    results = {}
    for metric in ["auc_roc", "f1_weighted"]:
        ref = reference_metrics.get(metric, 0)
        cur = current_metrics.get(metric, 0)
        drop = ref - cur
        results[metric] = {
            "reference":  round(ref, 4),
            "current":    round(cur, 4),
            "drop":       round(drop, 4),
            "drifted":    drop > threshold,
        }
    any_drift = any(v["drifted"] for v in results.values())
    return any_drift, results


# ════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ════════════════════════════════════════════════════════════

def run_pipeline():
    print("=" * 65)
    print("END-TO-END MLOps PIPELINE")
    print("=" * 65)

    tracker  = ExperimentTracker("churn_prediction_mlops")
    registry = ModelRegistry()

    # ── STEP 1: DATA ─────────────────────────────────────────
    print("\n[STEP 1] Data ingestion and versioning...")
    df, y, data_meta = generate_data(4000, version="v1.2")
    print(f"  Dataset  : {data_meta['rows']} rows | hash: {data_meta['hash']}")
    print(f"  Churn    : {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train_s, X_test_s, scaler = scale_data(X_train, X_test)

    # ── STEP 2: TRAIN 3 MODELS, COMPARE ──────────────────────
    print("\n[STEP 2] Training and comparing 3 candidate models...")
    candidates = {
        "LogisticRegression":    LogisticRegression(max_iter=1000, C=1.0),
        "RandomForest":          RandomForestClassifier(n_estimators=150, random_state=42),
        "GradientBoosting":      GradientBoostingClassifier(n_estimators=150,
                                   learning_rate=0.08, max_depth=4, random_state=42),
    }

    run_results = {}
    for name, model in candidates.items():
        with tracker.start_run(run_name=name):
            tracker.log_params({"model": name, "data_version": data_meta["version"],
                                  "data_hash": data_meta["hash"]})
            model.fit(X_train_s, y_train)

            y_pred = model.predict(X_test_s)
            y_prob = (model.predict_proba(X_test_s)[:, 1]
                      if hasattr(model, "predict_proba")
                      else model.decision_function(X_test_s))

            metrics = {
                "auc_roc":      roc_auc_score(y_test, y_prob),
                "f1_weighted":  f1_score(y_test, y_pred, average="weighted"),
                "accuracy":     accuracy_score(y_test, y_pred),
            }
            for k, v in metrics.items():
                tracker.log_metric(k, v)

            run_results[name] = {
                "model":   model,
                "metrics": metrics,
                "run_id":  tracker.run_id,
                "y_pred":  y_pred,
                "y_prob":  y_prob,
            }

    # ── STEP 3: SELECT BEST MODEL ─────────────────────────────
    print("\n[STEP 3] Model comparison and selection...")
    print(f"\n  {'Model':<25} {'AUC-ROC':<10} {'F1':<10} {'Accuracy'}")
    print(f"  {'-'*55}")
    for name, r in run_results.items():
        m = r["metrics"]
        print(f"  {name:<25} {m['auc_roc']:.4f}    {m['f1_weighted']:.4f}    {m['accuracy']:.4f}")

    best_name = max(run_results, key=lambda n: run_results[n]["metrics"]["auc_roc"])
    best      = run_results[best_name]
    print(f"\n  Best model: {best_name} (AUC = {best['metrics']['auc_roc']:.4f})")

    # ── STEP 4: REGISTER AND PROMOTE ─────────────────────────
    print("\n[STEP 4] Registering and promoting best model...")
    entry = registry.register(best_name, best["run_id"], best["metrics"], stage="Staging")

    # Promote to production if AUC > 0.70
    if best["metrics"]["auc_roc"] > 0.70:
        registry.promote(best_name, entry["version"], to_stage="Production")
        print(f"  AUC {best['metrics']['auc_roc']:.4f} > 0.70 threshold. Promoting to Production.")
    else:
        print(f"  AUC below threshold. Staying in Staging.")

    registry.print_registry()

    # ── STEP 5: DRIFT DETECTION ───────────────────────────────
    print("\n[STEP 5] Simulating drift detection...")
    # Simulate a degraded model (future production metrics)
    degraded_metrics = {
        "auc_roc":     best["metrics"]["auc_roc"] - 0.08,
        "f1_weighted": best["metrics"]["f1_weighted"] - 0.07,
    }
    drifted, drift_results = check_drift(best["metrics"], degraded_metrics, threshold=0.05)
    print(f"\n  Drift Detection Results:")
    for metric, result in drift_results.items():
        flag = "DRIFT DETECTED" if result["drifted"] else "OK"
        print(f"  {metric:<15} ref={result['reference']} cur={result['current']} "
              f"drop={result['drop']} [{flag}]")
    print(f"\n  Retraining triggered: {drifted}")

    # ── STEP 6: SAVE + PLOTS ──────────────────────────────────
    print("\n[STEP 6] Saving artifacts and generating plots...")
    with open("outputs/best_model.pkl", "wb") as f:
        pickle.dump(best["model"], f)
    with open("outputs/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    _plot_model_comparison(run_results)
    _plot_confusion(y_test, best["y_pred"], best_name)
    _plot_pipeline_runs(run_results)
    _plot_drift_simulation(best["metrics"])

    # ── SUMMARY ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print(f"  Best Model    : {best_name}")
    print(f"  AUC-ROC       : {best['metrics']['auc_roc']:.4f}")
    print(f"  Stage         : Production")
    print(f"  Runs logged   : {len(run_results)}")
    print(f"  Drift check   : {'RETRAINING TRIGGERED' if drifted else 'No drift detected'}")
    print("=" * 65)

    return best, run_results, registry


def _plot_model_comparison(run_results):
    names   = list(run_results.keys())
    aucs    = [r["metrics"]["auc_roc"]     for r in run_results.values()]
    f1s     = [r["metrics"]["f1_weighted"] for r in run_results.values()]
    x       = np.arange(len(names))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, aucs, width, label="AUC-ROC",    color="steelblue",  edgecolor="white")
    ax.bar(x + width/2, f1s,  width, label="F1 Weighted",color="darkorange", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Score"); ax.legend()
    ax.set_title("MLOps Experiment — Model Comparison", fontsize=13, fontweight="bold")
    for i, (a, f) in enumerate(zip(aucs, f1s)):
        ax.text(i - width/2, a + 0.005, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + width/2, f + 0.005, f"{f:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/plots/model_comparison.png", dpi=150)
    plt.close()
    print("  Saved: outputs/plots/model_comparison.png")


def _plot_confusion(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn","Churn"],
                yticklabels=["No Churn","Churn"])
    plt.title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("outputs/plots/best_model_confusion.png", dpi=150)
    plt.close()
    print("  Saved: outputs/plots/best_model_confusion.png")


def _plot_pipeline_runs(run_results):
    names = list(run_results.keys())
    aucs  = [r["metrics"]["auc_roc"] for r in run_results.values()]
    colors = ["tomato" if a < max(aucs) else "steelblue" for a in aucs]

    plt.figure(figsize=(9, 4))
    bars = plt.barh(names, aucs, color=colors, edgecolor="white")
    plt.axvline(0.70, color="black", linestyle="--", lw=1.5, label="Production threshold (0.70)")
    plt.xlabel("AUC-ROC")
    plt.title("MLOps Experiment Runs — AUC-ROC Comparison", fontsize=12, fontweight="bold")
    plt.legend(fontsize=9)
    for bar, val in zip(bars, aucs):
        plt.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("outputs/plots/experiment_runs.png", dpi=150)
    plt.close()
    print("  Saved: outputs/plots/experiment_runs.png")


def _plot_drift_simulation(reference_metrics):
    weeks = list(range(1, 13))
    np.random.seed(99)
    noise = np.random.normal(0, 0.008, 12)
    drift = np.linspace(0, -0.12, 12)
    auc_over_time = np.clip(reference_metrics["auc_roc"] + drift + noise, 0.50, 1.0)
    threshold = reference_metrics["auc_roc"] - 0.05

    plt.figure(figsize=(10, 4))
    plt.plot(weeks, auc_over_time, marker="o", color="steelblue", lw=2, label="AUC-ROC (live)")
    plt.axhline(reference_metrics["auc_roc"], color="green",  linestyle="--", lw=1.5, label="Baseline")
    plt.axhline(threshold,                    color="tomato", linestyle="--", lw=1.5, label="Drift threshold")
    drift_weeks = [w for w, a in zip(weeks, auc_over_time) if a < threshold]
    if drift_weeks:
        plt.axvline(drift_weeks[0], color="tomato", linestyle=":", lw=2, label=f"Retraining triggered (week {drift_weeks[0]})")
    plt.xlabel("Week"); plt.ylabel("AUC-ROC")
    plt.title("Model Drift Detection — AUC Over Time", fontsize=12, fontweight="bold")
    plt.legend(fontsize=9); plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig("outputs/plots/drift_simulation.png", dpi=150)
    plt.close()
    print("  Saved: outputs/plots/drift_simulation.png")


if __name__ == "__main__":
    run_pipeline()
