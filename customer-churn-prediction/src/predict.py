"""
predict.py
Customer Churn Prediction — Batch and Single Prediction
"""

import pandas as pd
import numpy as np
import pickle
import json
import os

from data_preprocessing import preprocess, generate_churn_data


def load_artifacts():
    with open("outputs/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("outputs/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("outputs/feature_names.json") as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


def predict_batch(df):
    """Run predictions on a full dataframe."""
    model, scaler, feature_names = load_artifacts()
    X, _, _, _ = preprocess(df)
    X = X[feature_names]
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    results = df[["customer_id"]].copy()
    results["churn_probability"] = np.round(probs, 4)
    results["predicted_churn"]   = preds
    results["risk_level"] = pd.cut(
        probs,
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["LOW", "MEDIUM", "HIGH"]
    )
    return results


def predict_single(customer_data: dict):
    """
    Predict churn for a single customer.
    customer_data: dict with all raw feature values
    """
    model, scaler, feature_names = load_artifacts()
    df = pd.DataFrame([customer_data])
    X, _, _, _ = preprocess(df)
    X = X[feature_names]
    prob = model.predict_proba(X)[0, 1]

    risk = "HIGH" if prob > 0.60 else "MEDIUM" if prob > 0.30 else "LOW"
    return {
        "churn_probability": round(float(prob), 4),
        "predicted_churn":   int(prob > 0.5),
        "risk_level":        risk,
        "recommendation":    _get_recommendation(risk, customer_data)
    }


def _get_recommendation(risk, data):
    recs = []
    if risk == "HIGH":
        recs.append("Immediate outreach recommended.")
        if data.get("contract_type") == "Month-to-Month":
            recs.append("Offer discounted annual contract.")
        if data.get("support_calls", 0) > 3:
            recs.append("Escalate open support issues.")
    elif risk == "MEDIUM":
        recs.append("Monitor closely. Consider loyalty offer.")
    else:
        recs.append("Low risk. Continue standard engagement.")
    return " ".join(recs)


if __name__ == "__main__":
    print("=" * 55)
    print("BATCH PREDICTION DEMO")
    print("=" * 55)
    df = generate_churn_data(20)
    results = predict_batch(df)
    print(results.to_string(index=False))

    print("\n" + "=" * 55)
    print("SINGLE CUSTOMER PREDICTION DEMO")
    print("=" * 55)
    sample_customers = [
        {
            "customer_id":      "C99001",
            "gender":           "Male",
            "senior_citizen":   0,
            "tenure":           3,
            "num_products":     1,
            "paperless_billing":1,
            "monthly_charges":  95.0,
            "total_charges":    285.0,
            "support_calls":    5,
            "last_login_days":  70,
            "contract_type":    "Month-to-Month",
            "payment_method":   "Electronic Check",
            "internet_service": "Fiber Optic",
        },
        {
            "customer_id":      "C99002",
            "gender":           "Female",
            "senior_citizen":   0,
            "tenure":           48,
            "num_products":     4,
            "paperless_billing":1,
            "monthly_charges":  45.0,
            "total_charges":    2160.0,
            "support_calls":    1,
            "last_login_days":  5,
            "contract_type":    "Two Year",
            "payment_method":   "Bank Transfer",
            "internet_service": "DSL",
        },
    ]

    for customer in sample_customers:
        cid = customer["customer_id"]
        result = predict_single(customer)
        print(f"\n  Customer: {cid}")
        print(f"  Churn Probability : {result['churn_probability']:.1%}")
        print(f"  Risk Level        : {result['risk_level']}")
        print(f"  Recommendation    : {result['recommendation']}")
