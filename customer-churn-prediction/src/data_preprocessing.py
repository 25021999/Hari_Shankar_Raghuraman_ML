"""
data_preprocessing.py
Customer Churn Prediction — Data Generation and Preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)

def generate_churn_data(n_samples=5000):
    """
    Generate realistic synthetic customer data for churn prediction.
    In production this would be replaced by real data from your data warehouse.
    """
    n = n_samples

    tenure          = np.random.randint(1, 72, n)
    monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
    total_charges   = np.round(monthly_charges * tenure * np.random.uniform(0.85, 1.0, n), 2)
    num_products    = np.random.randint(1, 6, n)
    support_calls   = np.random.poisson(2, n)
    last_login_days = np.random.randint(0, 90, n)
    contract_type   = np.random.choice(["Month-to-Month", "One Year", "Two Year"],
                                        n, p=[0.55, 0.25, 0.20])
    payment_method  = np.random.choice(["Electronic Check", "Mailed Check",
                                         "Bank Transfer", "Credit Card"],
                                        n, p=[0.35, 0.25, 0.25, 0.15])
    internet_service = np.random.choice(["DSL", "Fiber Optic", "No"],
                                         n, p=[0.35, 0.45, 0.20])
    gender          = np.random.choice(["Male", "Female"], n)
    senior_citizen  = np.random.choice([0, 1], n, p=[0.84, 0.16])
    paperless       = np.random.choice([0, 1], n, p=[0.40, 0.60])

    # Churn probability based on realistic business logic
    churn_prob = (
        0.05
        + 0.25 * (contract_type == "Month-to-Month")
        + 0.10 * (monthly_charges > 80)
        - 0.15 * (tenure > 24)
        + 0.10 * (support_calls > 3)
        + 0.08 * (last_login_days > 60)
        - 0.08 * (num_products > 3)
        + 0.05 * (internet_service == "Fiber Optic")
        + 0.04 * (senior_citizen == 1)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id":      [f"C{str(i).zfill(5)}" for i in range(n)],
        "gender":           gender,
        "senior_citizen":   senior_citizen,
        "tenure":           tenure,
        "num_products":     num_products,
        "paperless_billing":paperless,
        "monthly_charges":  monthly_charges,
        "total_charges":    total_charges,
        "support_calls":    support_calls,
        "last_login_days":  last_login_days,
        "contract_type":    contract_type,
        "payment_method":   payment_method,
        "internet_service": internet_service,
        "churn":            churn
    })
    return df


def preprocess(df):
    """
    Clean and encode the raw dataframe.
    Returns X (features), y (target), feature_names, scaler.
    """
    data = df.copy()

    # Drop ID column
    data.drop(columns=["customer_id"], inplace=True)

    # Encode binary columns
    data["gender"] = (data["gender"] == "Male").astype(int)

    # Label-encode categorical columns
    le = LabelEncoder()
    for col in ["contract_type", "payment_method", "internet_service"]:
        data[col] = le.fit_transform(data[col])

    # Feature engineering
    data["charges_per_product"]  = data["monthly_charges"] / data["num_products"]
    data["tenure_group"]         = pd.cut(data["tenure"],
                                          bins=[0, 12, 24, 48, 72],
                                          labels=[0, 1, 2, 3]).astype(int)
    data["high_support"]         = (data["support_calls"] > 3).astype(int)
    data["inactive_customer"]    = (data["last_login_days"] > 60).astype(int)
    data["avg_monthly_total"]    = data["total_charges"] / (data["tenure"] + 1)

    y = data.pop("churn") if "churn" in data.columns else pd.Series([0] * len(data))
    X = data

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y, X.columns.tolist(), scaler


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_churn_data(5000)
    df.to_csv("data/churn_raw.csv", index=False)
    print(f"Dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Churn rate: {df['churn'].mean():.1%}")
    print(df.head())
