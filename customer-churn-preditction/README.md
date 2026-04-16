# Customer Churn Prediction System

An end-to-end machine learning pipeline that predicts customer churn using XGBoost and PySpark, deployed as a real-time inference API on AWS SageMaker. Built to replace a legacy rule-based system with a data-driven approach that processes large-scale behavioral data.

## Overview

Customer churn is one of the most costly problems in enterprise businesses. This system ingests raw customer behavioral data, engineers meaningful features, trains an optimized XGBoost classifier, and serves predictions through a low-latency REST API — giving business teams the ability to intervene before a customer churns.

## Architecture

```
Raw Customer Data (behavioral logs, transactions, support tickets)
    │
    ▼
PySpark Data Ingestion and Cleaning Pipeline
    │
    ▼
Feature Engineering (usage patterns, recency, frequency, tenure)
    │
    ▼
XGBoost Classifier Training with Cross-Validation
    │
    ▼
MLflow Experiment Tracking and Model Registry
    │
    ▼
FastAPI Inference Endpoint on AWS SageMaker
    │
    ▼
Real-time Churn Score per Customer
```

## Key Features

- Processes large-scale structured customer data using PySpark for distributed computation
- Feature engineering pipeline covering recency, frequency, monetary value and behavioral signals
- XGBoost classifier with hyperparameter tuning via grid search and cross-validation
- MLflow integration for full experiment tracking, model versioning and registry
- FastAPI REST endpoint deployed on AWS SageMaker for real-time scoring
- Model performance monitoring with automated alerts on metric degradation

## Tech Stack

| Component | Technology |
|---|---|
| Data Processing | PySpark, Apache Spark |
| Feature Engineering | PySpark, Pandas, Scikit-learn |
| Model Training | XGBoost, Scikit-learn |
| Experiment Tracking | MLflow |
| API Framework | FastAPI |
| Deployment | AWS SageMaker, Docker |
| Orchestration | Apache Airflow |
| Storage | AWS S3 |

## Project Structure

```
customer-churn-prediction/
├── src/
│   ├── data/
│   │   ├── ingestion.py          # PySpark data loading and validation
│   │   └── preprocessing.py      # Cleaning, null handling, encoding
│   ├── features/
│   │   └── feature_engineering.py  # RFM features, behavioral signals
│   ├── models/
│   │   ├── train.py              # XGBoost training with cross-validation
│   │   ├── evaluate.py           # Metrics, confusion matrix, ROC curve
│   │   └── predict.py            # Batch and real-time prediction
│   ├── api/
│   │   └── app.py                # FastAPI inference endpoint
│   └── monitoring/
│       └── drift_detector.py     # Feature and prediction drift monitoring
├── notebooks/
│   ├── exploratory_analysis.ipynb   # Data exploration and visualization
│   └── model_training_demo.ipynb    # End-to-end training walkthrough
├── configs/
│   └── model_config.yaml         # Hyperparameters and training settings
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Results

| Metric | Rule-Based System | This Model |
|---|---|---|
| Precision | 61% | 84% |
| Recall | 58% | 79% |
| F1 Score | 59% | 81% |
| False Positive Rate | 39% | 16% |
| AUC-ROC | N/A | 0.91 |

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/25021999/Hari_Shankar_Raghuraman_ML.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run data preprocessing
python src/data/preprocessing.py --input data/raw/ --output data/processed/

# Train the model
python src/models/train.py --config configs/model_config.yaml

# Start the API server
uvicorn src.api.app:app --reload
```

## Sample API Request

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "customer_id": "C12345",
    "tenure_months": 14,
    "monthly_usage": 87.5,
    "support_tickets_last_90d": 3,
    "last_login_days_ago": 12
})

print(response.json())
# {"customer_id": "C12345", "churn_probability": 0.73, "risk_level": "HIGH"}
```

## Skills Demonstrated

`Python` `PySpark` `XGBoost` `Scikit-learn` `Feature Engineering` `MLflow` `FastAPI` `AWS SageMaker` `Docker` `Apache Airflow` `Model Monitoring`

---
*Part of the [AI/ML Portfolio](https://github.com/25021999/Hari_Shankar_Raghuraman_ML) by Hari Shankar Raghuraman*
