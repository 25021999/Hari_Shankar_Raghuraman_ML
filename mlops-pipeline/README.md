# End-to-End MLOps Pipeline

A production-grade MLOps framework that automates the full machine learning lifecycle including data ingestion, model training, versioning, deployment and monitoring. Built with MLflow, Docker, Kubernetes and Apache Airflow, this pipeline reduces model deployment time significantly and ensures reliable, reproducible ML workflows at scale.

## Overview

Most ML projects fail not because of bad models but because of poor infrastructure. Models trained in notebooks never make it to production, and when they do, they degrade silently with no alerting. This pipeline solves that by treating ML workflows the same way software engineers treat application code — with version control, automated testing, CI/CD and monitoring built in from the start.

## Architecture

```
Data Source (S3, databases, streaming)
    │
    ▼
Apache Airflow DAG (orchestration and scheduling)
    │
    ▼
dbt Data Transformation (SQL-based feature preparation)
    │
    ▼
Model Training Script (Scikit-learn, XGBoost, TensorFlow)
    │
    ▼
MLflow Experiment Tracking
(parameters, metrics, artifacts logged automatically)
    │
    ▼
MLflow Model Registry (staging and production stages)
    │
    ▼
Docker Container Build and Push to ECR
    │
    ▼
Kubernetes Deployment on AWS EKS
    │
    ▼
Production Model Serving (FastAPI)
    │
    ▼
Monitoring and Drift Detection
(performance metrics, feature drift, alerting)
```

## Key Features

- Fully automated model training and deployment triggered by Airflow DAGs
- MLflow experiment tracking logs every run with parameters, metrics and model artifacts
- Docker containerization ensures identical environments from development to production
- Kubernetes deployment with auto-scaling handles variable inference load
- dbt transformations create reproducible, tested feature datasets
- Automated model drift detection triggers retraining when performance degrades
- CI/CD pipeline runs tests, builds Docker image and deploys on every code push

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | Apache Airflow |
| Data Transformation | dbt |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| Containerization | Docker |
| Container Orchestration | Kubernetes (AWS EKS) |
| Model Serving | FastAPI |
| CI/CD | GitHub Actions |
| Cloud Storage | AWS S3 |
| Container Registry | AWS ECR |
| Monitoring | Prometheus, Grafana |

## Project Structure

```
mlops-pipeline/
├── airflow/
│   └── dags/
│       ├── training_pipeline.py      # End-to-end training DAG
│       └── retraining_trigger.py     # Drift-based retraining DAG
├── dbt/
│   ├── models/
│   │   ├── staging/                  # Raw data cleaning models
│   │   └── features/                 # Feature engineering models
│   └── tests/                        # Data quality tests
├── src/
│   ├── training/
│   │   ├── train.py                  # Model training with MLflow logging
│   │   └── evaluate.py               # Evaluation and registry promotion
│   ├── serving/
│   │   └── app.py                    # FastAPI inference service
│   └── monitoring/
│       ├── drift_detector.py         # Feature and prediction drift
│       └── alerting.py               # Slack and email alerts
├── docker/
│   └── Dockerfile                    # Production container definition
├── kubernetes/
│   ├── deployment.yaml               # K8s deployment configuration
│   ├── service.yaml                  # K8s service configuration
│   └── hpa.yaml                      # Horizontal pod autoscaler
├── .github/
│   └── workflows/
│       └── ci_cd.yaml                # GitHub Actions CI/CD pipeline
├── notebooks/
│   └── pipeline_demo.ipynb           # Full pipeline walkthrough
├── tests/
│   ├── test_training.py
│   └── test_serving.py
├── requirements.txt
└── README.md
```

## Results

| Metric | Before MLOps Pipeline | After MLOps Pipeline |
|---|---|---|
| Model Deployment Time | 3 to 5 days (manual) | Under 2 hours (automated) |
| Experiment Reproducibility | Low, notebook-based | Full, every run tracked |
| Model Degradation Detection | Weeks or never | Under 4 days |
| Deployment Errors | Frequent manual mistakes | Zero since automation |
| New Model Onboarding Time | 1 to 2 weeks | 1 to 2 days |

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/25021999/Hari_Shankar_Raghuraman_ML.git
cd mlops-pipeline

# Start local services (MLflow, Airflow) with Docker Compose
docker-compose up -d

# Install Python dependencies
pip install -r requirements.txt

# Initialize dbt
cd dbt && dbt deps && dbt run

# Trigger a training run via Airflow
# Open http://localhost:8080, enable the training_pipeline DAG

# View experiments in MLflow UI
# Open http://localhost:5000
```

## CI/CD Flow

Every push to the main branch automatically:

1. Runs unit tests on training and serving code
2. Runs dbt data quality tests
3. Builds and pushes a Docker image to AWS ECR
4. Deploys the updated image to Kubernetes on AWS EKS
5. Runs a smoke test on the live endpoint
6. Sends a Slack notification on success or failure

## Skills Demonstrated

`MLflow` `Apache Airflow` `dbt` `Docker` `Kubernetes` `AWS EKS` `AWS ECR` `AWS S3` `FastAPI` `CI/CD` `GitHub Actions` `Model Monitoring` `Drift Detection` `Python`

---
*Part of the [AI/ML Portfolio](https://github.com/25021999/Hari_Shankar_Raghuraman_ML) by Hari Shankar Raghuraman*
