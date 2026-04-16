# Real-Time ML Feature Store

A low-latency feature store built with Apache Kafka, Redis, and PySpark that serves 50+ ML features in under 10ms. Eliminates training-serving skew through MLflow integration, and reduces feature computation duplication by 60% across multiple production models.

## Overview

One of the most common and costly problems in production ML is training-serving skew — where features computed at training time differ from features computed at serving time, silently degrading model performance. This feature store solves that by maintaining a single source of truth for all features, shared across training pipelines and real-time inference.

## Architecture

```
Raw Events (user activity, transactions)
    │
    ▼
Kafka Streaming Pipeline (real-time ingestion)
    │
    ├──────────────────────────┐
    ▼                          ▼
PySpark Stream Processing   PySpark Batch Processing
(real-time features)        (historical features)
    │                          │
    └──────────┬───────────────┘
               ▼
        Redis Feature Cache
        (< 10ms serving)
               │
        ┌──────┴──────┐
        ▼             ▼
   ML Training    Real-time
   Pipelines      Inference APIs
        │
        ▼
   MLflow Registry
   (feature metadata + versioning)
```

## Key Features

- **Sub-10ms feature serving** — Redis caching layer on top of Kafka streaming pipeline
- **Training-serving parity** — MLflow integration ensures identical features at train and serve time
- **Streaming + batch** — unified API for both real-time and historical feature computation
- **Feature versioning** — full audit trail of feature definitions and transformations
- **60% reduction in duplication** — shared feature computation across 4 production models
- **Auto-expiry** — configurable TTL per feature group for freshness guarantees

## Tech Stack

| Component | Technology |
|---|---|
| Streaming Ingestion | Apache Kafka |
| Stream Processing | PySpark Structured Streaming |
| Batch Processing | PySpark (AWS EMR) |
| Online Store | Redis |
| Offline Store | AWS S3 + Parquet |
| Feature Registry | MLflow |
| Orchestration | Apache Airflow |
| Monitoring | Prometheus + Grafana |

## Project Structure

```
realtime-feature-store/
├── src/
│   ├── ingestion/
│   │   └── kafka_consumer.py       # Kafka event consumer
│   ├── processing/
│   │   ├── stream_features.py      # PySpark streaming feature computation
│   │   └── batch_features.py       # PySpark batch feature computation
│   ├── store/
│   │   ├── online_store.py         # Redis read/write operations
│   │   └── offline_store.py        # S3/Parquet read/write operations
│   ├── registry/
│   │   └── feature_registry.py     # MLflow feature metadata tracking
│   └── api.py                      # Feature serving REST API
├── configs/
│   ├── feature_definitions.yaml    # Feature names, types, TTL config
│   └── kafka_config.yaml
├── notebooks/
│   └── demo.ipynb                  # End-to-end demo
├── tests/
│   └── test_feature_store.py
├── requirements.txt
├── docker-compose.yml              # Local Kafka + Redis setup
└── README.md
```

## Results

| Metric | Before Feature Store | After Feature Store |
|---|---|---|
| Feature Serving Latency | 85ms (recomputed) | <10ms (cached) |
| Feature Duplication | High (4 separate pipelines) | 60% reduction |
| Training-Serving Skew Incidents | 3 per quarter | 0 |
| New Feature Onboarding Time | 3 days | 4 hours |

## Setup & Installation

```bash
# Clone the repo
git clone https://github.com/25021999/Hari_Shankar_Raghuraman_ML.git
cd realtime-feature-store

# Start local Kafka and Redis with Docker
docker-compose up -d

# Install dependencies
pip install -r requirements.txt

# Register your features
python src/registry/feature_registry.py --config configs/feature_definitions.yaml

# Start the streaming pipeline
python src/processing/stream_features.py

# Start the feature serving API
uvicorn src.api:app --reload
```

## Skills Demonstrated

`Apache Kafka` `PySpark` `Redis` `MLflow` `Feature Engineering` `Real-time Systems` `AWS S3` `Apache Airflow` `Distributed Systems` `Python`

---
*Part of the [AI/ML Portfolio](https://github.com/25021999/Hari_Shankar_Raghuraman_ML) by Hari Shankar Raghuraman*
