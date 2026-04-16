# NLP Sentiment Analysis and Text Classification Pipeline

A production-ready NLP pipeline that performs sentiment analysis and multi-class text classification on customer feedback using Transformer-based models (BERT) and traditional NLP approaches (NLTK). Built to process large volumes of unstructured text data and surface actionable insights for business teams.

## Overview

Manually reading and categorizing customer feedback at scale is impossible. This pipeline automatically classifies incoming text by sentiment (positive, negative, neutral) and routes it into business categories such as billing, technical support, product feedback and account issues — enabling teams to act on customer signals in real time instead of waiting for quarterly reports.

## Architecture

```
Raw Text Input (support tickets, reviews, feedback forms)
    │
    ▼
Text Preprocessing (NLTK tokenization, stopword removal, cleaning)
    │
    ▼
Feature Extraction (TF-IDF for baseline, BERT embeddings for deep model)
    │
    ├─────────────────────────────┐
    ▼                             ▼
Sentiment Classifier          Intent Classifier
(positive/negative/neutral)   (12 business categories)
    │                             │
    └──────────────┬──────────────┘
                   ▼
        Structured Output with Confidence Scores
                   │
                   ▼
        Dashboard and Alerting System
```

## Key Features

- Dual model approach: NLTK-based baseline and fine-tuned BERT for production
- Multi-label text classification across 12 business categories
- Confidence scoring on every prediction for human review flagging
- Handles noisy real-world text including typos, slang and mixed languages
- Batch processing for historical data and real-time scoring for new inputs
- Built-in evaluation suite with precision, recall, F1 and confusion matrix reporting

## Tech Stack

| Component | Technology |
|---|---|
| NLP Preprocessing | NLTK, spaCy |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Pretrained Model | BERT (bert-base-uncased) |
| Baseline Model | TF-IDF + Logistic Regression (Scikit-learn) |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Deployment | Docker, AWS EC2 |
| Data Processing | Pandas, PySpark |

## Project Structure

```
nlp-sentiment-classification/
├── src/
│   ├── preprocessing/
│   │   ├── text_cleaner.py          # Normalization, noise removal
│   │   └── nltk_processor.py        # Tokenization, stemming, stopwords
│   ├── models/
│   │   ├── baseline_model.py        # TF-IDF + Logistic Regression
│   │   ├── bert_classifier.py       # Fine-tuned BERT classifier
│   │   └── trainer.py               # Training loop with MLflow logging
│   ├── evaluation/
│   │   └── metrics.py               # F1, precision, recall, confusion matrix
│   ├── api/
│   │   └── app.py                   # FastAPI prediction endpoint
│   └── utils/
│       └── data_loader.py           # Dataset loading and splitting
├── notebooks/
│   ├── data_exploration.ipynb       # Text data analysis and visualization
│   ├── baseline_model.ipynb         # TF-IDF baseline walkthrough
│   └── bert_finetuning.ipynb        # BERT fine-tuning demo with outputs
├── configs/
│   └── training_config.yaml
├── tests/
│   └── test_classifier.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Results

| Model | Sentiment Accuracy | Classification F1 | Inference Speed |
|---|---|---|---|
| NLTK + Rule-based | 71% | N/A | 2ms |
| TF-IDF + Logistic Regression | 79% | 74% | 5ms |
| Fine-tuned BERT (ours) | 93% | 89% | 45ms |

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/25021999/Hari_Shankar_Raghuraman_ML.git
cd nlp-sentiment-classification

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Train the baseline model
python src/models/baseline_model.py --data data/train.csv

# Fine-tune BERT
python src/models/trainer.py --model bert-base-uncased --epochs 3

# Start the API
uvicorn src.api.app:app --reload
```

## Sample API Request

```python
import requests

response = requests.post("http://localhost:8000/analyze", json={
    "text": "I have been trying to resolve my billing issue for 2 weeks and nobody is helping me"
})

print(response.json())
# {
#   "sentiment": "negative",
#   "confidence": 0.96,
#   "category": "billing",
#   "priority": "HIGH"
# }
```

## Skills Demonstrated

`Python` `PyTorch` `HuggingFace Transformers` `BERT` `NLTK` `Scikit-learn` `NLP` `Text Classification` `Sentiment Analysis` `MLflow` `FastAPI` `Docker` `AWS EC2`

---
*Part of the [AI/ML Portfolio](https://github.com/25021999/Hari_Shankar_Raghuraman_ML) by Hari Shankar Raghuraman*
