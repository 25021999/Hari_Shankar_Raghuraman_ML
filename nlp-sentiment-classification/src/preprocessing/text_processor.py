"""
text_processor.py
NLP Sentiment Analysis — Data Generation and Text Preprocessing

NOTE: In production this uses NLTK for tokenization/stemming and
HuggingFace Transformers (BERT) for deep embeddings.
Here we implement the same preprocessing logic using stdlib + sklearn
so the code runs without heavy dependencies.
"""

import re
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(42)

# ── Stopwords (subset — in production use nltk.corpus.stopwords) ──
STOPWORDS = {
    "i","me","my","we","our","you","your","he","she","it","they","them",
    "is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","shall",
    "a","an","the","and","but","or","nor","so","yet","both","either",
    "not","no","nor","as","at","by","for","in","of","on","to","up","with",
    "about","after","before","between","into","through","during","above",
    "below","from","out","off","over","under","again","then","once","just",
    "also","very","too","more","most","other","some","such","than","that",
    "this","these","those","there","their","what","which","who","how","all"
}

CATEGORIES = [
    "billing", "technical_support", "account_management",
    "product_feedback", "shipping", "cancellation",
    "general_inquiry", "fraud_report", "refund_request",
    "onboarding", "feature_request", "escalation"
]

SENTIMENT_TEMPLATES = {
    "positive": [
        "I absolutely love this product, it works perfectly and exceeded all my expectations.",
        "The customer service team was incredibly helpful and resolved my issue right away.",
        "Amazing experience from start to finish, I will definitely recommend this to everyone.",
        "Everything works exactly as described, very happy with my purchase.",
        "The setup was easy and the product quality is outstanding, great value for money.",
        "I had a wonderful experience using this service, it saved me so much time.",
        "Super fast delivery and the product quality is even better than I expected.",
        "The team went above and beyond to help me, truly exceptional service.",
        "I have been using this for months and it keeps getting better, highly recommend.",
        "Smooth process from order to delivery, zero complaints whatsoever.",
    ],
    "negative": [
        "This is absolutely terrible, I have been waiting three weeks for a response.",
        "The product stopped working after just two days and nobody is helping me.",
        "I was charged twice for the same order and still have not received a refund.",
        "The customer support is non-existent, I keep getting automated replies.",
        "Worst experience ever, the product is completely different from what was advertised.",
        "I have called five times and each agent gives me a different answer, very frustrating.",
        "My account was locked for no reason and I cannot access any of my data.",
        "The billing team made an error and is refusing to acknowledge it, absolutely disgraceful.",
        "This product broke within a week and the warranty process is a complete nightmare.",
        "I requested a cancellation weeks ago and I am still being charged every month.",
    ],
    "neutral": [
        "I received my order today and it looks okay, will update after using it.",
        "The product works as described, nothing special but gets the job done.",
        "Delivery took about a week which is normal for this type of item.",
        "I have some questions about the billing cycle and would like clarification.",
        "The interface is straightforward but could benefit from a few improvements.",
        "My account was updated but I am not sure if all the changes went through.",
        "I am following up on my previous ticket from last Tuesday.",
        "The product is average, works for basic use but nothing remarkable.",
        "I need to update my payment method and am not sure how to do that.",
        "Just checking the status of my refund request submitted last week.",
    ]
}

CATEGORY_KEYWORDS = {
    "billing":            ["charged","invoice","payment","refund","bill","fee","subscription","price"],
    "technical_support":  ["broken","not working","error","bug","crash","install","setup","issue"],
    "account_management": ["account","login","password","access","locked","profile","settings"],
    "product_feedback":   ["product","quality","design","feature","love","terrible","great","awful"],
    "shipping":           ["delivery","shipped","order","package","arrived","tracking","late"],
    "cancellation":       ["cancel","cancellation","stop","end","terminate","subscription"],
    "general_inquiry":    ["question","how","what","information","help","wondering","ask"],
    "fraud_report":       ["fraud","scam","unauthorized","stolen","suspicious","hacked"],
    "refund_request":     ["refund","money back","return","reimburse","credit"],
    "onboarding":         ["new","start","begin","setup","first time","getting started"],
    "feature_request":    ["would be great","could you add","please add","wish","feature"],
    "escalation":         ["manager","escalate","unacceptable","legal","complaint","weeks"],
}


def assign_category(text):
    text_lower = text.lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else np.random.choice(CATEGORIES)


def generate_nlp_dataset(n=3000):
    """Generate realistic customer feedback dataset."""
    records = []
    sentiments = list(SENTIMENT_TEMPLATES.keys())
    weights    = [0.40, 0.35, 0.25]   # positive, negative, neutral

    for i in range(n):
        sentiment = np.random.choice(sentiments, p=weights)
        base_text = np.random.choice(SENTIMENT_TEMPLATES[sentiment])

        # Add some noise to make texts unique
        noise_words = ["Also,", "Additionally,", "Furthermore,", "By the way,", ""]
        prefix = np.random.choice(noise_words)
        text   = f"{prefix} {base_text}".strip() if prefix else base_text

        category = assign_category(text)
        records.append({
            "ticket_id": f"TKT{str(i).zfill(6)}",
            "text":      text,
            "sentiment": sentiment,
            "category":  category,
            "priority":  "HIGH" if sentiment == "negative" else
                         "MEDIUM" if sentiment == "neutral" else "LOW"
        })

    return pd.DataFrame(records)


def clean_text(text):
    """
    Text cleaning pipeline.
    In production: uses NLTK tokenizer, stemmer, and lemmatizer.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)               # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()             # normalize whitespace
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


def build_tfidf_features(texts_train, texts_test):
    """
    TF-IDF feature extraction.
    In production: replaced by BERT embeddings from HuggingFace.
    """
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(texts_train)
    X_test  = vectorizer.transform(texts_test)
    return X_train, X_test, vectorizer


if __name__ == "__main__":
    df = generate_nlp_dataset(3000)
    print(f"Dataset shape : {df.shape}")
    print(f"\nSentiment distribution:\n{df['sentiment'].value_counts()}")
    print(f"\nCategory distribution:\n{df['category'].value_counts()}")
    print(f"\nSample records:")
    print(df[['text','sentiment','category']].head(5).to_string())
