"""Sentiment: VADER lexicon scores plus an optional supervised baseline."""

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def label_from_compound(x: float) -> str:
    """Standard VADER thresholds."""
    if x >= 0.05:
        return "positive"
    if x <= -0.05:
        return "negative"
    return "neutral"


def score_vader(df, text_col="raw_response"):
    """Add vader_compound and vader_label columns. Uses the original text when
    available, because VADER relies on punctuation and capitalisation."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    col = text_col if text_col in df.columns else "free_response"
    sia = SentimentIntensityAnalyzer()
    df = df.copy()
    df["vader_compound"] = df[col].fillna("").astype(str).apply(lambda t: sia.polarity_scores(t)["compound"])
    df["vader_label"] = df["vader_compound"].apply(label_from_compound)
    return df


def likert_agreement(df, likert_col="likert_1_5", sentiment_col="vader_compound"):
    """Spearman-style rank correlation between Likert ratings and VADER scores."""
    d = df[[likert_col, sentiment_col]].dropna()
    if len(d) < 3:
        return None
    return float(d[likert_col].rank().corr(d[sentiment_col].rank()))


def evaluate_vader(df, label_col="sentiment_label"):
    """Accuracy of VADER labels against human labels, if present."""
    if label_col not in df.columns:
        return None
    d = df.dropna(subset=[label_col])
    return float(np.mean(d[label_col].astype(str) == d["vader_label"])) if len(d) else None


def train_supervised_sentiment(df, text_col="free_response", label_col="sentiment_label",
                               test_size=0.25, random_state=42):
    """TF-IDF + logistic regression trained on human labels."""
    d = df.dropna(subset=[label_col])
    if d.empty:
        raise ValueError(f"No labelled rows: add a '{label_col}' column with human labels.")
    X, y = d[text_col].astype(str), d[label_col].astype(str)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                              random_state=random_state, stratify=y)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return model, {"accuracy": accuracy_score(y_te, pred),
                   "report": classification_report(y_te, pred, digits=3, zero_division=0)}


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
