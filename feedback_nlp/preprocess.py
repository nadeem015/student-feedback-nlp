"""Text cleaning and TF-IDF vectorisation."""

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

EMPTY_ANSWERS = {"n/a", "na", "none", "nothing", "-", "no comment", "no comments"}


def clean_text(text) -> str:
    """Lower-case, keep letters/digits/spaces, collapse whitespace.
    Returns '' for missing values and placeholder answers such as 'N/A'."""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    if text in EMPTY_ANSWERS:
        return ""
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_and_clean(csv_path: str, text_col: str = "free_response", drop_empty: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"Expected text column '{text_col}' in CSV. Columns: {df.columns.tolist()}")
    df = df.copy()
    df["raw_response"] = df[text_col].fillna("").astype(str)
    df[text_col] = df["raw_response"].apply(clean_text)
    if drop_empty:
        df = df[df[text_col] != ""].reset_index(drop=True)
    return df


def tfidf_vectorize(texts, max_df=0.95, min_df=1, max_features=4000, stop_words="english", ngram_range=(1, 2)):
    vect = TfidfVectorizer(stop_words=stop_words, max_df=max_df, min_df=min_df,
                           max_features=max_features, ngram_range=ngram_range)
    X = vect.fit_transform(texts)
    return X, vect
