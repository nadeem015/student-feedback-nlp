import os

import numpy as np
import pandas as pd
import pytest

from feedback_nlp.cluster import choose_k, extract_summaries, kmeans_cluster, reduce_dimensionality, top_terms
from feedback_nlp.embeddings import cluster, cluster_keywords_and_examples
from feedback_nlp.preprocess import clean_text, load_and_clean, tfidf_vectorize
from feedback_nlp.sentiment import label_from_compound, likert_agreement, train_supervised_sentiment

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "sample_feedback.csv")


@pytest.fixture(scope="module")
def df():
    return load_and_clean(DATA)


def test_clean_text():
    assert clean_text("  Great   TA!! 10/10 ") == "great ta 10 10"
    assert clean_text("N/A") == ""
    assert clean_text(None) == ""


def test_load_drops_empty_answers(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame({"free_response": ["Good course", "n/a", None, "Too fast"]}).to_csv(p, index=False)
    out = load_and_clean(str(p))
    assert out["free_response"].tolist() == ["good course", "too fast"]
    assert out["raw_response"].tolist() == ["Good course", "Too fast"]


def test_load_requires_text_column(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame({"comment": ["x"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_and_clean(str(p))


def test_sample_data_shape(df):
    assert len(df) == 60
    assert set(df["sentiment_label"]) == {"positive", "neutral", "negative"}


def test_kmeans_and_keywords(df):
    X, vect = tfidf_vectorize(df["free_response"])
    labels, km, sil = kmeans_cluster(X, 4)
    assert len(labels) == len(df) and len(set(labels)) == 4
    assert -1 <= sil <= 1
    kw = top_terms(km, vect)
    assert len(kw) == 4 and all(len(v) == 6 for v in kw.values())
    summ = extract_summaries(df, labels)
    assert sum(v["n_responses"] for v in summ.values()) == len(df)


def test_choose_k_returns_best_silhouette(df):
    X, _ = tfidf_vectorize(df["free_response"])
    k, scores = choose_k(X, 2, 6)
    assert k == max(scores, key=scores.get)


def test_tsne_projection(df):
    X, _ = tfidf_vectorize(df["free_response"])
    X2, _, _ = reduce_dimensionality(X, tsne_iter=250)
    assert X2.shape == (len(df), 2)


@pytest.mark.parametrize("score,label", [(0.6, "positive"), (0.05, "positive"), (0.0, "neutral"),
                                         (-0.05, "negative"), (-0.9, "negative")])
def test_vader_thresholds(score, label):
    assert label_from_compound(score) == label


def test_likert_agreement_perfect():
    d = pd.DataFrame({"likert_1_5": [1, 2, 3, 4, 5], "vader_compound": [-0.9, -0.2, 0.0, 0.3, 0.8]})
    assert likert_agreement(d) == pytest.approx(1.0)


def test_supervised_baseline(df):
    model, stats = train_supervised_sentiment(df)
    assert 0 <= stats["accuracy"] <= 1
    assert model.predict(["the tutors were very helpful"])[0] in {"positive", "neutral", "negative"}


def test_vader_and_plots(df, tmp_path):
    pytest.importorskip("vaderSentiment")
    from feedback_nlp.sentiment import score_vader
    from run_classic import run

    scored = score_vader(df)
    assert scored["vader_compound"].between(-1, 1).all()
    _, summary = run(DATA, out_dir=str(tmp_path), n_clusters="auto", supervised=True)
    for f in ["clusters_tsne.png", "stacked_sentiment.png", "likert_vs_sentiment.png",
              "processed_feedback.csv", "summary.json"]:
        assert (tmp_path / f).exists()
    assert summary["likert_vader_rank_correlation"] > 0   # comments agree with ratings


def test_embedding_cluster_summary_without_models():
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(0, 0.05, (8, 5)) + 1, rng.normal(0, 0.05, (8, 5)) - 1])
    texts = ["helpful tutor support"] * 8 + ["homework too long"] * 8
    labels = cluster(emb, min_cluster_size=4)
    assert len(set(labels) - {-1}) == 2
    info = cluster_keywords_and_examples(texts, emb, labels)
    assert sum(d["size"] for d in info.values()) == 16
