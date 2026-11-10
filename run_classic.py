"""Classic pipeline: TF-IDF -> k-means topics -> VADER sentiment -> charts.

    python run_classic.py --data data/sample_feedback.csv --clusters auto --supervised
"""

import argparse
import json
import os

from feedback_nlp.cluster import choose_k, extract_summaries, kmeans_cluster, reduce_dimensionality, top_terms
from feedback_nlp.preprocess import load_and_clean, tfidf_vectorize
from feedback_nlp.sentiment import (evaluate_vader, likert_agreement, save_model, score_vader,
                                    train_supervised_sentiment)
from feedback_nlp.visualize import boxplot_compare_likert_sentiment, scatter_clusters, stacked_bar_sentiment


def run(data_path, out_dir="outputs", n_clusters="auto", supervised=False, label_col="sentiment_label",
        text_col="free_response"):
    os.makedirs(out_dir, exist_ok=True)
    df = load_and_clean(data_path, text_col=text_col)
    print(f"Loaded {len(df)} non-empty responses.")

    X, vect = tfidf_vectorize(df[text_col])
    if n_clusters == "auto":
        k, sil_by_k = choose_k(X)
        print("Silhouette by k:", {kk: round(v, 3) for kk, v in sil_by_k.items()}, "-> k =", k)
    else:
        k = int(n_clusters)
    labels, km, sil = kmeans_cluster(X, n_clusters=k)
    df["cluster"] = labels
    keywords = top_terms(km, vect)
    summaries = extract_summaries(df, labels, text_col="raw_response")
    print(f"k-means: {k} clusters, silhouette = {sil:.3f}" if sil is not None else f"k-means: {k} clusters")
    for c, info in summaries.items():
        print(f"\nCluster {c} (n={info['n_responses']}): {', '.join(keywords[c])}")
        for ex in info["examples"][:2]:
            print("   -", ex)

    df = score_vader(df)
    rho = likert_agreement(df)
    acc = evaluate_vader(df, label_col)
    print("\nSentiment distribution:", df["vader_label"].value_counts().to_dict())
    if rho is not None:
        print(f"Rank correlation Likert vs VADER: {rho:.3f}")
    if acc is not None:
        print(f"VADER accuracy vs human labels: {acc:.3f}")

    X2, _, _ = reduce_dimensionality(X)
    scatter_clusters(X2, labels, keywords, os.path.join(out_dir, "clusters_tsne.png"))
    stacked_bar_sentiment(df, outpath=os.path.join(out_dir, "stacked_sentiment.png"))
    if "likert_1_5" in df.columns:
        boxplot_compare_likert_sentiment(df, outpath=os.path.join(out_dir, "likert_vs_sentiment.png"))
    df.drop(columns=[text_col]).rename(columns={"raw_response": text_col}).to_csv(
        os.path.join(out_dir, "processed_feedback.csv"), index=False)

    summary = {"n_responses": len(df), "k": k, "silhouette": sil,
               "cluster_keywords": {str(c): v for c, v in keywords.items()},
               "sentiment_counts": df["vader_label"].value_counts().to_dict(),
               "likert_vader_rank_correlation": rho, "vader_accuracy_vs_labels": acc}

    if supervised:
        model, stats = train_supervised_sentiment(df, label_col=label_col)
        save_model(model, os.path.join(out_dir, "supervised_sentiment_model.joblib"))
        summary["supervised_accuracy"] = stats["accuracy"]
        print(f"\nSupervised TF-IDF + LR accuracy: {stats['accuracy']:.3f}\n{stats['report']}")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nOutputs written to {out_dir}/")
    return df, summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/sample_feedback.csv", help="CSV with a free-text column")
    p.add_argument("--text-col", default="free_response")
    p.add_argument("--clusters", default="auto", help="number of clusters, or 'auto' (silhouette)")
    p.add_argument("--supervised", action="store_true", help="also train a classifier on human labels")
    p.add_argument("--labelcol", default="sentiment_label")
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    run(a.data, a.out, a.clusters, a.supervised, a.labelcol, a.text_col)
