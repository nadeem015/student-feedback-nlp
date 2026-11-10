"""Embedding pipeline: sentence-transformers -> UMAP -> HDBSCAN.

Unlike TF-IDF, sentence embeddings place responses with the same meaning
close together even when they share no words ("the TA was great" and
"office hours really helped"). HDBSCAN then finds clusters of any shape and
marks outliers as -1 instead of forcing every response into a topic.

Heavy dependencies are imported inside the functions so that the rest of the
package (and its tests) work without them.
"""

import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import clean_text

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def embed(texts, model_name=DEFAULT_MODEL, batch_size=64):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(list(texts), batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)


def reduce(embeddings, n_components=10, n_neighbors=15, random_state=42):
    import umap

    n_neighbors = max(2, min(n_neighbors, len(embeddings) - 1))
    n_components = max(2, min(n_components, len(embeddings) - 2))
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        metric="cosine", random_state=random_state)
    return reducer.fit_transform(embeddings)


def cluster(points, min_cluster_size=None):
    """HDBSCAN from scikit-learn (>= 1.3), so no separate hdbscan package is needed."""
    from sklearn.cluster import HDBSCAN

    if min_cluster_size is None:
        min_cluster_size = max(3, int(len(points) * 0.05))
    return HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(points)


def cluster_keywords_and_examples(texts, embeddings, labels, top_n_terms=8, n_examples=3):
    """Top TF-IDF terms per cluster and the responses closest to its centroid."""
    texts = list(texts)
    labels = np.asarray(labels)
    out = {}
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        sub = [texts[i] for i in idx]
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            X = vec.fit_transform(sub)
            weights = np.asarray(X.mean(axis=0)).ravel()
            terms = np.array(vec.get_feature_names_out())[weights.argsort()[::-1][:top_n_terms]].tolist()
        except ValueError:        # only stop words in this cluster
            terms = []
        centroid = embeddings[idx].mean(axis=0)
        nearest = idx[np.argsort(np.linalg.norm(embeddings[idx] - centroid, axis=1))[:n_examples]]
        out[int(c)] = {"size": int(len(idx)), "keywords": terms, "examples": [texts[i] for i in nearest]}
    return out


def plot_clusters(points_2d, labels, outpath):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(set(labels.tolist())):
        m = labels == c
        ax.scatter(points_2d[m, 0], points_2d[m, 1], s=40, alpha=0.8,
                   label="noise" if c == -1 else f"cluster {c}", c="lightgrey" if c == -1 else None)
    ax.set_title("Responses by meaning (sentence embeddings, UMAP, HDBSCAN)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def run(df, text_col="free_response", out_dir="outputs", model_name=DEFAULT_MODEL, min_cluster_size=None):
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()
    df["_clean"] = df[text_col].apply(clean_text)
    df = df[df["_clean"] != ""].reset_index(drop=True)
    originals = df[text_col].astype(str).tolist()

    emb = embed(originals, model_name)
    labels = cluster(reduce(emb), min_cluster_size)
    df["embedding_cluster"] = labels
    info = cluster_keywords_and_examples(df["_clean"], emb, labels)
    plot_clusters(reduce(emb, n_components=2), labels, os.path.join(out_dir, "embedding_clusters.png"))

    with open(os.path.join(out_dir, "embedding_cluster_summary.txt"), "w", encoding="utf-8") as f:
        for c, d in info.items():
            name = "Noise (no clear topic)" if c == -1 else f"Cluster {c}"
            f.write(f"{name}  (size={d['size']})\n  Top terms: {', '.join(d['keywords'])}\n  Examples:\n")
            for ex in d["examples"]:
                f.write(f"   - {ex}\n")
            f.write("\n")
    df.drop(columns=["_clean"]).to_csv(os.path.join(out_dir, "responses_with_embedding_clusters.csv"), index=False)
    return df, info
