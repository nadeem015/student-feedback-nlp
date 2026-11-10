"""Topic clustering: PCA + t-SNE for display, k-means on TF-IDF for clusters."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def reduce_dimensionality(X, pca_components=10, tsne_perplexity=30, tsne_iter=1000, random_state=42):
    """Project TF-IDF vectors to 2-D (PCA first, then t-SNE) for plotting."""
    X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    n_comp = max(2, min(pca_components, X_arr.shape[1], X_arr.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=random_state)
    Xp = pca.fit_transform(X_arr)
    perplexity = float(min(tsne_perplexity, max(2, (Xp.shape[0] - 1) // 3)))
    tsne_kwargs = dict(n_components=2, perplexity=perplexity, learning_rate="auto",
                       init="pca", random_state=random_state)
    try:
        tsne = TSNE(max_iter=tsne_iter, **tsne_kwargs)          # scikit-learn >= 1.5
    except TypeError:
        tsne = TSNE(n_iter=tsne_iter, **tsne_kwargs)            # older scikit-learn
    return tsne.fit_transform(Xp), pca, tsne


def kmeans_cluster(X, n_clusters=5, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    n_found = len(set(labels))
    sil = silhouette_score(X, labels) if 1 < n_found < X.shape[0] else None
    return labels, km, sil


def choose_k(X, k_min=2, k_max=8, random_state=42):
    """Pick the number of clusters with the highest silhouette score."""
    k_max = min(k_max, X.shape[0] - 1)
    scores = {}
    for k in range(k_min, k_max + 1):
        _, _, sil = kmeans_cluster(X, k, random_state)
        if sil is not None:
            scores[k] = sil
    if not scores:
        return k_min, scores
    return max(scores, key=scores.get), scores


def top_terms(km, vectorizer, n_terms=6):
    """Highest-weighted TF-IDF terms of each k-means centroid."""
    terms = np.array(vectorizer.get_feature_names_out())
    order = km.cluster_centers_.argsort()[:, ::-1]
    return {c: terms[order[c, :n_terms]].tolist() for c in range(km.n_clusters)}


def extract_summaries(df, labels, text_col="free_response", top_k=5):
    """Representative responses per cluster: the ones whose words are most
    frequent within that cluster (a simple frequency-based summary)."""
    out = {}
    df = df.copy()
    df["cluster"] = labels
    for c in sorted(df["cluster"].unique()):
        texts = df.loc[df["cluster"] == c, text_col].astype(str).tolist()
        freq = {}
        for w in " ".join(texts).split():
            freq[w] = freq.get(w, 0) + 1

        def score(t):
            words = t.split()
            return sum(freq.get(w, 0) for w in words) / max(1, len(words))

        ranked = sorted(texts, key=score, reverse=True)
        out[int(c)] = {"n_responses": len(texts), "examples": ranked[:top_k]}
    return out
