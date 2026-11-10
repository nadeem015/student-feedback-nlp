"""Plots for the classic pipeline (saved as PNG)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SENTIMENT_COLOURS = {"negative": "#d62728", "neutral": "#9e9e9e", "positive": "#2ca02c"}


def _save(fig, outpath):
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def scatter_clusters(X2, labels, keywords=None, outpath="clusters_tsne.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = np.asarray(labels)
    for c in np.unique(labels):
        pts = X2[labels == c]
        name = f"{c}: " + ", ".join(keywords[c][:3]) if keywords and c in keywords else str(c)
        ax.scatter(pts[:, 0], pts[:, 1], s=40, alpha=0.75, label=name)
    ax.set_title("Response clusters (k-means on TF-IDF, t-SNE projection)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="cluster: top terms", fontsize=8, loc="best")
    return _save(fig, outpath)


def stacked_bar_sentiment(df, section_col="section", label_col="vader_label", outpath="stacked_sentiment.png"):
    order = ["negative", "neutral", "positive"]
    if section_col in df.columns:
        table = df.groupby([section_col, label_col]).size().unstack(fill_value=0)
    else:
        table = df[label_col].value_counts().to_frame("all").T
    table = table.reindex(columns=[c for c in order if c in table.columns], fill_value=0)
    pct = table.div(table.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    pct.plot(kind="bar", stacked=True, ax=ax, color=[SENTIMENT_COLOURS[c] for c in pct.columns])
    ax.set_ylabel("% of responses")
    ax.set_xlabel("")
    ax.set_title("Sentiment of free responses by survey section (VADER)")
    ax.legend(title="sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=0)
    return _save(fig, outpath)


def boxplot_compare_likert_sentiment(df, likert_col="likert_1_5", sentiment_col="vader_compound",
                                     outpath="likert_vs_sentiment.png"):
    """VADER compound score for each Likert rating (1-5)."""
    d = df.dropna(subset=[likert_col, sentiment_col])
    ratings = sorted(int(r) for r in d[likert_col].unique())
    data = [d.loc[d[likert_col] == r, sentiment_col].values for r in ratings]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data)
    ax.set_xticks(range(1, len(ratings) + 1))
    ax.set_xticklabels([str(r) for r in ratings])
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Likert rating (1 = strongly disagree, 5 = strongly agree)")
    ax.set_ylabel("VADER compound score")
    ax.set_title("Do written comments agree with numeric ratings?")
    return _save(fig, outpath)
