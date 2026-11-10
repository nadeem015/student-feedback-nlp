"""Embedding pipeline: sentence-transformers -> UMAP -> HDBSCAN topic discovery.

    pip install -r requirements-embeddings.txt
    python run_embeddings.py --data data/sample_feedback.csv
"""

import argparse

import pandas as pd

from feedback_nlp.embeddings import DEFAULT_MODEL, run

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/sample_feedback.csv")
    p.add_argument("--text-col", default="free_response")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="any sentence-transformers model, e.g. all-mpnet-base-v2 for higher quality")
    p.add_argument("--min-cluster-size", type=int, default=None)
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    df, info = run(pd.read_csv(a.data), a.text_col, a.out, a.model, a.min_cluster_size)
    print(f"{len(df)} responses -> {sum(1 for c in info if c != -1)} clusters "
          f"({info.get(-1, {}).get('size', 0)} unclustered)")
    for c, d in info.items():
        print(f"{'noise' if c == -1 else 'cluster ' + str(c)} (n={d['size']}): {', '.join(d['keywords'][:5])}")
    print(f"Outputs written to {a.out}/")
