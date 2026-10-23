"""Quick live demonstration (~1 min on CPU): one dataset, one seed, end to end.

Usage (from the project root):
    .venv/bin/python -W ignore -m gnn_segmentation.demo            # credit card
    .venv/bin/python -W ignore -m gnn_segmentation.demo --source bm # Bank Marketing
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .business import PALETTE
from .data import load_all, prepare_source, SOURCE_NAMES
from .evaluate import cluster, evaluate
from .explain import surrogate_shap
from .gnn import train_gnn
from .graph import knn_graph, to_edge_index, diagnostics


def step(msg):
    print(f"\n=== {msg} ===", flush=True)


def main(source="cc", K=4, k=5, dim=16):
    t0 = time.time()
    step(f"1. Load and preprocess: {SOURCE_NAMES[source]}")
    src = prepare_source(load_all(), source)
    print(f"customers={len(src.df):,}  features={src.features}")
    print(pd.Series(src.split).value_counts().to_dict())

    step(f"2. Build the kNN behavioral similarity graph (cosine, k={k})")
    A = knn_graph(src.X, k)
    d = diagnostics(A, src.ref)
    print({key: (round(v, 4) if isinstance(v, float) else v) for key, v in d.items()})

    step("3. Train GraphSAGE encoder with VGAE objective (no labels)")
    Z, info, _ = train_gnn(src.X, to_edge_index(A), src.idx("train"), "sage", "vgae", dim)
    print(f"embeddings {Z.shape}, validation link-prediction AUC={info['val_auc']:.3f}, "
          f"epochs={info['epochs']}")

    step(f"4. Cluster embeddings into K={K} segments; compare with K-Means on raw features")
    te = src.idx("test")
    gnn_lab, _ = cluster(Z, src.idx("train"), K)
    b1_lab, _ = cluster(src.X, src.idx("train"), K)
    res = pd.DataFrame({"GNN + K-Means": evaluate(src, gnn_lab, te),
                        "B1 K-Means (no graph)": evaluate(src, b1_lab, te)}).T
    print(res[["silhouette", "dbi", "ch", "ari_ref", "min_segment_share"]].round(3)
          .to_string())

    step("5. Segment profiles (feature means)")
    prof = pd.DataFrame(src.X_raw, columns=src.features).assign(segment=gnn_lab) \
        .groupby("segment").agg(["mean"]).droplevel(1, axis=1)
    prof.insert(0, "customers", np.bincount(gnn_lab))
    print(prof.round(2).to_string())

    step("6. Explain: SHAP on a surrogate model")
    sh = surrogate_shap(src, gnn_lab)
    print(f"surrogate fidelity (test macro-F1) = {sh['fidelity_macro_f1']:.3f}")
    print("global feature importance (mean |SHAP|):")
    print(sh["global"].round(3).to_string())

    os.makedirs("results/demo", exist_ok=True)
    P = PCA(2, random_state=0).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(5, 4))
    for c in range(K):
        m = gnn_lab == c
        ax.scatter(P[m, 0], P[m, 1], s=4, color=PALETTE[c], alpha=0.6, linewidths=0,
                   label=f"Segment {c} ({m.sum():,})")
    ax.set(title=f"{SOURCE_NAMES[source]}: GNN embeddings", xticks=[], yticks=[])
    ax.legend(markerscale=3, fontsize=7, frameon=False)
    fig.tight_layout()
    path = f"results/demo/{source}_segments.png"
    fig.savefig(path, dpi=160)
    print(f"\nfigure saved: {path}\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="cc", choices=["cc", "bm", "sa"])
    ap.add_argument("--K", type=int, default=4)
    a = ap.parse_args()
    main(a.source, a.K)
