"""Algorithm 1: kNN behavioral similarity graph (paper Sections 4.3, 7.1)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


def knn_graph(X, k=10, mode="union", weighted=False):
    """Cosine kNN graph, symmetrized by union (default) or mutual-kNN."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    # On unit vectors Euclidean order equals cosine order; this allows a tree index.
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(Xn)
    dist, ind = nn.kneighbors(Xn)
    n = X.shape[0]
    rows = np.repeat(np.arange(n), k)
    # Drop self (first neighbour normally) – handle duplicates by masking i==j.
    ind, dist = ind[:, 1:], dist[:, 1:]
    cols = ind.ravel()
    vals = (1 - dist.ravel() ** 2 / 2) if weighted else np.ones(n * k)
    keep = rows != cols
    A = sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n))
    A = A.maximum(A.T) if mode == "union" else A.minimum(A.T)
    A.eliminate_zeros()
    return A.tocsr()


def to_edge_index(A):
    A = A.tocoo()
    return torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)


def diagnostics(A, labels=None):
    n = A.shape[0]
    deg = np.asarray((A > 0).sum(1)).ravel()
    ncomp, _ = connected_components(A, directed=False)
    out = {"nodes": n, "edges": int(A.nnz // 2), "density": A.nnz / (n * (n - 1)),
           "deg_mean": float(deg.mean()), "deg_min": int(deg.min()), "deg_max": int(deg.max()),
           "components": int(ncomp)}
    if labels is not None:
        A = A.tocoo()
        lab = pd.Series(labels).astype(object)
        ok = (lab.notna().values[A.row]) & (lab.notna().values[A.col])
        lv = lab.values
        out["label_homophily"] = float((lv[A.row][ok] == lv[A.col][ok]).mean())
        # Expected homophily for random edges (sum of squared class shares).
        share = lab.dropna().value_counts(normalize=True).values
        out["random_homophily"] = float((share ** 2).sum())
    return out
