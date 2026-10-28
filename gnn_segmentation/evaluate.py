"""Clustering of representations and the evaluation protocol (Sections 4.5, 4.7, 7.6)."""
import itertools

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def cluster(Z, train_idx, K, method="kmeans", seed=0):
    """Fit on training embeddings, assign every customer with the fitted model."""
    if method == "kmeans":
        m = KMeans(K, n_init=20, random_state=seed).fit(Z[train_idx])
    else:
        m = GaussianMixture(K, covariance_type="full", random_state=seed,
                            reg_covar=1e-4).fit(Z[train_idx])
    return m.predict(Z), m


def common_space_scores(X, labels, sample=None, seed=0):
    """Internal indices in the standardized original feature space."""
    if len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "dbi": np.nan, "ch": np.nan}
    return {"silhouette": silhouette_score(X, labels, sample_size=sample, random_state=seed),
            "dbi": davies_bouldin_score(X, labels),
            "ch": calinski_harabasz_score(X, labels)}


def ref_scores(ref, labels):
    if ref is None:
        return {"ari_ref": np.nan, "nmi_ref": np.nan}
    ok = pd.notna(ref)
    return {"ari_ref": adjusted_rand_score(ref[ok], labels[ok]),
            "nmi_ref": normalized_mutual_info_score(ref[ok], labels[ok])}


def pairwise_ari(partitions):
    """Stability: mean ARI between partitions from different seeds."""
    vals = [adjusted_rand_score(a, b) for a, b in itertools.combinations(partitions, 2)]
    return float(np.mean(vals)), float(np.std(vals))


def evaluate(src, labels, idx):
    out = common_space_scores(src.X[idx], labels[idx])
    out.update(ref_scores(None if src.ref is None else src.ref[idx], labels[idx]))
    out["n_segments_used"] = len(np.unique(labels[idx]))
    out["min_segment_share"] = np.bincount(labels[idx]).min() / len(idx) \
        if labels[idx].min() >= 0 else np.nan
    return out


# ---------------- B0: base-paper pipeline (Kedar et al., 2026) ----------------
def base_paper_pipeline(Xp1, train_idx, K=4, seed=0):
    """K-Means, DBSCAN, hierarchical (Ward) and spectral clustering on P1 features.

    Non-inductive methods are fitted on the training split; other customers get
    the label of their nearest training customer (DBSCAN: nearest core point
    within eps, else noise = -1).
    """
    Xtr = Xp1[train_idx]
    res = {}
    res["B0 K-Means"] = KMeans(K, n_init=20, random_state=seed).fit(Xtr).predict(Xp1)

    lab = AgglomerativeClustering(K, linkage="ward").fit_predict(Xtr)
    res["B0 Hierarchical"] = KNeighborsClassifier(1).fit(Xtr, lab).predict(Xp1)

    lab = SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=10,
                             random_state=seed, assign_labels="cluster_qr").fit_predict(Xtr)
    res["B0 Spectral"] = KNeighborsClassifier(1).fit(Xtr, lab).predict(Xp1)

    # DBSCAN: min_samples = 2*d, eps = 95th pct of the min_samples-NN distance.
    ms = 2 * Xp1.shape[1]
    d, _ = NearestNeighbors(n_neighbors=ms).fit(Xtr).kneighbors(Xtr)
    eps = float(np.quantile(d[:, -1], 0.95))
    db = DBSCAN(eps=eps, min_samples=ms).fit(Xtr)
    core = db.core_sample_indices_
    nn = NearestNeighbors(n_neighbors=1).fit(Xtr[core])
    dist, j = nn.kneighbors(Xp1)
    lab = db.labels_[core][j.ravel()]
    lab[dist.ravel() > eps] = -1
    res["B0 DBSCAN"] = lab
    return res


def b0_scores(space, labels, idx):
    """Indices for B0; DBSCAN noise points are excluded, as is customary."""
    keep = idx[labels[idx] >= 0]
    out = common_space_scores(space[keep], labels[keep])
    out["noise_share"] = 1 - len(keep) / len(idx)
    out["n_segments_used"] = len(np.unique(labels[keep]))
    return out


# ---------------- one tiny-cluster rule for ALL methods ----------------
TINY_SHARE = 0.005   # clusters smaller than 0.5% of the evaluated nodes are "tiny"


def merge_tiny(X, labels, n_total, min_share=TINY_SHARE):
    """Merge customers of tiny clusters into the nearest large cluster.

    X, labels: evaluated customers (noise label -1 already removed); n_total: number of
    evaluated nodes used for the 0.5% threshold. "Nearest" = Euclidean distance to the
    centroid of each large cluster in X (the common space). Returns (new_labels, tiny_sizes).
    """
    ids, cnt = np.unique(labels, return_counts=True)
    tiny = ids[cnt < min_share * n_total]
    big = ids[cnt >= min_share * n_total]
    sizes = {int(c): int(k) for c, k in zip(ids, cnt) if c in tiny}
    if len(tiny) == 0 or len(big) == 0:
        return labels.copy(), sizes
    cent = np.stack([X[labels == c].mean(0) for c in big])
    out = labels.copy()
    m = np.isin(labels, tiny)
    out[m] = big[np.argmin(((X[m][:, None, :] - cent[None]) ** 2).sum(-1), axis=1)]
    return out, sizes


def evaluate_tiny(X, labels, n_total=None, min_share=TINY_SHARE):
    """Indices in the common space X (same for every method), both ways:
    *_all    = every cluster as produced by the method;
    *_merged = after merging tiny clusters (< min_share of n_total) into the nearest large one.
    Noise (label -1, DBSCAN) is excluded from both and reported as noise_share."""
    n_total = len(labels) if n_total is None else n_total
    keep = labels >= 0
    Xk, lk = X[keep], labels[keep]
    lm, tiny = merge_tiny(Xk, lk, n_total, min_share)
    a, m = common_space_scores(Xk, lk), common_space_scores(Xk, lm)
    out = {f"{k}_all": v for k, v in a.items()}
    out.update({f"{k}_merged": v for k, v in m.items()})
    out.update(n_clusters_all=len(np.unique(lk)), n_clusters_merged=len(np.unique(lm)),
               n_tiny_clusters=len(tiny), tiny_sizes=";".join(f"{c}:{s}" for c, s in tiny.items()),
               min_cluster_size=int(np.bincount(lk).min()) if len(lk) else 0,
               noise_share=1 - keep.mean())
    return out
