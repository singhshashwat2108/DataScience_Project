"""Two-level explainability layer (paper Sections 4.6, 7.5).

(i) Feature level: gradient-boosted surrogate predicting the segment from the
    original features, explained with SHAP; fidelity = test macro-F1.
(ii) Neighbor level: GNNExplainer on the trained encoder, with the segment
    (nearest K-Means centroid in embedding space) as the prediction target.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from .gnn import embed

warnings.filterwarnings("ignore", category=UserWarning)


def surrogate_shap(src, labels, seed=0, n_explain=2000):
    tr, te = src.idx("train"), src.idx("test")
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             random_state=seed, verbose=-1)
    Xr = pd.DataFrame(src.X_raw, columns=src.features)
    clf.fit(Xr.iloc[tr], labels[tr])
    f1 = f1_score(labels[te], clf.predict(Xr.iloc[te]), average="macro")

    rng = np.random.default_rng(seed)
    sel = rng.choice(te, size=min(n_explain, len(te)), replace=False)
    sv = shap.TreeExplainer(clf).shap_values(Xr.iloc[sel])
    sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)
    if sv.ndim == 2:  # binary case returns one array for the positive class
        sv = np.stack([-sv, sv], -1)
    K = sv.shape[-1]
    seg = labels[sel]
    # Per-segment drivers: mean |SHAP| of the segment's own class output,
    # over customers in that segment; sign = mean signed SHAP.
    rows = []
    for c in range(K):
        m = seg == c
        if not m.any():
            continue
        for j, f in enumerate(src.features):
            rows.append({"segment": c, "feature": f,
                         "mean_abs_shap": np.abs(sv[m, j, c]).mean(),
                         "mean_shap": sv[m, j, c].mean()})
    per_seg = pd.DataFrame(rows)
    glob = pd.Series(np.abs(sv).mean((0, 2)), index=src.features).sort_values(ascending=False)
    return {"fidelity_macro_f1": f1, "per_segment": per_seg, "global": glob,
            "shap_values": sv, "X_explained": Xr.iloc[sel], "model": clf}


class SegmentModel(nn.Module):
    """Encoder + fixed centroids: logits = -||z - mu_c||^2 (K-Means assignment)."""

    def __init__(self, gnn_model, objective, centroids):
        super().__init__()
        self.m, self.objective = gnn_model, objective
        self.register_buffer("C", torch.tensor(centroids, dtype=torch.float32))

    def forward(self, x, edge_index):
        z = embed(self.m, self.objective, x, edge_index)
        return -torch.cdist(z, self.C) ** 2


def gnn_explain(src, model, objective, centroids, edge_index, labels,
                per_segment=15, top=5, epochs=100, seed=0):
    torch.manual_seed(seed)
    seg_model = SegmentModel(model, objective, centroids).eval()
    x = torch.tensor(src.X, dtype=torch.float32)
    explainer = Explainer(
        model=seg_model, algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model", node_mask_type="attributes", edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node",
                          return_type="raw"))
    rng = np.random.default_rng(seed)
    te = src.idx("test")
    rows, feat_rows = [], []
    for c in np.unique(labels):
        cand = te[labels[te] == c]
        for i in rng.choice(cand, size=min(per_segment, len(cand)), replace=False):
            subset, sub_ei, mapping, _ = k_hop_subgraph(int(i), 2, edge_index,
                                                        relabel_nodes=True)
            ex = explainer(x[subset], sub_ei, index=int(mapping))
            em = ex.edge_mask.detach().numpy()
            # Direct neighbours j -> i and their edge importance.
            into = (sub_ei[1] == mapping).numpy()
            nbr = subset[sub_ei[0][into]].numpy()
            imp = em[into]
            if len(nbr) == 0:
                continue
            order = np.argsort(-imp)[:top]
            same_top = (labels[nbr[order]] == c).mean()
            same_all = (labels[nbr] == c).mean()
            rows.append({"segment": int(c), "node": int(i), "n_neighbors": len(nbr),
                         "top_same_segment_share": same_top,
                         "all_same_segment_share": same_all,
                         "top_edge_mask_mean": imp[order].mean(),
                         "other_edge_mask_mean": np.delete(imp, order).mean()
                         if len(imp) > top else np.nan})
            fm = ex.node_mask.detach().numpy()[int(mapping)]
            feat_rows.append({"segment": int(c), **dict(zip(src.features, fm))})
    return pd.DataFrame(rows), pd.DataFrame(feat_rows)
