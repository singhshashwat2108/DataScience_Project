"""Step 4: independent recomputation of Silhouette, DBI and CH from the SAVED labels, all seeds, all methods.

Deliberately shares no code with gnn_segmentation/: its own preprocessing (pandas/numpy from
unified/cc_full.csv + schema.json), its own metric implementations (numpy/scipy from the formulas of
paper Section 7.6, no sklearn.metrics) and its own tiny-cluster merge. Compares with results_by_seed.csv.
Usage: python audit_lean/check_independent.py [audit_lean]
"""
import json, sys
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist

OUT = sys.argv[1] if len(sys.argv) > 1 else "audit_lean"
df = pd.read_csv("unified/cc_full.csv")
feats = json.load(open("unified/schema.json"))["sources"]["cc"]["numeric"]
LOG = ["cc_balance", "cc_purchases", "cc_oneoff_purchases", "cc_installments_purchases", "cc_cash_advance",
       "cc_cash_advance_trx", "cc_purchases_trx", "cc_credit_limit", "cc_payments", "cc_minimum_payments"]
assert not df[feats].isna().any().any()
tr_mask = (df["split"] == "train").to_numpy()
A = df[feats].astype(float).copy(); A[LOG] = np.log1p(A[LOG])
B = df[feats].astype(float)
space = {name: ((M - M[tr_mask].mean()) / M[tr_mask].std(ddof=0)).to_numpy() for name, M in (("", A), ("std_", B))}


def silhouette(X, y):
    D = cdist(X, X); s = np.zeros(len(y)); ids = np.unique(y)
    for i in range(len(y)):
        own = y == y[i]
        if own.sum() <= 1:
            continue                                   # singleton: s(i) = 0 by convention
        a = D[i, own].sum() / (own.sum() - 1)
        b = min(D[i, y == c].mean() for c in ids if c != y[i])
        s[i] = (b - a) / max(a, b)
    return s.mean()


def dbi(X, y):
    ids = np.unique(y); C = np.stack([X[y == c].mean(0) for c in ids])
    S = np.array([np.linalg.norm(X[y == c] - C[j], axis=1).mean() for j, c in enumerate(ids)])
    M = cdist(C, C); np.fill_diagonal(M, np.inf)
    return float(np.mean([np.max((S[i] + S) / M[i]) for i in range(len(ids))]))   # M[i,i] = inf, so j = i contributes 0


def ch(X, y):
    ids = np.unique(y); n, k = len(y), len(ids); g = X.mean(0)
    Bt = sum((y == c).sum() * np.sum((X[y == c].mean(0) - g) ** 2) for c in ids)
    Wt = sum(np.sum((X[y == c] - X[y == c].mean(0)) ** 2) for c in ids)
    return (Bt / (k - 1)) / (Wt / (n - k))


def indices(X, y):
    if len(np.unique(y)) < 2:
        return dict(silhouette=np.nan, dbi=np.nan, ch=np.nan)
    return dict(silhouette=silhouette(X, y), dbi=dbi(X, y), ch=ch(X, y))


def merge_tiny(X, y, n_total, share=0.005):
    ids, cnt = np.unique(y, return_counts=True)
    tiny, big = ids[cnt < share * n_total], ids[cnt >= share * n_total]
    if len(tiny) == 0 or len(big) == 0:
        return y
    cent = np.stack([X[y == c].mean(0) for c in big]); y2 = y.copy(); m = np.isin(y, tiny)
    y2[m] = big[np.argmin(cdist(X[m], cent), axis=1)]
    return y2


lab = np.load(f"{OUT}/labels_all_seeds.npz")
te = lab["test_idx"]
assert (df["split"].to_numpy()[te] == "test").all() and (df["split"] == "test").sum() == len(te)
rep = pd.read_csv(f"{OUT}/results_by_seed.csv")
rows = []
for _, r in rep.iterrows():
    y_all = lab[f"{r.method}__seed{int(r.seed)}"].astype(int)[te]
    for pre, X in space.items():
        keep = y_all >= 0; Xk, yk = X[te][keep], y_all[keep]
        ym = merge_tiny(Xk, yk, len(te))
        for tag, yy in (("all", yk), ("merged", ym)):
            ind = indices(Xk, yy)
            for met in ("silhouette", "dbi", "ch"):
                rep_val = r[f"{pre}{met}_{tag}"]
                rows.append(dict(method=r.method, seed=int(r.seed), space="common (log1p+std)" if pre == "" else "plain std",
                                 variant=tag, metric=met, reported=rep_val, recomputed=ind[met],
                                 abs_diff=abs(rep_val - ind[met]) if np.isfinite(rep_val) else np.nan,
                                 rel_diff=abs(rep_val - ind[met]) / abs(ind[met]) if np.isfinite(rep_val) and ind[met] else np.nan))
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/independent_check.csv", index=False)
summ = dict(n_values_compared=int(res.abs_diff.notna().sum()), n_nan_both=int((res.reported.isna() & res.recomputed.isna()).sum()),
            n_nan_mismatch=int((res.reported.isna() != res.recomputed.isna()).sum()),
            max_abs_diff=float(res.abs_diff.max()), max_rel_diff=float(res.rel_diff.max()),
            worst_row=res.loc[res.rel_diff.idxmax(), ["method", "seed", "space", "variant", "metric", "reported", "recomputed"]].to_dict(),
            max_rel_diff_by_metric=res.groupby("metric").rel_diff.max().to_dict(),
            max_abs_diff_by_metric=res.groupby("metric").abs_diff.max().to_dict(),
            max_rel_diff_by_method=res.groupby("method").rel_diff.max().to_dict())
json.dump(summ, open(f"{OUT}/independent_check_summary.json", "w"), indent=2, default=str)
print(json.dumps(summ, indent=2, default=str))
