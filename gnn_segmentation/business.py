"""Business analysis and visualization layer (paper Section 4.8).

Segment assignments are stored in SQLite; segment profiles are produced with
SQL queries and exported as tables, and summary figures are drawn for managers.
"""
import os
import pickle
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .data import SOURCE_NAMES

# Validated categorical palette (fixed order, light surface).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})
FEATURES = ["age", "tenure_months", "balance", "n_products", "is_active", "has_credit_card",
            "has_loan", "has_mortgage", "has_savings_deposit"]


def write_segments(src, labels, name, out):
    con = sqlite3.connect(os.path.join(out, "segments.db"))
    df = src.df[["uid", "source", "split", "ref_label"] + FEATURES].copy()
    df["segment"] = labels.astype(int)
    con.execute("CREATE TABLE IF NOT EXISTS segments (uid TEXT PRIMARY KEY, source TEXT, "
                "split TEXT, ref_label TEXT, " + ", ".join(f"{c} REAL" for c in FEATURES)
                + ", segment INTEGER)")
    con.execute("DELETE FROM segments WHERE source = ?", (name,))
    df.to_sql("segments", con, if_exists="append", index=False)
    con.commit()
    con.close()


PROFILE_SQL = """
SELECT source, segment,
       COUNT(*) AS customers,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY source), 1) AS share_pct,
       {avgs}
FROM segments
GROUP BY source, segment
ORDER BY source, segment;
"""
REF_SQL = """
SELECT source, segment, ref_label, COUNT(*) AS customers
FROM segments WHERE ref_label IS NOT NULL
GROUP BY source, segment, ref_label ORDER BY source, segment, ref_label;
"""


def run_profiles(out):
    con = sqlite3.connect(os.path.join(out, "segments.db"))
    avgs = ",\n       ".join(f"ROUND(AVG({c}), 3) AS avg_{c}" for c in FEATURES)
    sql = PROFILE_SQL.format(avgs=avgs)
    prof = pd.read_sql(sql, con)
    ref = pd.read_sql(REF_SQL, con)
    con.close()
    with open(os.path.join(out, "segment_profiles.sql"), "w") as f:
        f.write("-- Segment profile query\n" + sql + "\n-- Segment x reference label\n" + REF_SQL)
    prof.to_csv(os.path.join(out, "segment_profiles.csv"), index=False)
    ref.to_csv(os.path.join(out, "segment_vs_reference.csv"), index=False)
    return prof, ref


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", name), dpi=160)
    plt.close(fig)


def profiles_and_figures(out, full):
    prof, _ = run_profiles(out)
    summ = pd.read_csv(os.path.join(out, "e2_comparison_summary.csv"))
    sens = pd.read_csv(os.path.join(out, "e3_sensitivity.csv"))
    diag = pd.read_csv(os.path.join(out, "graph_diagnostics.csv"))
    shap_seg = pd.read_csv(os.path.join(out, "e5_shap_per_segment.csv"))
    gx = pd.read_csv(os.path.join(out, "e5_gnnexplainer_nodes.csv"))
    state = pickle.load(open(os.path.join(out, "state.pkl"), "rb"))
    sources = list(state)

    # 1. Method comparison at K* (common-space indices, test split, mean ± sd over seeds).
    fig, axes = plt.subplots(len(sources), 3, figsize=(12, 3.0 * len(sources)))
    for r, s in enumerate(sources):
        g = summ[(summ.source == s) & (summ.K == state[s]["K_star"])]
        for c, (m, lab) in enumerate([("silhouette", "Silhouette (higher better)"),
                                      ("dbi", "Davies–Bouldin (lower better)"),
                                      ("ch", "Calinski–Harabasz (higher better)")]):
            ax = axes[r, c]
            colors = [PALETTE[0] if x.startswith("GNN") else "#a3a29b" for x in g.method]
            ax.barh(g.method, g[m], xerr=g[m + "_sd"], color=colors, height=0.6,
                    error_kw=dict(ecolor=INK2, lw=0.8))
            ax.invert_yaxis()
            ax.set_title(f"{SOURCE_NAMES[s]} (K={state[s]['K_star']}): {lab}", loc="left")
            if c:
                ax.set_yticklabels([])
    _save(fig, out, "comparison_indices.png")

    # 2. Sensitivity to k (credit card): validation silhouette and graph connectivity.
    cc = sens[(sens.source == "cc")]
    cfg = state["cc"]["cfg"]
    ks = cc[(cc.kind == cfg["kind"]) & (cc.objective == cfg["objective"]) & (cc.dim == 32)] \
        .drop_duplicates("k").sort_values("k")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(ks.k, ks.val_silhouette_K4, color=PALETTE[0], lw=2, marker="o", ms=6)
    axes[0].set(xlabel="neighbours k", title="Credit card: validation silhouette (K=4)")
    d = diag[diag.source == "cc"].sort_values("k")
    axes[1].plot(d.k, d.components, color=PALETTE[0], lw=2, marker="o", ms=6, label="union")
    axes[1].plot(d.k, d.mutual_components, color=PALETTE[1], lw=2, marker="o", ms=6,
                 label="mutual kNN")
    axes[1].set(xlabel="neighbours k", ylabel="connected components",
                title="Credit card: graph connectivity", yscale="log")
    axes[1].legend()
    _save(fig, out, "sensitivity_k.png")

    # 3. Embeddings (2-D PCA) coloured by segment.
    fig, axes = plt.subplots(1, len(sources), figsize=(4.2 * len(sources), 3.8))
    for ax, s in zip(axes, sources):
        Z, lab = state[s]["Z"], state[s]["labels"]["GNN + K-Means"]
        P = PCA(2, random_state=0).fit_transform(Z)
        idx = np.random.default_rng(0).choice(len(P), min(6000, len(P)), replace=False)
        for c in np.unique(lab):
            m = idx[lab[idx] == c]
            ax.scatter(P[m, 0], P[m, 1], s=4, color=PALETTE[c % 8], alpha=0.6,
                       label=f"Segment {c}", linewidths=0)
        ax.set_title(f"{SOURCE_NAMES[s]}: GNN embeddings", loc="left")
        ax.set_xticks([]), ax.set_yticks([])
        ax.legend(markerscale=3, fontsize=7, loc="best")
    _save(fig, out, "embeddings_pca.png")

    # 4. Segment profiles: standardized feature means per segment.
    fig, axes = plt.subplots(1, len(sources), figsize=(4.6 * len(sources), 3.6))
    for ax, s in zip(axes, sources):
        p = prof[prof.source == s].set_index("segment")
        cols = [f"avg_{f}" for f in state[s]["features"]]
        M = p[cols]
        Zs = (M - M.mean()) / (M.std() + 1e-9)
        lim = np.nanmax(np.abs(Zs.values))
        im = ax.imshow(Zs.values.T, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_yticks(range(len(cols)), [c[4:] for c in cols])
        ax.set_xticks(range(len(M)), [f"S{i}\n{n:,}" for i, n in zip(M.index, p.customers)])
        for (i, j), v in np.ndenumerate(M.values.T):
            ax.text(j, i, f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}", ha="center",
                    va="center", fontsize=6.5, color=INK)
        ax.grid(False)
        ax.set_title(f"{SOURCE_NAMES[s]}: segment means", loc="left")
    fig.colorbar(im, ax=axes, shrink=0.7, label="z-score across segments")
    fig.savefig(os.path.join(out, "figures", "segment_profiles.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    # 5. SHAP drivers per segment.
    fig, axes = plt.subplots(1, len(sources), figsize=(4.4 * len(sources), 3.6))
    for ax, s in zip(axes, sources):
        t = shap_seg[shap_seg.source == s].pivot(index="feature", columns="segment",
                                                 values="mean_abs_shap")
        t = t.loc[t.mean(1).sort_values().index]
        segs = list(t.columns)
        h = 0.8 / len(segs)
        for i, c in enumerate(segs):
            ax.barh(np.arange(len(t)) + i * h, t[c], height=h * 0.9, color=PALETTE[c % 8],
                    label=f"Segment {c}")
        ax.set_yticks(np.arange(len(t)) + 0.4 - h / 2, t.index)
        ax.set(xlabel="mean |SHAP| (segment's own output)")
        ax.set_title(f"{SOURCE_NAMES[s]}: feature drivers", loc="left")
        ax.legend(fontsize=7)
    _save(fig, out, "shap_drivers.png")

    # 6. GNNExplainer: are the most influential neighbours in the same segment?
    g = gx.groupby("source")[["top_same_segment_share", "all_same_segment_share"]].mean() \
        .reindex(sources)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    y = np.arange(len(g))
    ax.barh(y - 0.2, g.top_same_segment_share, 0.36, color=PALETTE[0],
            label="Top-5 neighbours by GNNExplainer")
    ax.barh(y + 0.2, g.all_same_segment_share, 0.36, color=PALETTE[1], label="All neighbours")
    ax.set_yticks(y, [SOURCE_NAMES[s] for s in g.index])
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="share of neighbours in the customer's own segment")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Neighbour-level explanations", loc="left")
    _save(fig, out, "gnnexplainer_neighbours.png")
    return prof
