"""Two static figures for the lean credit-card run (seed-0 GNN segments, test split).
Usage: python audit_lean/make_figures.py [audit_lean]"""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

OUT = sys.argv[1] if len(sys.argv) > 1 else "audit_lean"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEG = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4, fixed order
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

prof = pd.read_csv(f"{OUT}/segment_profiles_gnn_seed0.csv")
KEY = [("cc_balance", "Balance"), ("cc_purchases", "Purchases"), ("cc_cash_advance", "Cash advance"),
       ("cc_credit_limit", "Credit limit"), ("cc_payments", "Payments"), ("cc_purchases_frequency", "Purchases frequency (0-1)"),
       ("cc_cash_advance_frequency", "Cash-advance frequency (0-1)"), ("cc_purchases_trx", "Purchase transactions"),
       ("cc_prc_full_payment", "Share of full payment (0-1)"), ("cc_tenure", "Tenure (months)")]
seg = prof["segment"].astype(int).tolist()
labels = [f"Segment {s} (n={int(n)})" for s, n in zip(seg, prof["n_test"])]

fig, axs = plt.subplots(2, 5, figsize=(14, 5.6))
for ax, (col, title) in zip(axs.ravel(), KEY):
    ax.bar(range(len(seg)), prof[col], color=[SEG[s] for s in seg], width=0.6)
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.set_xticks(range(len(seg))); ax.set_xticklabels([str(s) for s in seg])
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(length=0)
fig.supxlabel("GNN segment (K-Means, K = 4, seed 0); test customers; means in original units", color=INK2, fontsize=9)
handles = [plt.Rectangle((0, 0), 1, 1, color=SEG[s]) for s in seg]
fig.legend(handles, labels, loc="upper center", ncol=len(seg), frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0.02, 1, 0.94))
fig.savefig(f"{OUT}/fig1_segment_profiles.png", dpi=150); plt.close(fig)

Z = np.load(f"{OUT}/embedding_seed0_gnn.npy")
lab = np.load(f"{OUT}/labels_all_seeds.npz")
te = lab["test_idx"]; y = lab["GNN VGAE+K-Means__seed0"].astype(int)
P = PCA(2, random_state=0).fit(Z[lab["train_idx"]])
Q = P.transform(Z[te])
fig, ax = plt.subplots(figsize=(6.4, 5.2))
for s in sorted(set(y[te])):
    m = y[te] == s
    ax.scatter(Q[m, 0], Q[m, 1], s=9, color=SEG[s], alpha=0.75, linewidths=0, label=f"Segment {s} (n={m.sum()})")
ax.set_xlabel(f"PC1 of GNN embedding ({P.explained_variance_ratio_[0]:.0%} of variance)")
ax.set_ylabel(f"PC2 ({P.explained_variance_ratio_[1]:.0%})")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.set_title("GNN embedding (2-D PCA fitted on train), test customers, seed 0", fontsize=9.5, loc="left")
ax.legend(frameon=False, markerscale=2.2, loc="best")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_embedding_scatter.png", dpi=150); plt.close(fig)
print("figures written")
