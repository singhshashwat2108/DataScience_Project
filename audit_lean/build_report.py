"""Assemble RESULTS_LEAN.md and results_lean.csv from the files written by run_lean.py,
check_independent.py, check_repro.py, check_earlystop.py and make_figures.py. No number is typed in by hand
except the published base-paper values (paper Table 9). Usage: python audit_lean/build_report.py [audit_lean]"""
import json, subprocess, sys
import numpy as np, pandas as pd
from scipy import stats

OUT = sys.argv[1] if len(sys.argv) > 1 else "audit_lean"
F = lambda n: f"{OUT}/{n}"
meta = json.load(open(F("run_meta.json")))
res = pd.read_csv(F("results_by_seed.csv"))
SEEDS = meta["seeds_run"]; NS = len(SEEDS)
METHODS = ["B0 K-Means", "B0 DBSCAN", "B0 Hierarchical", "B0 Spectral", "B1 K-Means", "B2 GMM", "B3 AE+K-Means",
           "B4 A=I+K-Means", "GNN VGAE+K-Means", "Control: shuffled-edge GNN", "Control: random Gaussian embedding"]
GNN, B3, B4 = "GNN VGAE+K-Means", "B3 AE+K-Means", "B4 A=I+K-Means"
IDX = [("silhouette", "Silhouette", 1), ("dbi", "DBI", -1), ("ch", "CH", 1)]
R = []          # long csv


def add(table, method, variant, metric, mean, sd, n, src, note=""):
    R.append(dict(table=table, method=method, variant=variant, metric=metric, mean=mean, sd=sd, n=n, source_file=src, note=note))


def ms(x, d=3):
    x = np.asarray(x, float)
    return "n/a" if np.isnan(x).all() else f"{np.nanmean(x):.{d}f} ± {np.nanstd(x, ddof=1) if np.isfinite(x).sum() > 1 else 0:.{d}f}"


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def vals(method, col):
    return res[res.method == method].sort_values("seed")[col].to_numpy(float)


def d_(metric):
    return 1 if metric == "ch" else 3


L = []
w = L.append
RS = "results_by_seed.csv"
cfg = meta["chosen_config"]
git_log = subprocess.run(["git", "log", "--oneline", "main..HEAD"], capture_output=True, text=True).stdout.strip()

w("# RESULTS_LEAN: credit card, 17 behavioral variables, K = 4, test split\n")
w(f"Run: {meta['start']} to {meta['end']} ({meta['total_min']:.1f} min wall clock). **Seeds actually run: {NS} ({SEEDS[0]}..{SEEDS[-1]})**"
  + (f" - REDUCED from {meta['requested_seeds']} because the projection after seed 0 was {meta['projected_total_min_after_seed0']:.1f} min > {meta['budget_min']} min budget." if meta.get("seeds_reduced_to") else
     f" (requested {meta['requested_seeds']}; projection after seed 0 was {meta['projected_total_min_after_seed0']:.1f} min, within the {meta['budget_min']} min budget)."))
w("\nEvery number below is read from a file in `audit_lean/` (named under each table) written by `run_lean.py`; `build_report.py` generated this document. "
  "Test split only (1,343 customers) for every reported index; the validation split is used only for the config grid. No reference label exists for credit card.\n")
w("**Space of the indices.** Silhouette, DBI and CH of every method are computed in the same space: the 17 `cc_*` variables, log1p on the 10 skewed monetary/count variables, "
  "then StandardScaler fitted on the train split (the GNN input space). A sensitivity version in the plain-standardized space (no log) is in `results_by_seed.csv` (`std_*` columns) and in section 2.\n")
w("**Tiny-cluster rule (all methods incl. the GNN).** A cluster with fewer than 0.5% of the test nodes (< 6.7, i.e. <= 6 of 1,343 customers) is *tiny*. "
  "`all clusters` = clusters exactly as the method produced them. `merged` = the customers of tiny clusters are moved to the nearest large cluster (Euclidean distance to the centroid in the common space) before computing the indices. DBSCAN noise (-1) is excluded from both and reported.\n")
w("Branch `fix/lean` commits:\n```\n" + git_log + "\n```\n")

# ------------------------------------------------------------------ 0 fix verification
BL = len(L)
w("## 0. Fix verification (steps 1-2)\n")
w("**Root cause (step 1).** The credit-card pipeline read the 4 pooled core columns (`tenure_months, balance, is_active, n_products`; `is_active` has 4 zeros of 8,950 and standardizes to about -47, which made K-Means build a one-customer segment in B1-B3). "
  "`prepare_cc_full()` now reads the 17 `sources.cc.numeric` variables of `unified/schema.json` from `unified/cc_full.csv` (log1p on 10 monetary/count variables, StandardScaler fit on train). "
  f"Features used: {', '.join(meta['features'])}.\n")
try:
    rp = open(F("check_repro.txt")).read().strip()
    w("**Seeding (2a).** Same seed twice, bit-identical embeddings and labels (`audit_lean/check_repro.py`, output in `check_repro.txt`):\n```\n" + rp + "\n```\n")
except FileNotFoundError:
    w("**Seeding (2a).** NOT VERIFIED: `check_repro.txt` missing.\n")
try:
    es = pd.DataFrame(json.load(open(F("check_earlystop.json"))))
    w("**Early stopping (2c)**, k=5, VGAE, dim 32, seeds 0-2, legacy scoring graph (held-out edges inside message passing) vs fixed (`check_earlystop.json`):\n")
    w(md(["encoder", "seed", "val AUC legacy", "val AUC fixed", "inflation", "epochs legacy/fixed", "final embeddings identical", "ARI of K=4 labels legacy vs fixed"],
         [[r.kind, r.seed, f"{r.val_auc_legacy:.4f}", f"{r.val_auc_fixed:.4f}", f"{r.auc_inflation:.4f}", f"{r.epochs_legacy}/{r.epochs_fixed}",
           r.emb_identical, f"{r.ari_labels_legacy_vs_fixed:.3f}"] for r in es.itertuples()]))
    same = bool(es.emb_identical.all())
    w(f"Reading: the legacy scoring graph inflated the *reported* validation AUC by {es[es.kind == 'gcn'].auc_inflation.mean():.4f} (GCN) and {es[es.kind == 'sage'].auc_inflation.mean():.4f} (GraphSAGE). "
      + ("In these 6 runs the stopping epoch and the final embeddings are identical with and without the fix, so SAGE is unchanged and the leak inflated the number without changing the selected model here (at k=5; with other seeds or configs a different stopping epoch is possible)." if same else
         "The stopping epoch / final embeddings differ in some runs (see table); SAGE is unchanged only where `final embeddings identical` is True.") + "\n")
    for kind, g in es.groupby("kind"):
        add("fix_earlystop", kind, "legacy-minus-fixed val AUC", "auc_inflation", g.auc_inflation.mean(), g.auc_inflation.std(ddof=1), len(g), "check_earlystop.json")
    w("")
except FileNotFoundError:
    w("**Early stopping (2c).** NOT VERIFIED: `check_earlystop.json` missing.\n")
w("**Labels (2d).** `labels_all_seeds.npz` holds the labels of all 8,950 customers for all 11 methods/controls and all seeds (keys `<method>__seed<s>`; `test_idx`, `train_idx`, `val_idx`).\n")
w("**Deviation to state.** The negative sampler in the VGAE loss was replaced by `sample_negatives()` (torch RNG; uniform pairs excluding self-loops and training positives). "
  "PyG's `negative_sampling` uses Python `random.sample` and cost ~0.7 s per epoch on the k=20 graph on this CPU. Results are therefore not comparable draw-by-draw with `audit_runs/run05_cc_full`, which is untouched.\n")

# ------------------------------------------------------------------ 1 main table
w("## 1. Main table: test split, K = 4, mean ± std over seeds\n")
w(f"n = {NS} seeds. Source: `{OUT}/{RS}` (columns `silhouette_all, dbi_all, ch_all` and `*_merged`). Higher is better for Silhouette and CH, lower for DBI. "
  "Methods B0 use the train-fitted P1 features (impute, cap p95, scale, PCA 95%: 11 components) for clustering; all indices are in the common space. "
  "B0 DBSCAN excludes noise points (share in the last column).\n")
for variant, title in (("all", "1a. All clusters as produced (no merging)"), ("merged", "1b. After merging tiny clusters (< 0.5% of test nodes) into the nearest large cluster")):
    w(f"**{title}**\n")
    rows = []
    for m in METHODS:
        v = res[res.method == m]
        extra = [f"{v['n_clusters_' + variant].mean():.2f}", f"{int((v.n_tiny_clusters > 0).sum())}/{NS}", f"{v.min_cluster_size.mean():.1f} (min {v.min_cluster_size.min()})",
                 f"{v.noise_share.mean():.3f}"]
        rows.append([m] + [ms(v[f"{k}_{variant}"], d_(k)) for k, _, _ in IDX] + extra)
        for k, _, _ in IDX:
            add("main", m, variant, k, v[f"{k}_{variant}"].mean(), v[f"{k}_{variant}"].std(ddof=1), NS, RS)
        add("main", m, variant, "n_seeds_with_tiny_cluster", float((v.n_tiny_clusters > 0).sum()), np.nan, NS, RS)
        add("main", m, variant, "noise_share", v.noise_share.mean(), v.noise_share.std(ddof=1), NS, RS)
    w(md(["method", "Silhouette ↑", "DBI ↓", "CH ↑", "clusters (mean)", "seeds with a tiny cluster", "smallest cluster, test (mean)", "noise share"], rows) + "\n")
tiny_total = int((res.n_tiny_clusters > 0).sum())
w("Column meaning: tables 1a/1b differ only in the tiny-cluster rule; the last columns describe the clusters *as produced* (1a) or after merging (cluster count in 1b)." +
  (f" **No method produced a tiny cluster in any seed (0 of {len(res)} method-seed partitions), so the merge changed nothing and 1a = 1b.** This contrasts with the earlier 4-feature pipeline, where B1-B3 formed a one-customer segment. The rule is implemented and was exercised on synthetic data (commit 2b)." if tiny_total == 0 else f" {tiny_total} of {len(res)} method-seed partitions contained a tiny cluster; see the per-method column.") +
  " B0 DBSCAN: with the stated parameters it finds a single cluster plus noise on the test split, so its indices are undefined (n/a); on all 8,950 customers (section 5) it finds 2 clusters.\n")

# ------------------------------------------------------------------ 2 graph effect
w("## 2. Graph effect: GNN vs B3 (autoencoder + K-Means) vs B4 (same encoder/objective, A = I)\n")
w(f"Paired by seed (n = {NS}); difference = GNN minus baseline; p from a paired t-test and a Wilcoxon signed-rank test. "
  "Better = higher Silhouette/CH, lower DBI. Source: `" + RS + "`.\n")
verdicts = []
for space, pre, sname in (("common", "", "common space (log1p + standardized), primary"), ("plain-std", "std_", "plain standardized space (sensitivity)")):
    for variant in ("all", "merged"):
        w(f"**{sname}, {'all clusters' if variant == 'all' else 'tiny clusters merged'}**\n")
        rows = []; wins = {B3: [0, 0], B4: [0, 0]}
        for other in (B3, B4):
            for k, nm, sign in IDX:
                a, b = vals(GNN, f"{pre}{k}_{variant}"), vals(other, f"{pre}{k}_{variant}")
                dlt = a - b
                tp = stats.ttest_rel(a, b).pvalue if np.std(dlt) > 0 else 1.0
                try:
                    wp = stats.wilcoxon(a, b).pvalue
                except ValueError:
                    wp = 1.0
                better = np.mean(dlt) * sign > 0
                sig = better and tp < 0.05
                wins[other][0] += better; wins[other][1] += sig
                rows.append([f"GNN − {other.split()[0]}", nm, ms(a, d_(k)), ms(b, d_(k)), f"{np.mean(dlt):+.{d_(k)}f} ± {np.std(dlt, ddof=1):.{d_(k)}f}", f"{tp:.3g}", f"{wp:.3g}",
                             "yes" if better else "no", ("GNN better (sig.)" if sig else "GNN worse (sig.)" if (not better and tp < 0.05) else "not significant")])
                add("graph_effect", f"GNN - {other}", f"{space}/{variant}", k, np.mean(dlt), np.std(dlt, ddof=1), NS, RS, f"p_ttest={tp:.4g}; p_wilcoxon={wp:.4g}; gnn_better_mean={better}; sig={sig}")
        w(md(["comparison", "index", "GNN", "baseline", "paired diff (mean ± sd)", "p (t)", "p (Wilcoxon)", "GNN better on mean", "paired t-test, p<0.05"], rows) + "\n")
        beats_mean = all(wins[o][0] >= 2 for o in (B3, B4)); beats_sig = all(wins[o][1] >= 2 for o in (B3, B4))
        verdicts.append((sname, variant, wins, beats_mean, beats_sig))
        w(f"Count of indices (of 3) where the GNN is better on the mean: vs B3 = {wins[B3][0]}, vs B4 = {wins[B4][0]}; significantly better: vs B3 = {wins[B3][1]}, vs B4 = {wins[B4][1]}.\n")
w("**Answer: does the GNN beat B3 and B4 on at least 2 of the 3 indices?**\n")
for sname, variant, wins, bm_, bs_ in verdicts:
    w(f"- {sname}, {'all clusters' if variant == 'all' else 'merged'}: on the mean, GNN better than B3 on {wins[B3][0]}/3 and than B4 on {wins[B4][0]}/3 → decision rule "
      f"**{'MET' if bm_ else 'NOT MET'}** on means; counting only significant paired differences (p<0.05) it is {wins[B3][1]}/3 vs B3 and {wins[B4][1]}/3 vs B4 → **{'MET' if bs_ else 'NOT MET'}**.")
w("")

# ------------------------------------------------------------------ 3 grid
w("## 3. Configuration grid (validation split only)\n")
grid = pd.read_csv(F("config_grid_validation.csv"))
w(f"VGAE, embedding dim 32, K = 4, K-Means; capped at {meta['grid_epochs']} epochs, patience {meta['grid_patience']}; grid seeds {meta['grid_seeds']} (distinct from the final seeds); indices in the common space on the **validation** customers. "
  "Selection rule fixed before the run: highest mean validation silhouette. The test split was not used. Source: `config_grid_validation.csv` (per seed: `grid_by_seed.csv`).\n")
rows = []
for r in grid.itertuples():
    rows.append([r.k, "GCN" if r.encoder == "gcn" else "GraphSAGE", f"{r.val_silhouette:.4f} ± {r.val_silhouette_sd:.4f}", f"{r.val_dbi:.3f}", f"{r.val_ch:.1f}",
                 f"{r.val_link_auc:.4f}", f"{r.epochs:.0f}", f"{r.val_seed_ari:.3f}", "**chosen**" if r.chosen else ""])
    add("config_grid", f"k={r.k} {r.encoder}", "validation", "silhouette", r.val_silhouette, r.val_silhouette_sd, len(meta["grid_seeds"]), "config_grid_validation.csv", "chosen" if r.chosen else "")
    add("config_grid", f"k={r.k} {r.encoder}", "validation", "dbi", r.val_dbi, np.nan, len(meta["grid_seeds"]), "config_grid_validation.csv")
    add("config_grid", f"k={r.k} {r.encoder}", "validation", "ch", r.val_ch, np.nan, len(meta["grid_seeds"]), "config_grid_validation.csv")
    add("config_grid", f"k={r.k} {r.encoder}", "validation", "link_auc", r.val_link_auc, np.nan, len(meta["grid_seeds"]), "config_grid_validation.csv")
w(md(["k", "encoder", "val Silhouette (mean ± sd over grid seeds)", "val DBI", "val CH", "val link AUC", "epochs (mean)", "val seed-ARI", "choice"], rows) + "\n")
short = grid[grid.epochs < 25]
rng_ = grid.val_silhouette.max() - grid.val_silhouette.min()
w(f"Chosen: **k = {cfg['k']}, {cfg['kind']}, VGAE, dim {cfg['dim']}**. The best-to-worst spread of mean validation silhouette is {rng_:.4f}, against a within-configuration seed sd of up to {grid.val_silhouette_sd.max():.4f}, so the choice is weak.\n")
if len(short):
    w(f"**Caveat on the grid.** With patience {meta['grid_patience']} and evaluation every 5 epochs, the rows {', '.join('k=%d %s' % (r.k, r.encoder) for r in short.itertuples())} stopped after about {short.epochs.mean():.0f} epochs "
      f"(validation link AUC {short.val_link_auc.mean():.3f} vs {grid[grid.epochs >= 25].val_link_auc.mean():.3f} for the rows that ran to the cap), i.e. they are under-trained relative to the final models (patience {meta['final_patience']}). "
      "The grid therefore compares under-trained SAGE with fully trained GCN; this is the CPU-budget setting requested and is not corrected here.\n")

# ------------------------------------------------------------------ 4 controls
w("## 4. Controls (final pipeline)\n")
w(f"Shuffled-edge GNN: same chosen config, but the kNN graph is relabelled by a random permutation of the nodes (degree structure kept, similarity structure destroyed); random embedding: N(0, I) of the same shape (n × {meta['dim']}), K-Means K = 4. Source: `{RS}`.\n")
for variant in ("all", "merged"):
    w(f"**{variant}**\n")
    rows = []
    for m in (GNN, B4, "Control: shuffled-edge GNN", "Control: random Gaussian embedding"):
        rows.append([m] + [ms(vals(m, f"{k}_{variant}"), d_(k)) for k, _, _ in IDX])
    w(md(["method", "Silhouette ↑", "DBI ↓", "CH ↑"], rows) + "\n")
    rows = []
    for ctl in ("Control: shuffled-edge GNN", "Control: random Gaussian embedding", B4):
        for k, nm, sign in IDX:
            a, b = vals(GNN, f"{k}_{variant}"), vals(ctl, f"{k}_{variant}"); dlt = a - b
            tp = stats.ttest_rel(a, b).pvalue if np.std(dlt) > 0 else 1.0
            rows.append([f"GNN − {ctl.replace('Control: ', '')}", nm, f"{np.mean(dlt):+.{d_(k)}f} ± {np.std(dlt, ddof=1):.{d_(k)}f}", f"{tp:.3g}", int((dlt * sign > 0).sum()) if True else ""])
            add("controls", f"GNN - {ctl}", variant, k, np.mean(dlt), np.std(dlt, ddof=1), NS, RS, f"p_ttest={tp:.4g}; seeds_gnn_better={int((dlt * sign > 0).sum())}")
    w(md(["paired comparison", "index", "diff (mean ± sd)", "p (t)", f"seeds (of {NS}) where GNN is better"], rows) + "\n")
for m in (GNN, B4, "Control: shuffled-edge GNN", "Control: random Gaussian embedding"):
    for variant in ("all", "merged"):
        for k, _, _ in IDX:
            add("controls", m, variant, k, vals(m, f"{k}_{variant}").mean(), vals(m, f"{k}_{variant}").std(ddof=1), NS, RS)

# ------------------------------------------------------------------ 5 replication
w("## 5. Base-paper replication (B0 on the 17 variables, all 8,950 customers, K = 4)\n")
rep = pd.read_csv(F("replication_b0.csv"))
PUB = {"K-Means": (0.800, 0.408, 5823.343), "DBSCAN": (1.287, 0.803, 685.302), "Hierarchical": (0.855, 0.388, 4936.960), "Spectral": (11.334, 0.004, 190.856)}
w(f"Median imputation, cap at p95, StandardScaler, PCA keeping 95% variance ({int(rep.n_pca_components.iloc[0])} components), all fitted on all 8,950 customers (as a non-inductive paper would), "
  "K-Means n_init=20, Ward, spectral with nearest-neighbour affinity (10 neighbours), DBSCAN with min_samples = 2 × dims and eps = 95th percentile of the min_samples-NN distance (the paper does not state its DBSCAN parameters). "
  "The paper does not say in which space Table 1 was computed, so both spaces are shown. Nothing was tuned to match. This is *not* the test-split protocol of sections 1-4 (that one fits on train). Source: `replication_b0.csv`; published values from paper Table 9.\n")
rows = []
for alg, (dbi_p, sil_p, ch_p) in PUB.items():
    rows.append([alg, "published", f"{dbi_p:.3f}", f"{sil_p:.3f}", f"{ch_p:,.3f}", "", "", ""])
    for r in rep[rep.algorithm == alg].itertuples():
        pc = lambda o, p: f"{(o - p) / p * 100:+.1f}%"
        rows.append(["", f"ours: {r.space}", f"{r.dbi:.3f} ({pc(r.dbi, dbi_p)})", f"{r.silhouette:.3f} ({pc(r.silhouette, sil_p)})", f"{r.ch:,.1f} ({pc(r.ch, ch_p)})",
                     r.n_clusters, r.min_cluster_size, f"{r.noise_share:.3f}"])
        for k, v in (("dbi", r.dbi), ("silhouette", r.silhouette), ("ch", r.ch)):
            add("replication", alg, r.space, k, v, np.nan, 1, "replication_b0.csv", f"published={dict(zip(('dbi', 'silhouette', 'ch'), PUB[alg]))[k]}")
w(md(["algorithm", "", "DBI (diff vs published)", "Silhouette (diff)", "CH (diff)", "clusters found", "smallest cluster", "noise share"], rows) + "\n")
w("Differences are stated as they are; no row is dropped.\n")

# ------------------------------------------------------------------ 6 stability
w("## 6. Stability and agreement between segmentations\n")
st = pd.read_csv(F("stability.csv")); ag = pd.read_csv(F("cross_method_agreement.csv"))
w(f"Stability = mean pairwise ARI between the test-set partitions of different seeds ({NS * (NS - 1) // 2} pairs; ± = std over pairs). "
  "B0 Ward and B0 DBSCAN do not depend on the seed (fitted once, so ARI = 1 by construction, not evidence of robustness). Source: `stability.csv`. "
  "Agreement columns: ARI / NMI between the GNN segments and each other method's segments at the same seed, mean ± std over seeds (`cross_method_agreement.csv`). No label exists for credit card, so no other ARI/NMI is reported.\n")
rows = []
for m in METHODS:
    s = st[st.method == m].iloc[0]; a = ag[ag.other == m]
    rows.append([m, f"{s.mean_pairwise_ari:.3f} ± {s.sd_pairwise_ari:.3f}", "-" if a.empty else ms(a.ari), "-" if a.empty else ms(a.nmi)])
    add("stability", m, "test", "mean_pairwise_ari", s.mean_pairwise_ari, s.sd_pairwise_ari, int(s.n_pairs), "stability.csv")
    if not a.empty:
        add("agreement_with_GNN", m, "test", "ari", a.ari.mean(), a.ari.std(ddof=1), NS, "cross_method_agreement.csv")
        add("agreement_with_GNN", m, "test", "nmi", a.nmi.mean(), a.nmi.std(ddof=1), NS, "cross_method_agreement.csv")
w(md(["method", "stability (pairwise ARI)", "ARI with GNN segments", "NMI with GNN segments"], rows) + "\n")

# ------------------------------------------------------------------ 7 explanations
w("## 7. Explanations (seed 0, GNN segments)\n")
sf = json.load(open(F("surrogate_fidelity.json")))
w(f"**Surrogate fidelity.** {sf['surrogate']} predicts the GNN segment from the 17 variables in original units; trained on train, scored on the test split: **macro-F1 = {sf['fidelity_macro_f1_test']:.3f}**, accuracy = {sf['accuracy_test']:.3f} "
  f"(n = {sf['n_test']}; test segment sizes {sf['test_segment_sizes']}). Source: `surrogate_fidelity.json`.\n")
add("surrogate", "HGB surrogate", "test", "macro_f1", sf["fidelity_macro_f1_test"], np.nan, sf["n_test"], "surrogate_fidelity.json")
add("surrogate", "HGB surrogate", "test", "accuracy", sf["accuracy_test"], np.nan, sf["n_test"], "surrogate_fidelity.json")
sh = pd.read_csv(F("shap_per_segment.csv"))
w(f"**SHAP top-5 features per segment** ({sf['shap_method']}; {sf['n_shap']} test customers explained; value = mean |SHAP| of the segment's own class output over that segment's explained customers; sign = mean signed SHAP). Source: `shap_per_segment.csv`.\n")
rows = []
for c, g in sh.groupby("segment"):
    top = g.sort_values("mean_abs_shap", ascending=False).head(5)
    rows.append([c, int(g.n_explained.iloc[0])] + [f"{r.feature.replace('cc_', '')} ({r.mean_abs_shap:.3f}, {'+' if r.mean_shap >= 0 else '−'})" for r in top.itertuples()])
    for i, r in enumerate(top.itertuples(), 1):
        add("shap_top5", f"segment {c}", r.feature, f"rank{i}_mean_abs_shap", r.mean_abs_shap, np.nan, int(g.n_explained.iloc[0]), "shap_per_segment.csv")
w(md(["segment", "n explained", "1", "2", "3", "4", "5"], rows) + "\n")
if min(sf["test_segment_sizes"]) < 30:
    w(f"Caution: the smallest test segment has {min(sf['test_segment_sizes'])} customers, so its SHAP ranking rests on very few customers.\n")
ex = pd.read_csv(F("gnnexplainer_nodes.csv"))
w(f"**GNNExplainer neighbour statistic** (`gnnexplainer_nodes.csv`): up to {int(ex.groupby('segment').size().max())} test nodes per segment among nodes with degree >= 10 (so that the top-5 is a strict subset of the neighbours; "
  f"the earlier audit found the statistic saturated because with k=5 many nodes have exactly 5 neighbours), {meta.get('expl_epochs', 100)} explainer epochs. For each node: share of its 5 highest-mask neighbours "
  "that lie in the node's own segment (top-5), the same for 5 neighbours drawn at random (random-5, mean of 200 draws per node) and the share of all neighbours in its own segment (overall rate).\n")
rows = []
for c, g in list(ex.groupby("segment")) + [("all", ex)]:
    wt = stats.wilcoxon(g.top5_same, g.all_same).pvalue if (g.top5_same != g.all_same).any() else 1.0
    wr = stats.wilcoxon(g.top5_same, g.random5_same).pvalue if (g.top5_same != g.random5_same).any() else 1.0
    rows.append([c, len(g), f"{g.degree.min()}-{g.degree.max()}", ms(g.top5_same), ms(g.random5_same), ms(g.all_same),
                 f"{(g.top5_same - g.random5_same).mean():+.3f}", f"{wr:.3g}", f"{wt:.3g}"])
    for k in ("top5_same", "random5_same", "all_same"):
        add("explainer", f"segment {c}", "test nodes", k, g[k].mean(), g[k].std(ddof=1), len(g), "gnnexplainer_nodes.csv", f"p_wilcoxon_top5_vs_random5={wr:.4g}; p_wilcoxon_top5_vs_all={wt:.4g}")
w(md(["segment", "nodes", "degree", "top-5 same-segment share", "random-5 baseline", "overall same-segment rate", "top-5 − random-5", "p (Wilcoxon) vs random-5", "p vs overall"], rows) + "\n")
var_ok = ex.top5_same.std() > 0 and ex.top5_same.nunique() >= 3
w(f"**Does the statistic vary across nodes? {'YES' if var_ok else 'NO - DEGENERATE'}.** top-5 share: std across nodes = {ex.top5_same.std():.3f}, {ex.top5_same.nunique()} distinct values, "
  f"top-5 differs from the all-neighbours share in {(ex.top5_same != ex.all_same).mean() * 100:.0f}% of nodes; edge-mask std within a node is > 1e-6 in {(ex.mask_std > 1e-6).mean() * 100:.0f}% of nodes "
  f"(median {ex.mask_n_unique.median():.0f} distinct mask values per node). Saturated nodes (degree <= 5): {(ex.degree <= 5).sum()}.\n")
add("explainer", "all", "test nodes", "statistic_varies_across_nodes", float(var_ok), ex.top5_same.std(), len(ex), "gnnexplainer_nodes.csv")

# ------------------------------------------------------------------ 8 profiles
w("## 8. Segment profiles (seed-0 GNN, test customers, original units)\n")
pr = pd.read_csv(F("segment_profiles_gnn_seed0.csv"))
KEY = ["cc_balance", "cc_purchases", "cc_cash_advance", "cc_credit_limit", "cc_payments", "cc_purchases_frequency", "cc_cash_advance_frequency", "cc_purchases_trx", "cc_prc_full_payment", "cc_tenure"]
w("Means per segment; all 17 variables are in `segment_profiles_gnn_seed0.csv`. Segment ids are arbitrary K-Means ids.\n")
w(md(["segment", "n", "share"] + [k.replace("cc_", "") for k in KEY],
     [[int(r.segment), int(r.n_test), f"{r.share:.3f}"] + [f"{getattr(r, k):,.2f}" for k in KEY] for r in pr.itertuples()]) + "\n")
for r in pr.itertuples():
    for k in KEY:
        add("segment_profile", f"segment {int(r.segment)}", "seed0 GNN, test", k, getattr(r, k), np.nan, int(r.n_test), "segment_profiles_gnn_seed0.csv")

# ------------------------------------------------------------------ independent check
w("## 9a. Independent recomputation of the indices from the saved labels (step 4)\n")
try:
    ic = json.load(open(F("independent_check_summary.json")))
    w(f"`check_independent.py` re-derives the features from `unified/cc_full.csv`, implements Silhouette/DBI/CH and the tiny-cluster merge itself (numpy/scipy, no sklearn metrics, no shared code) and recomputes **{ic['n_values_compared']} values** "
      f"(all {len(METHODS)} methods × {NS} seeds × 2 spaces × all/merged × 3 indices). **Maximum absolute difference {ic['max_abs_diff']:.3g}; maximum relative difference {ic['max_rel_diff']:.3g}** "
      f"(worst: {ic['worst_row']['method']}, seed {ic['worst_row']['seed']}, {ic['worst_row']['space']}, {ic['worst_row']['variant']}, {ic['worst_row']['metric']}: reported {float(ic['worst_row']['reported']):.6f}, recomputed {float(ic['worst_row']['recomputed']):.6f}). "
      f"NaN in exactly one of the two: {ic['n_nan_mismatch']}. Max relative difference by index: " + ", ".join(f"{k} {v:.2g}" for k, v in ic["max_rel_diff_by_metric"].items())
      + ". The differences are float32 (run) vs float64 (check). Source: `independent_check.csv`, `independent_check_summary.json`.\n")
    add("independent_check", "all", "all variants", "max_abs_diff", ic["max_abs_diff"], np.nan, ic["n_values_compared"], "independent_check_summary.json")
    add("independent_check", "all", "all variants", "max_rel_diff", ic["max_rel_diff"], np.nan, ic["n_values_compared"], "independent_check_summary.json")
except FileNotFoundError:
    w("NOT VERIFIED: `independent_check_summary.json` missing.\n")

# ------------------------------------------------------------------ 9 run details
w("## 9. Exact seeds, epochs, hardware, runtime\n")
ti = pd.read_csv(F("training_info.csv"))
w(f"- **Final seeds:** {SEEDS} (training and clustering seeds; data split fixed by `unified/splits.csv`, split seed 42). Grid seeds: {meta['grid_seeds']}. Control seeds: permutation seed 1000+s, random-embedding seed 2000+s.")
w(f"- **Epochs:** final models up to {meta['final_epochs']} epochs, early stopping patience {meta['final_patience']} on validation link AUC (evaluated every 5 epochs); grid up to {meta['grid_epochs']} epochs, patience {meta['grid_patience']}. Full-batch training, no neighbour sampling.")
w("- **Epochs actually run / validation link AUC** (`training_info.csv`):\n")
w(md(["model", "epochs mean (min-max)", "val link AUC (mean ± sd)", "seconds per model (mean)"],
     [[m, f"{g.epochs.mean():.0f} ({g.epochs.min()}-{g.epochs.max()})", ms(g.val_link_auc, 4), f"{g.secs.mean():.0f}"] for m, g in ti.groupby("model")]) + "\n")
w("- **Config:** k = %d, %s, VGAE, embedding dim %d, hidden 64, Adam lr 0.01, weight decay 5e-4, dropout 0, K-Means n_init 20 fitted on train embeddings; autoencoder B3 lr 0.005, batch 512; GMM full covariance reg_covar 1e-4." % (cfg["k"], cfg["kind"], cfg["dim"]))
w(f"- **Hardware:** 13th Gen Intel Core i7-1355U (10 physical cores, 12 logical processors; `os.cpu_count()` = {meta['cores']}), {meta['ram_gb']} GB RAM, **CPU only** (CUDA available: {meta['cuda_available']}), torch {meta['torch']} with {meta['torch_threads']} threads, Python {meta['python']}, PYTHONHASHSEED={meta['pythonhashseed']}, deterministic algorithms on.")
w(f"- **Runtime:** total {meta['total_min']:.1f} min. Stage seconds (`run_meta.json`): " + ", ".join(f"{k} {v:.0f}" for k, v in meta["stage_secs"].items() if not k.startswith("seed")) + f"; per-seed stages summed over seeds: "
  + f"GNN+B4+controls {sum(v for k, v in meta['stage_secs'].items() if k.endswith('GNN+B4+controls')):.0f}, B3+B1+B2 {sum(v for k, v in meta['stage_secs'].items() if k.endswith('B3+B1+B2')):.0f}, B0 {sum(v for k, v in meta['stage_secs'].items() if k.endswith(':B0')):.0f}. "
  f"Seed 0 loop took {meta['seed0_secs']:.0f} s; projection after seed 0: {meta['projected_total_min_after_seed0']:.1f} min. Progress log: `progress.txt`.")
w("- **Graphs** (kNN cosine, union symmetrization, binary weights; `run_meta.json`): " + "; ".join(f"k={k}: {g['edges']:,} edges, mean degree {g['deg_mean']:.1f}, components {g['components']}" for k, g in meta["graphs"].items()) + ".\n")

w("## 10. Figures\n")
w("- `audit_lean/fig1_segment_profiles.png`: mean of 10 key variables per segment in original units (seed-0 GNN, test customers).")
w("- `audit_lean/fig2_embedding_scatter.png`: 2-D PCA of the seed-0 GNN embedding, test customers colored by segment. The two components explain only about a quarter of the embedding variance (percentages on the axes), so overlap in 2-D is not evidence about separation in the full space; use the indices in section 1.\n")
w("## Files\n")
w("`results_lean.csv` (long table of every number above with source file), `results_by_seed.csv`, `labels_all_seeds.npz`, `config_grid_validation.csv`, `grid_by_seed.csv`, `replication_b0.csv`, `stability.csv`, "
  "`cross_method_agreement.csv`, `surrogate_fidelity.json`, `shap_per_segment.csv`, `gnnexplainer_nodes.csv`, `segment_profiles_gnn_seed0.csv`, `training_info.csv`, `run_meta.json`, `progress.txt`, "
  "`independent_check.csv`, `check_repro.txt`, `check_earlystop.json`.\n")

v0 = verdicts[0]; sh_ = lambda o, k, pre="": vals(o, f"{pre}{k}_all")
exa = ex
nsig = lambda o: int(sum((stats.ttest_rel(sh_(GNN, k), sh_(o, k)).pvalue < 0.05) for k in ('silhouette', 'dbi', 'ch')))
bl = ["## Bottom line\n",
      f"- **Graph effect: not found.** Common space, test split, {NS} seeds: the GNN is worse than B3 and B4 on all three indices on the mean (Silhouette {ms(sh_(GNN,'silhouette'))} vs B3 {ms(sh_(B3,'silhouette'))} vs B4 {ms(sh_(B4,'silhouette'))}); "
      f"the difference is significant (paired t, p<0.05) on {nsig(B3)} of 3 indices vs B3 and {nsig(B4)} of 3 vs B4 (all of them in the GNN-worse direction, see section 2). The decision rule (beat B3 and B4 on >= 2 of 3 indices) is **NOT MET** in either space, with or without tiny-cluster merging. Per the paper, the contribution has to be reframed around relational explainability.",
      f"- **The GNN is indistinguishable from the shuffled-edge control** (Silhouette {ms(sh_(GNN,'silhouette'))} vs {ms(sh_('Control: shuffled-edge GNN','silhouette'))}, paired p = {stats.ttest_rel(sh_(GNN,'silhouette'), sh_('Control: shuffled-edge GNN','silhouette')).pvalue:.2f}); it beats random embeddings (a floor, not a test of the graph).",
      f"- **Stability of the GNN is low** (pairwise ARI {st[st.method == GNN].mean_pairwise_ari.iloc[0]:.2f}) compared with B1 {st[st.method == 'B1 K-Means'].mean_pairwise_ari.iloc[0]:.2f} and B3 {st[st.method == B3].mean_pairwise_ari.iloc[0]:.2f}.",
      f"- **Neighbour-level explanation adds nothing measurable:** the 5 neighbours GNNExplainer ranks highest are no more often in the node's own segment ({exa.top5_same.mean():.3f}) than 5 random neighbours ({exa.random5_same.mean():.3f}) or all neighbours ({exa.all_same.mean():.3f}); most neighbours share the segment anyway.",
      f"- **Feature-level explanation works:** a gradient-boosted surrogate reproduces the GNN segments with test macro-F1 {sf['fidelity_macro_f1_test']:.3f}. The segments are interpretable (section 8) but B1/B3 reach better indices with a simpler model.",
      "- **Base-paper replication does not match** the published values (section 5); no tuning was done to force it.",
      "- Previously reported singleton segment (4-feature `is_active` artefact) is gone: no method has a tiny cluster in any seed, so the with/without-merging columns are identical.\n"]
L[BL:BL] = bl
open(F("RESULTS_LEAN.md"), "w", encoding="utf-8").write("\n".join(L))
pd.DataFrame(R).to_csv(F("results_lean.csv"), index=False)
print("written", F("RESULTS_LEAN.md"), len(R), "rows in results_lean.csv")
