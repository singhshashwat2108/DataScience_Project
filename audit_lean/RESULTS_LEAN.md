# RESULTS_LEAN: credit card, 17 behavioral variables, K = 4, test split

Run: 2026-10-28 10:09:36 to 2026-10-28 10:33:48 (24.2 min wall clock). **Seeds actually run: 10 (0..9)** (requested 10; projection after seed 0 was 30.3 min, within the 45.0 min budget).

Every number below is read from a file in `audit_lean/` (named under each table) written by `run_lean.py`; `build_report.py` generated this document. Test split only (1,343 customers) for every reported index; the validation split is used only for the config grid. No reference label exists for credit card.

**Space of the indices.** Silhouette, DBI and CH of every method are computed in the same space: the 17 `cc_*` variables, log1p on the 10 skewed monetary/count variables, then StandardScaler fitted on the train split (the GNN input space). A sensitivity version in the plain-standardized space (no log) is in `results_by_seed.csv` (`std_*` columns) and in section 2.

**Tiny-cluster rule (all methods incl. the GNN).** A cluster with fewer than 0.5% of the test nodes (< 6.7, i.e. <= 6 of 1,343 customers) is *tiny*. `all clusters` = clusters exactly as the method produced them. `merged` = the customers of tiny clusters are moved to the nearest large cluster (Euclidean distance to the centroid in the common space) before computing the indices. DBSCAN noise (-1) is excluded from both and reported.

Branch `fix/lean` commits:
```
fda0ac0 Early stopping scored without held-out edges in the message-passing graph; torch-seeded negative sampler
dc32b2d One tiny-cluster rule for all methods: report clusters < 0.5% of nodes, indices with and without merging
8872c70 Seed Python random, NumPy, torch, PYTHONHASHSEED and enable deterministic mode in training
2b69609 Credit card: use the 17 cc_* variables from unified/cc_full.csv (schema.json), log1p + train-only scaling
```

## Bottom line

- **Graph effect: not found.** Common space, test split, 10 seeds: the GNN is worse than B3 and B4 on all three indices on the mean (Silhouette 0.155 ± 0.011 vs B3 0.225 ± 0.001 vs B4 0.186 ± 0.022); the difference is significant (paired t, p<0.05) on 3 of 3 indices vs B3 and 2 of 3 vs B4 (all of them in the GNN-worse direction, see section 2). The decision rule (beat B3 and B4 on >= 2 of 3 indices) is **NOT MET** in either space, with or without tiny-cluster merging. Per the paper, the contribution has to be reframed around relational explainability.
- **The GNN is indistinguishable from the shuffled-edge control** (Silhouette 0.155 ± 0.011 vs 0.156 ± 0.065, paired p = 0.99); it beats random embeddings (a floor, not a test of the graph).
- **Stability of the GNN is low** (pairwise ARI 0.49) compared with B1 1.00 and B3 0.93.
- **Neighbour-level explanation adds nothing measurable:** the 5 neighbours GNNExplainer ranks highest are no more often in the node's own segment (0.932) than 5 random neighbours (0.949) or all neighbours (0.949); most neighbours share the segment anyway.
- **Feature-level explanation works:** a gradient-boosted surrogate reproduces the GNN segments with test macro-F1 0.986. The segments are interpretable (section 8) but B1/B3 reach better indices with a simpler model.
- **Base-paper replication does not match** the published values (section 5); no tuning was done to force it.
- Previously reported singleton segment (4-feature `is_active` artefact) is gone: no method has a tiny cluster in any seed, so the with/without-merging columns are identical.

## 0. Fix verification (steps 1-2)

**Root cause (step 1).** The credit-card pipeline read the 4 pooled core columns (`tenure_months, balance, is_active, n_products`; `is_active` has 4 zeros of 8,950 and standardizes to about -47, which made K-Means build a one-customer segment in B1-B3). `prepare_cc_full()` now reads the 17 `sources.cc.numeric` variables of `unified/schema.json` from `unified/cc_full.csv` (log1p on 10 monetary/count variables, StandardScaler fit on train). Features used: cc_balance, cc_balance_frequency, cc_purchases, cc_oneoff_purchases, cc_installments_purchases, cc_cash_advance, cc_purchases_frequency, cc_oneoff_purchases_frequency, cc_purchases_installments_frequency, cc_cash_advance_frequency, cc_cash_advance_trx, cc_purchases_trx, cc_credit_limit, cc_payments, cc_minimum_payments, cc_prc_full_payment, cc_tenure.

**Seeding (2a).** Same seed twice, bit-identical embeddings and labels (`audit_lean/check_repro.py`, output in `check_repro.txt`):
```
12 threads sage bit_identical True maxdiff 0.0 labels_identical True
12 threads gcn bit_identical True maxdiff 0.0 labels_identical True
AE bit_identical True
```

**Early stopping (2c)**, k=5, VGAE, dim 32, seeds 0-2, legacy scoring graph (held-out edges inside message passing) vs fixed (`check_earlystop.json`):

| encoder | seed | val AUC legacy | val AUC fixed | inflation | epochs legacy/fixed | final embeddings identical | ARI of K=4 labels legacy vs fixed |
|---|---|---|---|---|---|---|---|
| gcn | 0 | 0.9916 | 0.9863 | 0.0053 | 171/171 | True | 1.000 |
| gcn | 1 | 0.9950 | 0.9903 | 0.0047 | 156/156 | True | 1.000 |
| gcn | 2 | 0.9946 | 0.9900 | 0.0046 | 186/186 | True | 1.000 |
| sage | 0 | 0.9913 | 0.9891 | 0.0023 | 181/181 | True | 1.000 |
| sage | 1 | 0.9930 | 0.9910 | 0.0021 | 200/200 | True | 1.000 |
| sage | 2 | 0.9933 | 0.9910 | 0.0023 | 200/200 | True | 1.000 |
Reading: the legacy scoring graph inflated the *reported* validation AUC by 0.0049 (GCN) and 0.0022 (GraphSAGE). In these 6 runs the stopping epoch and the final embeddings are identical with and without the fix, so SAGE is unchanged and the leak inflated the number without changing the selected model here (at k=5; with other seeds or configs a different stopping epoch is possible).


**Labels (2d).** `labels_all_seeds.npz` holds the labels of all 8,950 customers for all 11 methods/controls and all seeds (keys `<method>__seed<s>`; `test_idx`, `train_idx`, `val_idx`).

**Deviation to state.** The negative sampler in the VGAE loss was replaced by `sample_negatives()` (torch RNG; uniform pairs excluding self-loops and training positives). PyG's `negative_sampling` uses Python `random.sample` and cost ~0.7 s per epoch on the k=20 graph on this CPU. Results are therefore not comparable draw-by-draw with `audit_runs/run05_cc_full`, which is untouched.

## 1. Main table: test split, K = 4, mean ± std over seeds

n = 10 seeds. Source: `audit_lean/results_by_seed.csv` (columns `silhouette_all, dbi_all, ch_all` and `*_merged`). Higher is better for Silhouette and CH, lower for DBI. Methods B0 use the train-fitted P1 features (impute, cap p95, scale, PCA 95%: 11 components) for clustering; all indices are in the common space. B0 DBSCAN excludes noise points (share in the last column).

**1a. All clusters as produced (no merging)**

| method | Silhouette ↑ | DBI ↓ | CH ↑ | clusters (mean) | seeds with a tiny cluster | smallest cluster, test (mean) | noise share |
|---|---|---|---|---|---|---|---|
| B0 K-Means | 0.121 ± 0.000 | 1.872 ± 0.000 | 252.3 ± 0.1 | 4.00 | 0/10 | 192.0 (min 192) | 0.000 |
| B0 DBSCAN | n/a | n/a | n/a | 1.00 | 0/10 | 1329.0 (min 1329) | 0.010 |
| B0 Hierarchical | 0.084 ± 0.000 | 1.961 ± 0.000 | 188.1 ± 0.0 | 4.00 | 0/10 | 121.0 (min 121) | 0.000 |
| B0 Spectral | 0.118 ± 0.000 | 2.051 ± 0.000 | 247.5 ± 0.0 | 4.00 | 0/10 | 278.0 (min 278) | 0.000 |
| B1 K-Means | 0.225 ± 0.000 | 1.677 ± 0.001 | 344.0 ± 0.0 | 4.00 | 0/10 | 238.4 (min 237) | 0.000 |
| B2 GMM | 0.184 ± 0.005 | 1.966 ± 0.277 | 264.6 ± 18.4 | 4.00 | 0/10 | 221.4 (min 131) | 0.000 |
| B3 AE+K-Means | 0.225 ± 0.001 | 1.669 ± 0.006 | 332.9 ± 1.2 | 4.00 | 0/10 | 178.5 (min 174) | 0.000 |
| B4 A=I+K-Means | 0.186 ± 0.022 | 1.862 ± 0.123 | 282.7 ± 27.6 | 4.00 | 0/10 | 236.3 (min 180) | 0.000 |
| GNN VGAE+K-Means | 0.155 ± 0.011 | 1.974 ± 0.100 | 239.5 ± 17.1 | 4.00 | 0/10 | 236.8 (min 199) | 0.000 |
| Control: shuffled-edge GNN | 0.156 ± 0.065 | 2.808 ± 2.758 | 239.7 ± 84.1 | 4.00 | 0/10 | 258.2 (min 191) | 0.000 |
| Control: random Gaussian embedding | -0.012 ± 0.001 | 37.351 ± 5.361 | 0.9 ± 0.3 | 4.00 | 0/10 | 310.2 (min 289) | 0.000 |

**1b. After merging tiny clusters (< 0.5% of test nodes) into the nearest large cluster**

| method | Silhouette ↑ | DBI ↓ | CH ↑ | clusters (mean) | seeds with a tiny cluster | smallest cluster, test (mean) | noise share |
|---|---|---|---|---|---|---|---|
| B0 K-Means | 0.121 ± 0.000 | 1.872 ± 0.000 | 252.3 ± 0.1 | 4.00 | 0/10 | 192.0 (min 192) | 0.000 |
| B0 DBSCAN | n/a | n/a | n/a | 1.00 | 0/10 | 1329.0 (min 1329) | 0.010 |
| B0 Hierarchical | 0.084 ± 0.000 | 1.961 ± 0.000 | 188.1 ± 0.0 | 4.00 | 0/10 | 121.0 (min 121) | 0.000 |
| B0 Spectral | 0.118 ± 0.000 | 2.051 ± 0.000 | 247.5 ± 0.0 | 4.00 | 0/10 | 278.0 (min 278) | 0.000 |
| B1 K-Means | 0.225 ± 0.000 | 1.677 ± 0.001 | 344.0 ± 0.0 | 4.00 | 0/10 | 238.4 (min 237) | 0.000 |
| B2 GMM | 0.184 ± 0.005 | 1.966 ± 0.277 | 264.6 ± 18.4 | 4.00 | 0/10 | 221.4 (min 131) | 0.000 |
| B3 AE+K-Means | 0.225 ± 0.001 | 1.669 ± 0.006 | 332.9 ± 1.2 | 4.00 | 0/10 | 178.5 (min 174) | 0.000 |
| B4 A=I+K-Means | 0.186 ± 0.022 | 1.862 ± 0.123 | 282.7 ± 27.6 | 4.00 | 0/10 | 236.3 (min 180) | 0.000 |
| GNN VGAE+K-Means | 0.155 ± 0.011 | 1.974 ± 0.100 | 239.5 ± 17.1 | 4.00 | 0/10 | 236.8 (min 199) | 0.000 |
| Control: shuffled-edge GNN | 0.156 ± 0.065 | 2.808 ± 2.758 | 239.7 ± 84.1 | 4.00 | 0/10 | 258.2 (min 191) | 0.000 |
| Control: random Gaussian embedding | -0.012 ± 0.001 | 37.351 ± 5.361 | 0.9 ± 0.3 | 4.00 | 0/10 | 310.2 (min 289) | 0.000 |

Column meaning: tables 1a/1b differ only in the tiny-cluster rule; the last columns describe the clusters *as produced* (1a) or after merging (cluster count in 1b). **No method produced a tiny cluster in any seed (0 of 110 method-seed partitions), so the merge changed nothing and 1a = 1b.** This contrasts with the earlier 4-feature pipeline, where B1-B3 formed a one-customer segment. The rule is implemented and was exercised on synthetic data (commit 2b). B0 DBSCAN: with the stated parameters it finds a single cluster plus noise on the test split, so its indices are undefined (n/a); on all 8,950 customers (section 5) it finds 2 clusters.

## 2. Graph effect: GNN vs B3 (autoencoder + K-Means) vs B4 (same encoder/objective, A = I)

Paired by seed (n = 10); difference = GNN minus baseline; p from a paired t-test and a Wilcoxon signed-rank test. Better = higher Silhouette/CH, lower DBI. Source: `results_by_seed.csv`.

**common space (log1p + standardized), primary, all clusters**

| comparison | index | GNN | baseline | paired diff (mean ± sd) | p (t) | p (Wilcoxon) | GNN better on mean | paired t-test, p<0.05 |
|---|---|---|---|---|---|---|---|---|
| GNN − B3 | Silhouette | 0.155 ± 0.011 | 0.225 ± 0.001 | -0.070 ± 0.011 | 9.9e-09 | 0.00195 | no | GNN worse (sig.) |
| GNN − B3 | DBI | 1.974 ± 0.100 | 1.669 ± 0.006 | +0.305 ± 0.101 | 5.37e-06 | 0.00195 | no | GNN worse (sig.) |
| GNN − B3 | CH | 239.5 ± 17.1 | 332.9 ± 1.2 | -93.4 ± 16.9 | 2.94e-08 | 0.00195 | no | GNN worse (sig.) |
| GNN − B4 | Silhouette | 0.155 ± 0.011 | 0.186 ± 0.022 | -0.031 ± 0.024 | 0.00268 | 0.00586 | no | GNN worse (sig.) |
| GNN − B4 | DBI | 1.974 ± 0.100 | 1.862 ± 0.123 | +0.112 ± 0.190 | 0.0964 | 0.084 | no | not significant |
| GNN − B4 | CH | 239.5 ± 17.1 | 282.7 ± 27.6 | -43.2 ± 30.8 | 0.00166 | 0.00586 | no | GNN worse (sig.) |

Count of indices (of 3) where the GNN is better on the mean: vs B3 = 0, vs B4 = 0; significantly better: vs B3 = 0, vs B4 = 0.

**common space (log1p + standardized), primary, tiny clusters merged**

| comparison | index | GNN | baseline | paired diff (mean ± sd) | p (t) | p (Wilcoxon) | GNN better on mean | paired t-test, p<0.05 |
|---|---|---|---|---|---|---|---|---|
| GNN − B3 | Silhouette | 0.155 ± 0.011 | 0.225 ± 0.001 | -0.070 ± 0.011 | 9.9e-09 | 0.00195 | no | GNN worse (sig.) |
| GNN − B3 | DBI | 1.974 ± 0.100 | 1.669 ± 0.006 | +0.305 ± 0.101 | 5.37e-06 | 0.00195 | no | GNN worse (sig.) |
| GNN − B3 | CH | 239.5 ± 17.1 | 332.9 ± 1.2 | -93.4 ± 16.9 | 2.94e-08 | 0.00195 | no | GNN worse (sig.) |
| GNN − B4 | Silhouette | 0.155 ± 0.011 | 0.186 ± 0.022 | -0.031 ± 0.024 | 0.00268 | 0.00586 | no | GNN worse (sig.) |
| GNN − B4 | DBI | 1.974 ± 0.100 | 1.862 ± 0.123 | +0.112 ± 0.190 | 0.0964 | 0.084 | no | not significant |
| GNN − B4 | CH | 239.5 ± 17.1 | 282.7 ± 27.6 | -43.2 ± 30.8 | 0.00166 | 0.00586 | no | GNN worse (sig.) |

Count of indices (of 3) where the GNN is better on the mean: vs B3 = 0, vs B4 = 0; significantly better: vs B3 = 0, vs B4 = 0.

**plain standardized space (sensitivity), all clusters**

| comparison | index | GNN | baseline | paired diff (mean ± sd) | p (t) | p (Wilcoxon) | GNN better on mean | paired t-test, p<0.05 |
|---|---|---|---|---|---|---|---|---|
| GNN − B3 | Silhouette | 0.062 ± 0.013 | 0.158 ± 0.003 | -0.096 ± 0.015 | 1.11e-08 | 0.00195 | no | GNN worse (sig.) |
| GNN − B3 | DBI | 2.523 ± 0.092 | 1.829 ± 0.019 | +0.694 ± 0.103 | 5.02e-09 | 0.00195 | no | GNN worse (sig.) |
| GNN − B3 | CH | 115.3 ± 8.4 | 189.0 ± 2.9 | -73.7 ± 9.4 | 1.39e-09 | 0.00195 | no | GNN worse (sig.) |
| GNN − B4 | Silhouette | 0.062 ± 0.013 | 0.104 ± 0.023 | -0.041 ± 0.023 | 0.000333 | 0.00195 | no | GNN worse (sig.) |
| GNN − B4 | DBI | 2.523 ± 0.092 | 2.220 ± 0.224 | +0.303 ± 0.287 | 0.00863 | 0.0273 | no | GNN worse (sig.) |
| GNN − B4 | CH | 115.3 ± 8.4 | 161.0 ± 16.4 | -45.7 ± 19.4 | 3.83e-05 | 0.00195 | no | GNN worse (sig.) |

Count of indices (of 3) where the GNN is better on the mean: vs B3 = 0, vs B4 = 0; significantly better: vs B3 = 0, vs B4 = 0.

**plain standardized space (sensitivity), tiny clusters merged**

| comparison | index | GNN | baseline | paired diff (mean ± sd) | p (t) | p (Wilcoxon) | GNN better on mean | paired t-test, p<0.05 |
|---|---|---|---|---|---|---|---|---|
| GNN − B3 | Silhouette | 0.062 ± 0.013 | 0.158 ± 0.003 | -0.096 ± 0.015 | 1.11e-08 | 0.00195 | no | GNN worse (sig.) |
| GNN − B3 | DBI | 2.523 ± 0.092 | 1.829 ± 0.019 | +0.694 ± 0.103 | 5.02e-09 | 0.00195 | no | GNN worse (sig.) |
| GNN − B3 | CH | 115.3 ± 8.4 | 189.0 ± 2.9 | -73.7 ± 9.4 | 1.39e-09 | 0.00195 | no | GNN worse (sig.) |
| GNN − B4 | Silhouette | 0.062 ± 0.013 | 0.104 ± 0.023 | -0.041 ± 0.023 | 0.000333 | 0.00195 | no | GNN worse (sig.) |
| GNN − B4 | DBI | 2.523 ± 0.092 | 2.220 ± 0.224 | +0.303 ± 0.287 | 0.00863 | 0.0273 | no | GNN worse (sig.) |
| GNN − B4 | CH | 115.3 ± 8.4 | 161.0 ± 16.4 | -45.7 ± 19.4 | 3.83e-05 | 0.00195 | no | GNN worse (sig.) |

Count of indices (of 3) where the GNN is better on the mean: vs B3 = 0, vs B4 = 0; significantly better: vs B3 = 0, vs B4 = 0.

**Answer: does the GNN beat B3 and B4 on at least 2 of the 3 indices?**

- common space (log1p + standardized), primary, all clusters: on the mean, GNN better than B3 on 0/3 and than B4 on 0/3 → decision rule **NOT MET** on means; counting only significant paired differences (p<0.05) it is 0/3 vs B3 and 0/3 vs B4 → **NOT MET**.
- common space (log1p + standardized), primary, merged: on the mean, GNN better than B3 on 0/3 and than B4 on 0/3 → decision rule **NOT MET** on means; counting only significant paired differences (p<0.05) it is 0/3 vs B3 and 0/3 vs B4 → **NOT MET**.
- plain standardized space (sensitivity), all clusters: on the mean, GNN better than B3 on 0/3 and than B4 on 0/3 → decision rule **NOT MET** on means; counting only significant paired differences (p<0.05) it is 0/3 vs B3 and 0/3 vs B4 → **NOT MET**.
- plain standardized space (sensitivity), merged: on the mean, GNN better than B3 on 0/3 and than B4 on 0/3 → decision rule **NOT MET** on means; counting only significant paired differences (p<0.05) it is 0/3 vs B3 and 0/3 vs B4 → **NOT MET**.

## 3. Configuration grid (validation split only)

VGAE, embedding dim 32, K = 4, K-Means; capped at 100 epochs, patience 10; grid seeds [100, 101] (distinct from the final seeds); indices in the common space on the **validation** customers. Selection rule fixed before the run: highest mean validation silhouette. The test split was not used. Source: `config_grid_validation.csv` (per seed: `grid_by_seed.csv`).

| k | encoder | val Silhouette (mean ± sd over grid seeds) | val DBI | val CH | val link AUC | epochs (mean) | val seed-ARI | choice |
|---|---|---|---|---|---|---|---|---|
| 5 | GCN | 0.1464 ± 0.0044 | 1.894 | 244.1 | 0.9883 | 100 | 0.855 |  |
| 5 | GraphSAGE | 0.1746 ± 0.0316 | 1.924 | 278.2 | 0.9210 | 11 | 0.327 |  |
| 10 | GCN | 0.1738 ± 0.0030 | 1.932 | 261.8 | 0.9877 | 100 | 0.495 |  |
| 10 | GraphSAGE | 0.1765 ± 0.0300 | 1.908 | 280.0 | 0.9127 | 11 | 0.368 |  |
| 20 | GCN | 0.1676 ± 0.0344 | 1.912 | 263.0 | 0.9888 | 100 | 0.595 |  |
| 20 | GraphSAGE | 0.1768 ± 0.0300 | 1.912 | 280.3 | 0.9108 | 11 | 0.359 | **chosen** |

Chosen: **k = 20, sage, VGAE, dim 32**. The best-to-worst spread of mean validation silhouette is 0.0304, against a within-configuration seed sd of up to 0.0344, so the choice is weak.

**Caveat on the grid.** With patience 10 and evaluation every 5 epochs, the rows k=5 sage, k=10 sage, k=20 sage stopped after about 11 epochs (validation link AUC 0.915 vs 0.988 for the rows that ran to the cap), i.e. they are under-trained relative to the final models (patience 20). The grid therefore compares under-trained SAGE with fully trained GCN; this is the CPU-budget setting requested and is not corrected here.

## 4. Controls (final pipeline)

Shuffled-edge GNN: same chosen config, but the kNN graph is relabelled by a random permutation of the nodes (degree structure kept, similarity structure destroyed); random embedding: N(0, I) of the same shape (n × 32), K-Means K = 4. Source: `results_by_seed.csv`.

**all**

| method | Silhouette ↑ | DBI ↓ | CH ↑ |
|---|---|---|---|
| GNN VGAE+K-Means | 0.155 ± 0.011 | 1.974 ± 0.100 | 239.5 ± 17.1 |
| B4 A=I+K-Means | 0.186 ± 0.022 | 1.862 ± 0.123 | 282.7 ± 27.6 |
| Control: shuffled-edge GNN | 0.156 ± 0.065 | 2.808 ± 2.758 | 239.7 ± 84.1 |
| Control: random Gaussian embedding | -0.012 ± 0.001 | 37.351 ± 5.361 | 0.9 ± 0.3 |

| paired comparison | index | diff (mean ± sd) | p (t) | seeds (of 10) where GNN is better |
|---|---|---|---|---|
| GNN − shuffled-edge GNN | Silhouette | -0.000 ± 0.064 | 0.987 | 3 |
| GNN − shuffled-edge GNN | DBI | -0.834 ± 2.702 | 0.355 | 4 |
| GNN − shuffled-edge GNN | CH | -0.2 ± 77.8 | 0.993 | 3 |
| GNN − random Gaussian embedding | Silhouette | +0.167 ± 0.011 | 3.79e-12 | 10 |
| GNN − random Gaussian embedding | DBI | -35.377 ± 5.346 | 6.09e-09 | 10 |
| GNN − random Gaussian embedding | CH | +238.5 ± 17.2 | 8.21e-12 | 10 |
| GNN − B4 A=I+K-Means | Silhouette | -0.031 ± 0.024 | 0.00268 | 1 |
| GNN − B4 A=I+K-Means | DBI | +0.112 ± 0.190 | 0.0964 | 2 |
| GNN − B4 A=I+K-Means | CH | -43.2 ± 30.8 | 0.00166 | 1 |

**merged**

| method | Silhouette ↑ | DBI ↓ | CH ↑ |
|---|---|---|---|
| GNN VGAE+K-Means | 0.155 ± 0.011 | 1.974 ± 0.100 | 239.5 ± 17.1 |
| B4 A=I+K-Means | 0.186 ± 0.022 | 1.862 ± 0.123 | 282.7 ± 27.6 |
| Control: shuffled-edge GNN | 0.156 ± 0.065 | 2.808 ± 2.758 | 239.7 ± 84.1 |
| Control: random Gaussian embedding | -0.012 ± 0.001 | 37.351 ± 5.361 | 0.9 ± 0.3 |

| paired comparison | index | diff (mean ± sd) | p (t) | seeds (of 10) where GNN is better |
|---|---|---|---|---|
| GNN − shuffled-edge GNN | Silhouette | -0.000 ± 0.064 | 0.987 | 3 |
| GNN − shuffled-edge GNN | DBI | -0.834 ± 2.702 | 0.355 | 4 |
| GNN − shuffled-edge GNN | CH | -0.2 ± 77.8 | 0.993 | 3 |
| GNN − random Gaussian embedding | Silhouette | +0.167 ± 0.011 | 3.79e-12 | 10 |
| GNN − random Gaussian embedding | DBI | -35.377 ± 5.346 | 6.09e-09 | 10 |
| GNN − random Gaussian embedding | CH | +238.5 ± 17.2 | 8.21e-12 | 10 |
| GNN − B4 A=I+K-Means | Silhouette | -0.031 ± 0.024 | 0.00268 | 1 |
| GNN − B4 A=I+K-Means | DBI | +0.112 ± 0.190 | 0.0964 | 2 |
| GNN − B4 A=I+K-Means | CH | -43.2 ± 30.8 | 0.00166 | 1 |

## 5. Base-paper replication (B0 on the 17 variables, all 8,950 customers, K = 4)

Median imputation, cap at p95, StandardScaler, PCA keeping 95% variance (11 components), all fitted on all 8,950 customers (as a non-inductive paper would), K-Means n_init=20, Ward, spectral with nearest-neighbour affinity (10 neighbours), DBSCAN with min_samples = 2 × dims and eps = 95th percentile of the min_samples-NN distance (the paper does not state its DBSCAN parameters). The paper does not say in which space Table 1 was computed, so both spaces are shown. Nothing was tuned to match. This is *not* the test-split protocol of sections 1-4 (that one fits on train). Source: `replication_b0.csv`; published values from paper Table 9.

| algorithm |  | DBI (diff vs published) | Silhouette (diff) | CH (diff) | clusters found | smallest cluster | noise share |
|---|---|---|---|---|---|---|---|
| K-Means | published | 0.800 | 0.408 | 5,823.343 |  |  |  |
|  | ours: PCA 95% space | 1.563 (+95.4%) | 0.205 (-49.7%) | 2,416.5 (-58.5%) | 4 | 1226 | 0.000 |
|  | ours: capped+standardized 17-var space | 1.617 (+102.1%) | 0.194 (-52.4%) | 2,254.0 (-61.3%) | 4 | 1226 | 0.000 |
| DBSCAN | published | 1.287 | 0.803 | 685.302 |  |  |  |
|  | ours: PCA 95% space | 0.689 (-46.5%) | 0.409 (-49.1%) | 83.3 (-87.8%) | 2 | 19 | 0.008 |
|  | ours: capped+standardized 17-var space | 0.737 (-42.7%) | 0.404 (-49.7%) | 80.1 (-88.3%) | 2 | 19 | 0.008 |
| Hierarchical | published | 0.855 | 0.388 | 4,936.960 |  |  |  |
|  | ours: PCA 95% space | 1.632 (+90.9%) | 0.166 (-57.3%) | 1,870.1 (-62.1%) | 4 | 1358 | 0.000 |
|  | ours: capped+standardized 17-var space | 1.693 (+98.0%) | 0.159 (-59.0%) | 1,754.7 (-64.5%) | 4 | 1358 | 0.000 |
| Spectral | published | 11.334 | 0.004 | 190.856 |  |  |  |
|  | ours: PCA 95% space | 1.596 (-85.9%) | 0.096 (+2310.7%) | 1,744.9 (+814.2%) | 4 | 1519 | 0.000 |
|  | ours: capped+standardized 17-var space | 1.642 (-85.5%) | 0.086 (+2055.5%) | 1,640.6 (+759.6%) | 4 | 1519 | 0.000 |

Differences are stated as they are; no row is dropped.

## 6. Stability and agreement between segmentations

Stability = mean pairwise ARI between the test-set partitions of different seeds (45 pairs; ± = std over pairs). B0 Ward and B0 DBSCAN do not depend on the seed (fitted once, so ARI = 1 by construction, not evidence of robustness). Source: `stability.csv`. Agreement columns: ARI / NMI between the GNN segments and each other method's segments at the same seed, mean ± std over seeds (`cross_method_agreement.csv`). No label exists for credit card, so no other ARI/NMI is reported.

| method | stability (pairwise ARI) | ARI with GNN segments | NMI with GNN segments |
|---|---|---|---|
| B0 K-Means | 0.999 ± 0.001 | 0.143 ± 0.027 | 0.215 ± 0.021 |
| B0 DBSCAN | 1.000 ± 0.000 | 0.001 ± 0.002 | 0.012 ± 0.004 |
| B0 Hierarchical | 1.000 ± 0.000 | 0.151 ± 0.030 | 0.222 ± 0.023 |
| B0 Spectral | 1.000 ± 0.000 | 0.187 ± 0.048 | 0.246 ± 0.042 |
| B1 K-Means | 0.996 ± 0.003 | 0.281 ± 0.071 | 0.356 ± 0.055 |
| B2 GMM | 0.716 ± 0.204 | 0.284 ± 0.059 | 0.388 ± 0.056 |
| B3 AE+K-Means | 0.934 ± 0.029 | 0.237 ± 0.069 | 0.324 ± 0.064 |
| B4 A=I+K-Means | 0.401 ± 0.098 | 0.305 ± 0.109 | 0.376 ± 0.096 |
| GNN VGAE+K-Means | 0.485 ± 0.239 | - | - |
| Control: shuffled-edge GNN | 0.286 ± 0.151 | 0.237 ± 0.098 | 0.296 ± 0.111 |
| Control: random Gaussian embedding | 0.000 ± 0.001 | -0.000 ± 0.001 | 0.002 ± 0.001 |

## 7. Explanations (seed 0, GNN segments)

**Surrogate fidelity.** sklearn HistGradientBoostingClassifier(max_iter=200) predicts the GNN segment from the 17 variables in original units; trained on train, scored on the test split: **macro-F1 = 0.986**, accuracy = 0.986 (n = 1343; test segment sizes [431, 267, 351, 294]). Source: `surrogate_fidelity.json`.

**SHAP top-5 features per segment** (shap permutation explainer, 20 background train customers, 4 permutations, stratified test subsample (<=100 per segment); 400 test customers explained; value = mean |SHAP| of the segment's own class output over that segment's explained customers; sign = mean signed SHAP). Source: `shap_per_segment.csv`.

| segment | n explained | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0 | 100 | installments_purchases (0.379, +) | tenure (0.211, +) | oneoff_purchases (0.132, +) | payments (0.090, +) | purchases_installments_frequency (0.038, +) |
| 1 | 100 | purchases (0.300, +) | oneoff_purchases (0.190, +) | installments_purchases (0.113, +) | cash_advance (0.087, +) | tenure (0.068, +) |
| 2 | 100 | oneoff_purchases (0.301, +) | installments_purchases (0.120, +) | tenure (0.065, +) | purchases_trx (0.015, +) | payments (0.013, +) |
| 3 | 100 | oneoff_purchases (0.463, +) | installments_purchases (0.109, +) | payments (0.052, +) | tenure (0.048, +) | purchases (0.044, +) |

**GNNExplainer neighbour statistic** (`gnnexplainer_nodes.csv`): up to 20 test nodes per segment among nodes with degree >= 10 (so that the top-5 is a strict subset of the neighbours; the earlier audit found the statistic saturated because with k=5 many nodes have exactly 5 neighbours), 100 explainer epochs. For each node: share of its 5 highest-mask neighbours that lie in the node's own segment (top-5), the same for 5 neighbours drawn at random (random-5, mean of 200 draws per node) and the share of all neighbours in its own segment (overall rate).

| segment | nodes | degree | top-5 same-segment share | random-5 baseline | overall same-segment rate | top-5 − random-5 | p (Wilcoxon) vs random-5 | p vs overall |
|---|---|---|---|---|---|---|---|---|
| 0 | 20 | 21-40 | 0.850 ± 0.296 | 0.923 ± 0.189 | 0.923 ± 0.189 | -0.073 | 0.0277 | 0.0277 |
| 1 | 20 | 22-40 | 0.960 ± 0.139 | 0.966 ± 0.105 | 0.967 ± 0.103 | -0.006 | 0.686 | 0.686 |
| 2 | 20 | 20-44 | 0.940 ± 0.131 | 0.929 ± 0.154 | 0.926 ± 0.155 | +0.011 | 0.345 | 0.257 |
| 3 | 20 | 22-49 | 0.980 ± 0.089 | 0.978 ± 0.055 | 0.979 ± 0.057 | +0.002 | 0.715 | 0.715 |
| all | 80 | 20-49 | 0.932 ± 0.185 | 0.949 ± 0.135 | 0.949 ± 0.135 | -0.017 | 0.117 | 0.112 |

**Does the statistic vary across nodes? YES.** top-5 share: std across nodes = 0.185, 6 distinct values, top-5 differs from the all-neighbours share in 24% of nodes; edge-mask std within a node is > 1e-6 in 100% of nodes (median 28 distinct mask values per node). Saturated nodes (degree <= 5): 0.

## 8. Segment profiles (seed-0 GNN, test customers, original units)

Means per segment; all 17 variables are in `segment_profiles_gnn_seed0.csv`. Segment ids are arbitrary K-Means ids.

| segment | n | share | balance | purchases | cash_advance | credit_limit | payments | purchases_frequency | cash_advance_frequency | purchases_trx | prc_full_payment | tenure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 431 | 0.321 | 1,285.09 | 658.61 | 960.05 | 4,015.24 | 1,177.48 | 0.36 | 0.14 | 6.86 | 0.14 | 10.60 |
| 1 | 267 | 0.199 | 2,225.91 | 0.61 | 1,844.65 | 3,907.68 | 1,575.24 | 0.00 | 0.26 | 0.03 | 0.03 | 11.91 |
| 2 | 351 | 0.261 | 2,000.53 | 2,321.23 | 823.53 | 5,782.85 | 2,710.68 | 0.82 | 0.11 | 35.34 | 0.17 | 11.94 |
| 3 | 294 | 0.219 | 828.50 | 600.51 | 391.68 | 3,308.67 | 1,145.36 | 0.69 | 0.05 | 13.76 | 0.24 | 11.90 |

## 9a. Independent recomputation of the indices from the saved labels (step 4)

`check_independent.py` re-derives the features from `unified/cc_full.csv`, implements Silhouette/DBI/CH and the tiny-cluster merge itself (numpy/scipy, no sklearn metrics, no shared code) and recomputes **1200 values** (all 11 methods × 10 seeds × 2 spaces × all/merged × 3 indices). **Maximum absolute difference 0.00024; maximum relative difference 1.43e-06** (worst: Control: random Gaussian embedding, seed 3, plain std, all, ch: reported 1.266944, recomputed 1.266942). NaN in exactly one of the two: 0. Max relative difference by index: ch 1.4e-06, dbi 1.1e-06, silhouette 2.6e-07. The differences are float32 (run) vs float64 (check). Source: `independent_check.csv`, `independent_check_summary.json`.

## 9. Exact seeds, epochs, hardware, runtime

- **Final seeds:** [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (training and clustering seeds; data split fixed by `unified/splits.csv`, split seed 42). Grid seeds: [100, 101]. Control seeds: permutation seed 1000+s, random-embedding seed 2000+s.
- **Epochs:** final models up to 200 epochs, early stopping patience 20 on validation link AUC (evaluated every 5 epochs); grid up to 100 epochs, patience 10. Full-batch training, no neighbour sampling.
- **Epochs actually run / validation link AUC** (`training_info.csv`):

| model | epochs mean (min-max) | val link AUC (mean ± sd) | seconds per model (mean) |
|---|---|---|---|
| B4 | 36 (21-176) | 0.9131 ± 0.0294 | 8 |
| GNN | 180 (146-200) | 0.9872 ± 0.0012 | 53 |
| shuffled-edge GNN | 39 (21-200) | 0.5496 ± 0.0992 | 12 |

- **Config:** k = 20, sage, VGAE, embedding dim 32, hidden 64, Adam lr 0.01, weight decay 5e-4, dropout 0, K-Means n_init 20 fitted on train embeddings; autoencoder B3 lr 0.005, batch 512; GMM full covariance reg_covar 1e-4.
- **Hardware:** 13th Gen Intel Core i7-1355U (10 physical cores, 12 logical processors; `os.cpu_count()` = 12), 15.7 GB RAM, **CPU only** (CUDA available: False), torch 2.14.1+cpu with 12 threads, Python 3.10.11, PYTHONHASHSEED=0, deterministic algorithms on.
- **Runtime:** total 24.2 min. Stage seconds (`run_meta.json`): env 0, data+graphs 2, config-grid 230, B0-replication 52, explain:surrogate+SHAP 93, explain:GNNExplainer 124; per-seed stages summed over seeds: GNN+B4+controls 738, B3+B1+B2 121, B0 68. Seed 0 loop took 120 s; projection after seed 0: 30.3 min. Progress log: `progress.txt`.
- **Graphs** (kNN cosine, union symmetrization, binary weights; `run_meta.json`): k=5: 31,824 edges, mean degree 7.1, components 1; k=10: 62,494 edges, mean degree 14.0, components 1; k=20: 123,556 edges, mean degree 27.6, components 1.

## 10. Figures

- `audit_lean/fig1_segment_profiles.png`: mean of 10 key variables per segment in original units (seed-0 GNN, test customers).
- `audit_lean/fig2_embedding_scatter.png`: 2-D PCA of the seed-0 GNN embedding, test customers colored by segment. The two components explain only about a quarter of the embedding variance (percentages on the axes), so overlap in 2-D is not evidence about separation in the full space; use the indices in section 1.

## Files

`results_lean.csv` (long table of every number above with source file), `results_by_seed.csv`, `labels_all_seeds.npz`, `config_grid_validation.csv`, `grid_by_seed.csv`, `replication_b0.csv`, `stability.csv`, `cross_method_agreement.csv`, `surrogate_fidelity.json`, `shap_per_segment.csv`, `gnnexplainer_nodes.csv`, `segment_profiles_gnn_seed0.csv`, `training_info.csv`, `run_meta.json`, `progress.txt`, `independent_check.csv`, `check_repro.txt`, `check_earlystop.json`.
