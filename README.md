# GNN-Based Behavioral Customer Segmentation — Group 16

Implementation of *Customer Behavior Segmentation and Explainable Decision Analytics
for Intelligent Banking* (`Group16_DataScience (1)/Papers/GNN_Project_Paper.pdf`)
on the harmonized dataset `Group16_DataScience (1)/Dataset.csv`.

## How to run

Requirements: Python 3.10 or newer (tested on 3.11). Linux, macOS or Windows. No GPU needed.

### Step 1 — Open a terminal in the project folder

```bash
cd fods-project        # the folder containing this README and gnn_segmentation/
```

The data file must be at `Group16_DataScience (1)/Dataset.csv` (already in place).

### Step 2 — Create a virtual environment (one time)

```bash
python3 -m venv .venv
```

### Step 3 — Install the libraries (one time, ~5 minutes)

```bash
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

On Windows, replace `.venv/bin/` with `.venv\Scripts\` in every command.

Check the install:

```bash
.venv/bin/python -c "import torch, torch_geometric, shap, lightgbm; print('OK')"
```

### Step 4 — Run the live demonstration (~1 minute)

```bash
.venv/bin/python -W ignore -m gnn_segmentation.demo
```

Options:

```bash
.venv/bin/python -W ignore -m gnn_segmentation.demo --source bm   # Bank Marketing
.venv/bin/python -W ignore -m gnn_segmentation.demo --source sa   # Santander (larger, slower)
.venv/bin/python -W ignore -m gnn_segmentation.demo --K 5         # 5 segments instead of 4
```

The demo prints six numbered steps:

1. **Load and preprocess:** customers, features and the train/validation/test split.
2. **Build the kNN graph:** each customer is linked to the 5 most similar customers.
3. **Train the GNN:** GraphSAGE encoder with the VGAE objective, without labels.
4. **Cluster and compare:** K segments, scored against plain K-Means on the raw features.
5. **Segment profiles:** the average customer in each segment.
6. **Explain:** SHAP feature importance from a surrogate model.

It saves the embedding plot to `results/demo/<source>_segments.png`.

### Step 5 — Run the full experiments (optional, long)

This runs E1–E5 from the paper on all three datasets. It includes the baselines,
sensitivity sweeps, explanations, the SQL profiles and the figures.

```bash
.venv/bin/python -W ignore -m gnn_segmentation.run_all --fast   # 3 seeds, quicker
.venv/bin/python -W ignore -m gnn_segmentation.run_all          # 10 seeds, as in the paper (over 1 hour on CPU)
```

To keep it running in the background and follow progress:

```bash
nohup .venv/bin/python -W ignore -m gnn_segmentation.run_all > results/run_log.txt 2>&1 &
tail -f results/run_log.txt
```

The run is finished when the log ends with `done`. All outputs are written to
`results/` (see the table below).

### Step 6 — Look at the results

- Figures: `results/figures/*.png`
- Main comparison table: `results/e2_comparison_summary.csv`
- Segment profiles: `results/segment_profiles.csv`, or query the database:

```bash
sqlite3 results/segments.db "SELECT source, segment, COUNT(*) FROM segments GROUP BY 1, 2;"
```

## Pipeline (paper Table 7)

| Stage | Module | Paper section |
|---|---|---|
| Data preparation (P1 / P2, train-fitted scalers) | `gnn_segmentation/data.py` | 4.2, 5 |
| kNN cosine similarity graph + diagnostics (Algorithm 1) | `gnn_segmentation/graph.py` | 4.3, 7.1 |
| GCN / GraphSAGE encoders, VGAE / DGI objectives, B3 autoencoder, B4 (A = I) | `gnn_segmentation/gnn.py` | 4.4, 7.2–7.3 |
| K-Means / GMM, common-space indices, ARI/NMI, seed stability, B0 pipeline | `gnn_segmentation/evaluate.py` | 4.5, 4.7, 7.4, 7.6 |
| SHAP on a LightGBM surrogate, GNNExplainer | `gnn_segmentation/explain.py` | 4.6, 7.5 |
| SQLite segment store, SQL profiles, figures | `gnn_segmentation/business.py` | 4.8 |
| Experiments E1–E5 (Algorithms 2, 3) | `gnn_segmentation/run_all.py` | 7.7, 8.3 |

## Outputs (`results/`)

| File | Content |
|---|---|
| `e1_base_paper_replication.csv` | B0: K-Means, DBSCAN, hierarchical, spectral on P1 features (credit card) |
| `e2_comparison_summary.csv` / `_all_seeds.csv` | GNN vs B1–B4, test split, mean ± sd over 10 seeds, K = 4 and K* |
| `e2_decision_rule.csv` | Pre-stated rule: GNN must beat B3 and B4 on ≥ 2 of 3 indices |
| `e3_sensitivity.csv` | Encoder × objective, k, embedding dimension (validation) |
| `k_selection.csv` | Validation silhouette and seed stability for K = 2…10 |
| `graph_diagnostics.csv` | Edges, density, degrees, components, label homophily per k |
| `e5_*` | Surrogate fidelity, SHAP drivers per segment, top-3 stability, GNNExplainer results |
| `segments.db`, `segment_profiles.sql`, `segment_profiles.csv` | Business layer |
| `figures/*.png` | Comparison, sensitivity, embeddings, profiles, SHAP, GNNExplainer |
