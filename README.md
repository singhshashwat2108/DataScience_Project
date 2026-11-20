# Customer Behavior Segmentation and Explainable Decision Analytics for Intelligent Banking

**Do Graph Neural Networks Improve Behavioral Segmentation? A Controlled Evaluation**

Shashwat Singh, Vansh Dhir, Jashesh Rajesh Patel (Vellore Institute of Technology)

This repository holds the code, saved results and papers behind the final project paper. The paper tests whether a behavioral kNN similarity graph and a GraphSAGE + VGAE encoder improve segmentation of credit-card customers compared with graph-free baselines. The result is negative: the GNN segments are less well separated than K-Means, and indistinguishable from a shuffled-edge control.

## Which branch and tag to use

| Ref | Meaning |
|---|---|
| `final_sub` | Clean submission branch (this one): code, results, data needed to run, papers |
| `paper-results` (tag) | Experiment commit `097c5e9`, the code that produced the paper's main-study numbers |
| `fix/lean` | Full development history of the lean experiment, including scratch files removed from `final_sub` |
| `main` | Earlier pipeline, before the controlled evaluation |

The paper (Section 8.1) refers to the code as being on `fix/lean` at commit `097c5e9`. `final_sub` contains the same code (same commit in its history) without the scratch files and the old entry points.

## Headline results (credit-card test split, K = 4, 10 seeds)

| Method | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|
| K-Means (B1) | 0.225 | 1.677 | 344.0 |
| Autoencoder + K-Means (B3) | 0.225 | 1.669 | 332.9 |
| Same encoder, graph removed (B4) | 0.186 | 1.862 | 282.7 |
| **GNN (GraphSAGE + VGAE) + K-Means** | **0.155** | **1.974** | **239.5** |
| GNN with shuffled edges (control) | 0.156 | 2.808 | 239.7 |

Selected configuration: k = 20 neighbors, GraphSAGE. The k = 20 graph has 123,556 edges, mean degree 27.6 and one connected component. The full tables are in the paper (Tables 9 to 12, 16 to 20) and in `audit_lean/results_lean.csv`.

## Repository layout

```
Papers/
  Final_Paper.docx          Final paper
  BasePaper.pdf             Base paper (Kedar et al., 2026)
gnn_segmentation/           Library code: data loading, graph, GNN, evaluation, explanations
  data.py  graph.py  gnn.py  evaluate.py  explain.py  seed.py
audit_lean/                 The experiment that produced the paper's main-study results
  run_lean.py               Driver: full 10-seed credit-card experiment
  make_figures.py           Figures 3 and 4 of the paper
  build_report.py           Assembles RESULTS_LEAN.md and results_lean.csv from the outputs
  check_independent.py      Independent recomputation of the indices (numpy/scipy only)
  check_repro.py            Same-seed determinism check
  check_earlystop.py        Early-stopping check
  results_by_seed.csv, results_lean.csv, stability.csv, ...   Saved results
  labels_all_seeds.npz      Cluster labels for every method and seed
docs/
  harmonize_datasets.py     Cleaning and schema harmonization script
  DATA_DICTIONARY.md        Column descriptions
unified/                    Harmonized credit-card data and the fixed split
  cc_full.csv  schema.json  splits.csv  harmonize_report.txt
requirements.txt
```

## Setup

The paper's results were produced with Python 3.10.11 and a CPU build of PyTorch 2.14.1 on a CPU-only laptop.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`torch_geometric` depends on the installed PyTorch version. If `pip install` fails for it, follow the installation instructions on the PyG website for your PyTorch version.

## Reproducing the main study

Run from the repository root:

```bash
python audit_lean/run_lean.py
```

This runs the full credit-card experiment (validation grid, 10 final seeds, baselines B0 to B4, controls, SHAP, rules, GNNExplainer). It took about 24 minutes on a 13th-generation Intel Core i7 (CPU only). Training is seeded and deterministic: the same seed gives identical embeddings and labels.

The script writes its outputs to `audit_lean/`, so running it overwrites the committed results. To compare, run it in a copy of the repository, or restore the committed files afterwards with `git checkout -- audit_lean`.

Quick check without retraining (recomputes 1,200 index values from the saved labels with an independent implementation):

```bash
python audit_lean/check_independent.py
```

## Data

The main study uses the Kaggle *Credit Card Dataset for Clustering* (Bhasin, 2018; CC0): 8,950 customers and 17 behavioral variables. `unified/cc_full.csv` is the cleaned and harmonized version, and `unified/splits.csv` holds the fixed 70/15/15 customer-level split (seed 42; 6,265 / 1,342 / 1,343). `docs/harmonize_datasets.py` produces these files from the raw datasets.

The Bank Marketing (UCI, CC BY 4.0) and Santander (Kaggle competition) data are not included. Bank Marketing was used only in the supplementary experiment; Santander was prepared but not evaluated.

## Scope and known limitations

- **Supplementary experiment not included.** The graph-free and graph-clustering comparison on credit card and Bank Marketing, the edge-validity checks and the rule-fidelity results (paper Tables 13, 14, 15 and 18, and the rule macro-F1 values in Section 9.9) were produced by a separate script that is **not part of this repository**. The main-study results (Tables 9 to 12, 16, 17, 19, 20) are backed by the saved outputs in `audit_lean/`. <!-- EDIT THIS BULLET if you add or recreate the supplementary script. -->
- **Base-paper replication.** The base paper's published indices were not reproduced; its preprocessing and parameters are not fully stated. All comparisons in the paper are made under one common protocol.
- **Transductive training.** The GNN is trained on the graph of all customers; features (not labels) of validation and test customers take part in message passing.
- **Statistics.** Paired tests over 10 seeds are not corrected for multiple comparisons.

## Reference

Kedar, S., Rathod, K., Bodhak, D., Deshmukh, P. V., & More, A. S. (2026). Explainable machine learning for credit card customer segmentation using clustering algorithms. *Ingénierie des Systèmes d'Information, 31*(1), 261-268. https://doi.org/10.18280/isi.310125
