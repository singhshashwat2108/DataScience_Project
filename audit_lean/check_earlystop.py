"""Early stopping with vs without held-out edges in the message-passing graph (legacy vs fixed).
Same seed, same everything else. Reports validation link AUC, stopping epoch, and how much the final
embeddings / K=4 segments change. Usage: python audit_lean/check_earlystop.py OUT.json"""
import sys, os, json, warnings
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")
import numpy as np, torch
torch.set_num_threads(os.cpu_count())
from sklearn.metrics import adjusted_rand_score
import gnn_segmentation.gnn as G
from gnn_segmentation.data import prepare_cc_full
from gnn_segmentation.graph import knn_graph, to_edge_index
from gnn_segmentation.evaluate import cluster

src = prepare_cc_full(); tr = src.idx("train"); rows = []
ei = to_edge_index(knn_graph(src.X, 5))
for kind in ("gcn", "sage"):
    for seed in (0, 1, 2):
        Zo, io, _ = G.train_gnn(src.X, ei, tr, kind, "vgae", 32, seed=seed, legacy_val_graph=True)
        Zn, inn, _ = G.train_gnn(src.X, ei, tr, kind, "vgae", 32, seed=seed)
        lo, ln = cluster(Zo, tr, 4, "kmeans", seed)[0], cluster(Zn, tr, 4, "kmeans", seed)[0]
        r = dict(kind=kind, seed=seed, val_auc_legacy=io["val_auc"], val_auc_fixed=inn["val_auc"],
                 auc_inflation=io["val_auc"] - inn["val_auc"], epochs_legacy=io["epochs"], epochs_fixed=inn["epochs"],
                 emb_max_abs_diff=float(np.abs(Zo - Zn).max()), emb_identical=bool(np.array_equal(Zo, Zn)),
                 ari_labels_legacy_vs_fixed=float(adjusted_rand_score(lo, ln)))
        print(r, flush=True); rows.append(r)
json.dump(rows, open(sys.argv[1], "w"), indent=2)
