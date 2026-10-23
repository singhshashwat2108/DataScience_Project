"""Algorithms 2 and 3 end to end: experiments E1-E5, explanations, business layer.

Usage (from the project root):  python -m gnn_segmentation.run_all [--fast]
Outputs go to results/ (tables, figures, segments.db, RESULTS.md).
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from . import business
from .data import load_all, prepare_source, SOURCE_NAMES
from .evaluate import (cluster, evaluate, pairwise_ari, base_paper_pipeline, b0_scores,
                       common_space_scores)
from .explain import surrogate_shap, gnn_explain
from .gnn import train_gnn, train_autoencoder
from .graph import knn_graph, to_edge_index, diagnostics

OUT = "results"
SEEDS = list(range(10))
K_RANGE = range(2, 11)
DEFAULT = dict(k=10, kind="gcn", objective="vgae", dim=32)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)
    return df


def val_silhouette(src, Z, K=4, seed=0):
    lab, _ = cluster(Z, src.idx("train"), K, "kmeans", seed)
    v = src.idx("val")
    return common_space_scores(src.X[v], lab[v])["silhouette"]


# ---------------------------------------------------------------- E3 / config
def select_config(src, graphs, full_sweep):
    """Sensitivity sweep on the validation split (seed 0, K = 4)."""
    rows = []

    def run(k, kind, objective, dim, tag):
        ei = graphs[k]
        Z, info, _ = train_gnn(src.X, ei, src.idx("train"), kind, objective, dim, seed=0)
        r = dict(source=src.name, sweep=tag, k=k, kind=kind, objective=objective, dim=dim,
                 val_link_auc=info["val_auc"], epochs=info["epochs"],
                 val_silhouette_K4=val_silhouette(src, Z))
        log(f"  [{src.name}] {tag}: {r}")
        rows.append(r)
        return r

    for kind in ["gcn", "sage"]:
        for obj in ["vgae", "dgi"]:
            run(DEFAULT["k"], kind, obj, DEFAULT["dim"], "encoder x objective")
    best = max(rows, key=lambda r: r["val_silhouette_K4"])
    cfg = dict(k=best["k"], kind=best["kind"], objective=best["objective"], dim=best["dim"])
    if full_sweep:
        for k in [5, 15, 20, 30]:
            rows.append(run(k, cfg["kind"], cfg["objective"], cfg["dim"], "k"))
        best_k = max([r for r in rows if r["kind"] == cfg["kind"]
                      and r["objective"] == cfg["objective"] and r["dim"] == cfg["dim"]],
                     key=lambda r: r["val_silhouette_K4"])["k"]
        cfg["k"] = best_k
        for dim in [16, 64]:
            rows.append(run(cfg["k"], cfg["kind"], cfg["objective"], dim, "dim"))
        cfg["dim"] = max([r for r in rows if r["kind"] == cfg["kind"]
                          and r["objective"] == cfg["objective"] and r["k"] == cfg["k"]],
                         key=lambda r: r["val_silhouette_K4"])["dim"]
    return cfg, pd.DataFrame(rows)


# ------------------------------------------------------------- representations
def representations(src, graphs, cfg, seeds):
    """Embeddings per seed for the GNN, the graph-ablated encoder (B4) and B3."""
    ei = graphs[cfg["k"]]
    tr, va = src.idx("train"), src.idx("val")
    reps = {"GNN": [], "B4 encoder, A=I": [], "B3 autoencoder": []}
    models, infos = [], []
    for s in seeds:
        Z, info, m = train_gnn(src.X, ei, tr, cfg["kind"], cfg["objective"], cfg["dim"], seed=s)
        reps["GNN"].append(Z)
        models.append(m)
        infos.append({"seed": s, "model": "GNN", "val_auc": info["val_auc"], "epochs": info["epochs"]})
        Zb, ib, _ = train_gnn(src.X, ei, tr, cfg["kind"], cfg["objective"], cfg["dim"],
                              seed=s, ablate=True)
        reps["B4 encoder, A=I"].append(Zb)
        infos.append({"seed": s, "model": "B4", "val_auc": ib["val_auc"], "epochs": ib["epochs"]})
        reps["B3 autoencoder"].append(train_autoencoder(src.X, tr, va, cfg["dim"], seed=s))
        log(f"  [{src.name}] seed {s}: GNN auc={info['val_auc']:.3f} "
            f"B4 auc={ib['val_auc']:.3f}")
    reps["B1/B2 raw features"] = [src.X] * len(seeds)
    return reps, models, pd.DataFrame(infos)


def choose_K(src, reps, seeds):
    """K on validation: common-space silhouette subject to seed stability."""
    rows = []
    v = src.idx("val")
    for K in K_RANGE:
        for method in ["kmeans", "gmm"]:
            parts, sils = [], []
            for s, Z in zip(seeds, reps["GNN"]):
                lab, _ = cluster(Z, src.idx("train"), K, method, s)
                parts.append(lab[v])
                sils.append(common_space_scores(src.X[v], lab[v])["silhouette"])
            stab, _ = pairwise_ari(parts)
            rows.append(dict(source=src.name, K=K, method=method, val_silhouette=np.nanmean(sils),
                             val_silhouette_std=np.nanstd(sils), val_seed_ari=stab))
    df = pd.DataFrame(rows)
    km = df[(df.method == "kmeans") & (df.val_seed_ari >= 0.7)]
    K_star = int((km if len(km) else df[df.method == "kmeans"])
                 .sort_values("val_silhouette", ascending=False).K.iloc[0])
    return K_star, df


def compare(src, reps, seeds, K):
    """E2: test-split indices for the GNN and baselines B1-B4, 10 seeds."""
    te, tr = src.idx("test"), src.idx("train")
    methods = [("GNN + K-Means", "GNN", "kmeans"), ("GNN + GMM", "GNN", "gmm"),
               ("B1 K-Means", "B1/B2 raw features", "kmeans"),
               ("B2 GMM", "B1/B2 raw features", "gmm"),
               ("B3 AE + K-Means", "B3 autoencoder", "kmeans"),
               ("B4 encoder A=I + K-Means", "B4 encoder, A=I", "kmeans")]
    rows, summary, labels_seed0 = [], [], {}
    for name, rep, method in methods:
        parts = []
        for s, Z in zip(seeds, reps[rep]):
            lab, _ = cluster(Z, tr, K, method, s)
            parts.append(lab[te])
            if s == seeds[0]:
                labels_seed0[name] = lab
            rows.append(dict(source=src.name, K=K, method=name, seed=s, **evaluate(src, lab, te)))
        d = pd.DataFrame([r for r in rows if r["method"] == name])
        stab, stab_sd = pairwise_ari(parts)
        summ = dict(source=src.name, K=K, method=name)
        for m in ["silhouette", "dbi", "ch", "ari_ref", "nmi_ref"]:
            summ[m] = d[m].mean()
            summ[m + "_sd"] = d[m].std()
        summ["seed_ari"], summ["seed_ari_sd"] = stab, stab_sd
        summary.append(summ)
    return pd.DataFrame(rows), pd.DataFrame(summary), labels_seed0


def decision_rule(summary):
    """GNN adds value only if it beats B3 and B4 on >= 2 of the 3 common-space indices."""
    out = []
    for (srcn, K), g in summary.groupby(["source", "K"]):
        gnn = g[g.method == "GNN + K-Means"].iloc[0]
        for base in ["B3 AE + K-Means", "B4 encoder A=I + K-Means"]:
            b = g[g.method == base].iloc[0]
            wins = int(gnn.silhouette > b.silhouette) + int(gnn.dbi < b.dbi) + int(gnn.ch > b.ch)
            out.append(dict(source=srcn, K=K, versus=base, indices_won=wins))
        w = [r["indices_won"] for r in out if r["source"] == srcn and r["K"] == K]
        out.append(dict(source=srcn, K=K, versus="DECISION",
                        indices_won=int(all(x >= 2 for x in w))))
    return pd.DataFrame(out)


# --------------------------------------------------------------------- main
def main(fast=False):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    seeds = SEEDS[:3] if fast else SEEDS
    full = load_all()
    all_summ, all_runs, all_K, all_sens, all_diag, all_dec = [], [], [], [], [], []
    all_infos, explain_rows, shap_tabs, gx_tabs, gx_feat, shap_stab = [], [], [], [], [], []
    state = {}

    for name in ["cc", "bm", "sa"]:
        log(f"=== {SOURCE_NAMES[name]} ===")
        src = prepare_source(full, name)
        log(f"  features: {src.features}; P1 PCA dims: {src.X_p1.shape[1]}")

        # Algorithm 1 + diagnostics for the k sweep.
        graphs = {}
        for k in [5, 10, 15, 20, 30]:
            A = knn_graph(src.X, k)
            graphs[k] = to_edge_index(A)
            d = diagnostics(A, src.ref)
            mA = knn_graph(src.X, k, mode="mutual")
            d.update(source=name, k=k, mutual_edges=int(mA.nnz // 2),
                     mutual_components=diagnostics(mA)["components"])
            all_diag.append(d)
        log(f"  graph k=10: {[d for d in all_diag if d['source']==name and d['k']==10][0]}")

        # E3: sensitivity / configuration selection on validation.
        cfg, sens = select_config(src, graphs, full_sweep=(name == "cc"))
        all_sens.append(sens)
        log(f"  selected config: {cfg}")

        reps, models, infos = representations(src, graphs, cfg, seeds)
        infos["source"] = name
        all_infos.append(infos)
        K_star, kdf = choose_K(src, reps, seeds)
        all_K.append(kdf)
        log(f"  K* = {K_star}")

        labels_main = None
        for K in sorted({4, K_star}):
            runs, summ, lab0 = compare(src, reps, seeds, K)
            all_runs.append(runs)
            all_summ.append(summ)
            if K == K_star:
                labels_main = lab0
            log("\n" + summ[["method", "silhouette", "dbi", "ch", "ari_ref", "seed_ari"]]
                .round(3).to_string(index=False))

        # E1: base-paper pipeline on the credit-card data.
        if name == "cc":
            b0 = []
            for K in sorted({4, K_star}):
                res = base_paper_pipeline(src.X_p1, src.idx("train"), K)
                for m, lab in res.items():
                    for space, Xs, idx in [("test, common space", src.X, src.idx("test")),
                                           ("test, P1 space", src.X_p1, src.idx("test")),
                                           ("all rows, P1 space", src.X_p1, np.arange(len(lab)))]:
                        b0.append(dict(K=K, method=m, space=space, **b0_scores(Xs, lab, idx)))
            save(pd.DataFrame(b0), "e1_base_paper_replication.csv")

        # E5: explanations for the selected GNN partition (K*, seed 0).
        lab = labels_main["GNN + K-Means"]
        sh = surrogate_shap(src, lab, seed=0)
        explain_rows.append(dict(source=name, K=K_star, fidelity_macro_f1=sh["fidelity_macro_f1"]))
        t = sh["per_segment"]
        t["source"] = name
        shap_tabs.append(t)
        # Stability of the top-3 SHAP features across seeds.
        tops = []
        for s in seeds[:3]:
            l_s, _ = cluster(reps["GNN"][s], src.idx("train"), K_star, "kmeans", s)
            tops.append(set(surrogate_shap(src, l_s, seed=s)["global"].index[:3]))
        jac = [len(a & b) / len(a | b) for i, a in enumerate(tops) for b in tops[i + 1:]]
        shap_stab.append(dict(source=name, top3_jaccard_mean=float(np.mean(jac))))

        _, km = cluster(reps["GNN"][0], src.idx("train"), K_star, "kmeans", 0)
        gx, gf = gnn_explain(src, models[0], cfg["objective"], km.cluster_centers_,
                             graphs[cfg["k"]], lab, per_segment=10 if fast else 15)
        gx["source"], gf["source"] = name, name
        gx_tabs.append(gx)
        gx_feat.append(gf)
        log(f"  surrogate F1={sh['fidelity_macro_f1']:.3f}; GNNExplainer top-5 same-segment "
            f"share={gx.top_same_segment_share.mean():.3f} vs all neighbours "
            f"{gx.all_same_segment_share.mean():.3f}")

        state[name] = dict(cfg=cfg, K_star=K_star, labels=labels_main,
                           Z=reps["GNN"][0], shap=sh, features=src.features)
        business.write_segments(src, lab, name, OUT)

    summary = save(pd.concat(all_summ), "e2_comparison_summary.csv")
    save(pd.concat(all_runs), "e2_comparison_all_seeds.csv")
    save(decision_rule(summary), "e2_decision_rule.csv")
    save(pd.concat(all_K), "k_selection.csv")
    save(pd.concat(all_sens), "e3_sensitivity.csv")
    save(pd.DataFrame(all_diag), "graph_diagnostics.csv")
    save(pd.concat(all_infos), "training_log.csv")
    save(pd.DataFrame(explain_rows), "e5_surrogate_fidelity.csv")
    save(pd.DataFrame(shap_stab), "e5_shap_top3_stability.csv")
    save(pd.concat(shap_tabs), "e5_shap_per_segment.csv")
    save(pd.concat(gx_tabs), "e5_gnnexplainer_nodes.csv")
    save(pd.concat(gx_feat), "e5_gnnexplainer_feature_masks.csv")
    with open(os.path.join(OUT, "state.pkl"), "wb") as f:
        pickle.dump({k: {kk: vv for kk, vv in v.items() if kk != "shap"} |
                     {"shap_values": v["shap"]["shap_values"],
                      "X_explained": v["shap"]["X_explained"]} for k, v in state.items()}, f)
    json.dump({k: dict(v["cfg"], K_star=v["K_star"]) for k, v in state.items()},
              open(os.path.join(OUT, "selected_config.json"), "w"), indent=2)
    business.profiles_and_figures(OUT, full)
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="3 seeds instead of 10")
    main(ap.parse_args().fast)
