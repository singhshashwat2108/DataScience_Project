"""Lean credit-card experiment (CPU only): 17 cc_* variables, K = 4, test split for all reported numbers.

Usage (from the project root):
    PYTHONHASHSEED=0 python audit_lean/run_lean.py [--seeds 10] [--budget-min 45] [--out audit_lean]
Stages: env -> data/graphs -> config grid (validation only) -> B0 replication -> per-seed loop
(GNN, B4, B3, B1, B2, B0, controls; explanations after seed 0) -> done.
Everything the report needs is written to --out; build_report.py assembles RESULTS_LEAN.md.
"""
import argparse, json, os, platform, sys, time, warnings
sys.path.insert(0, os.getcwd()); warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, torch
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--budget-min", type=float, default=45)
ap.add_argument("--out", default="audit_lean")
ap.add_argument("--grid-seeds", type=int, default=2)
ap.add_argument("--grid-epochs", type=int, default=100)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--expl-nodes", type=int, default=20)
ap.add_argument("--expl-epochs", type=int, default=100)
ap.add_argument("--shap-per-seg", type=int, default=100)   # stratified subsample of test customers, <= 1000 in total
args = ap.parse_args()
OUT = args.out; os.makedirs(OUT, exist_ok=True)
NCORES = os.cpu_count(); torch.set_num_threads(NCORES)

import gnn_segmentation.gnn as G
from gnn_segmentation.data import prepare_cc_full
from gnn_segmentation.evaluate import cluster, evaluate_tiny, common_space_scores, pairwise_ari
from gnn_segmentation.explain import SegmentModel
from gnn_segmentation.graph import knn_graph, to_edge_index, diagnostics
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

T0 = time.time(); PROG = os.path.join(OUT, "progress.txt"); K = 4; DIM = 32
open(PROG, "w").close()
times = {}


def prog(msg):
    line = f"{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}min  {msg}"
    print(line, flush=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")


def stage(name):
    class S:
        def __enter__(s): s.t = time.time(); prog(f"[stage start] {name}")
        def __exit__(s, *a):
            times[name] = times.get(name, 0) + time.time() - s.t
            prog(f"[stage done]  {name}  {time.time() - s.t:.1f}s")
    return S()


def cpu_name():
    try:
        import subprocess
        return subprocess.check_output(["wmic", "cpu", "get", "name"], text=True).split("\n")[1].strip()
    except Exception:
        return platform.processor()


# ------------------------------------------------------------------ 0. environment
with stage("env"):
    import psutil
    meta = dict(cpu=cpu_name(), cores=NCORES, torch_threads=torch.get_num_threads(),
                ram_gb=round(psutil.virtual_memory().total / 2 ** 30, 1), cuda_available=torch.cuda.is_available(),
                device="cpu", torch=torch.__version__, python=platform.python_version(),
                pythonhashseed=os.environ.get("PYTHONHASHSEED"), deterministic=True, K=K, dim=DIM,
                final_epochs=args.epochs, final_patience=20, grid_epochs=args.grid_epochs, grid_patience=10,
                grid_seeds=[100 + i for i in range(args.grid_seeds)], requested_seeds=args.seeds,
                budget_min=args.budget_min, start=time.strftime("%Y-%m-%d %H:%M:%S"))
    prog(f"hardware: {meta['cpu']} | {NCORES} cores | {meta['ram_gb']} GB RAM | CUDA={meta['cuda_available']} | threads={meta['torch_threads']}")

# ------------------------------------------------------------------ 1. data and graphs
with stage("data+graphs"):
    src = prepare_cc_full()
    tr, va, te = src.idx("train"), src.idx("val"), src.idx("test")
    n = len(src.X)
    GRAPHS = {k: to_edge_index(knn_graph(src.X, k)) for k in (5, 10, 20)}
    gdiag = {}
    for k in GRAPHS:
        from gnn_segmentation.graph import knn_graph as _kg
        gdiag[k] = diagnostics(_kg(src.X, k))
        prog(f"graph k={k}: {gdiag[k]}")
    meta["graphs"] = gdiag
    meta["n_train_val_test"] = [len(tr), len(va), len(te)]
    meta["features"] = src.features

# ------------------------------------------------------------------ 2. config grid (validation only)
with stage("config-grid"):
    rows = []
    for k in (5, 10, 20):
        for kind in ("gcn", "sage"):
            parts, sils = [], []
            for gs in meta["grid_seeds"]:
                t = time.time()
                Z, info, _ = G.train_gnn(src.X, GRAPHS[k], tr, kind, "vgae", DIM, seed=gs,
                                         epochs=args.grid_epochs, patience=10)
                lab, _m = cluster(Z, tr, K, "kmeans", gs)
                sc = common_space_scores(src.X[va], lab[va])
                parts.append(lab[va])
                rows.append(dict(k=k, encoder=kind, objective="vgae", dim=DIM, seed=gs, val_silhouette=sc["silhouette"],
                                 val_dbi=sc["dbi"], val_ch=sc["ch"], val_link_auc=info["val_auc"], epochs=info["epochs"],
                                 min_cluster_val=int(np.bincount(lab[va], minlength=K).min()), secs=time.time() - t))
            stab = pairwise_ari(parts)[0]
            for r in rows[-len(meta["grid_seeds"]):]:
                r["val_seed_ari_config"] = stab
            prog(f"grid k={k} {kind}: val_sil={np.mean([r['val_silhouette'] for r in rows[-len(meta['grid_seeds']):]]):.4f} val_ARI={stab:.3f}")
    grid_seeds_df = pd.DataFrame(rows); grid_seeds_df.to_csv(f"{OUT}/grid_by_seed.csv", index=False)
    grid = grid_seeds_df.groupby(["k", "encoder", "objective", "dim"]).agg(
        val_silhouette=("val_silhouette", "mean"), val_silhouette_sd=("val_silhouette", "std"),
        val_dbi=("val_dbi", "mean"), val_ch=("val_ch", "mean"), val_link_auc=("val_link_auc", "mean"),
        epochs=("epochs", "mean"), val_seed_ari=("val_seed_ari_config", "first")).reset_index()
    best = grid.sort_values("val_silhouette", ascending=False).iloc[0]
    grid["chosen"] = (grid.k == best.k) & (grid.encoder == best.encoder)
    grid.to_csv(f"{OUT}/config_grid_validation.csv", index=False)
    CFG = dict(k=int(best.k), kind=str(best.encoder), objective="vgae", dim=DIM)
    meta["chosen_config"] = CFG
    prog(f"chosen config (max mean validation silhouette, validation only): {CFG}")
EI = GRAPHS[CFG["k"]]

# ------------------------------------------------------------------ 3. B0 replication on all 8,950
with stage("B0-replication"):
    X0 = pd.DataFrame(src.X_raw)
    imp = X0.fillna(X0.median()); cap = imp.clip(upper=imp.quantile(0.95), axis=1)
    Z1 = np.nan_to_num(StandardScaler().fit_transform(cap))
    P = PCA(n_components=0.95, svd_solver="full").fit_transform(Z1)
    ms = 2 * P.shape[1]
    d, _ = NearestNeighbors(n_neighbors=ms).fit(P).kneighbors(P)
    eps = float(np.quantile(d[:, -1], 0.95))
    labs = {"K-Means": KMeans(K, n_init=20, random_state=0).fit_predict(P),
            "DBSCAN": DBSCAN(eps=eps, min_samples=ms).fit_predict(P),
            "Hierarchical": AgglomerativeClustering(K, linkage="ward").fit_predict(P),
            "Spectral": SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=10, random_state=0,
                                           assign_labels="cluster_qr").fit_predict(P)}
    rep = []
    for name, lab in labs.items():
        for space, M in (("PCA 95% space", P), ("capped+standardized 17-var space", Z1)):
            r = evaluate_tiny(M, lab, n_total=len(lab))
            rep.append(dict(algorithm=name, space=space, n_customers=len(lab), n_pca_components=P.shape[1],
                            silhouette=r["silhouette_all"], dbi=r["dbi_all"], ch=r["ch_all"],
                            n_clusters=r["n_clusters_all"], min_cluster_size=r["min_cluster_size"],
                            noise_share=r["noise_share"], dbscan_eps=eps if name == "DBSCAN" else np.nan,
                            dbscan_min_samples=ms if name == "DBSCAN" else np.nan))
    pd.DataFrame(rep).to_csv(f"{OUT}/replication_b0.csv", index=False)
    prog("B0 replication (all 8,950, PCA space): " + "; ".join(
        f"{r['algorithm']} sil={r['silhouette']:.3f} dbi={r['dbi']:.3f} ch={r['ch']:.0f}" for r in rep if r["space"].startswith("PCA")))


# ------------------------------------------------------------------ per-seed methods
def km_labels(Z, seed):
    lab, m = cluster(Z, tr, K, "kmeans", seed)
    return lab, m


def b0_partial(Xp1, seed, which):
    """B0 on the train-fitted P1 features: fitted on train, others by 1-NN (DBSCAN: nearest core point)."""
    Xtr, res = Xp1[tr], {}
    if "kmeans" in which:
        res["B0 K-Means"] = KMeans(K, n_init=20, random_state=seed).fit(Xtr).predict(Xp1)
    if "ward" in which:
        res["B0 Hierarchical"] = KNeighborsClassifier(1).fit(Xtr, AgglomerativeClustering(K, linkage="ward").fit_predict(Xtr)).predict(Xp1)
    if "spectral" in which:
        lab = SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=10, random_state=seed,
                                 assign_labels="cluster_qr").fit_predict(Xtr)
        res["B0 Spectral"] = KNeighborsClassifier(1).fit(Xtr, lab).predict(Xp1)
    if "dbscan" in which:
        m_ = 2 * Xp1.shape[1]
        dd, _ = NearestNeighbors(n_neighbors=m_).fit(Xtr).kneighbors(Xtr)
        e_ = float(np.quantile(dd[:, -1], 0.95))
        db = DBSCAN(eps=e_, min_samples=m_).fit(Xtr)
        core = db.core_sample_indices_
        dist, j = NearestNeighbors(n_neighbors=1).fit(Xtr[core]).kneighbors(Xp1)
        lab = db.labels_[core][j.ravel()]; lab[dist.ravel() > e_] = -1
        res["B0 DBSCAN"] = lab
    return res


METHODS = ["B0 K-Means", "B0 DBSCAN", "B0 Hierarchical", "B0 Spectral", "B1 K-Means", "B2 GMM",
           "B3 AE+K-Means", "B4 A=I+K-Means", "GNN VGAE+K-Means",
           "Control: shuffled-edge GNN", "Control: random Gaussian embedding"]
LABELS, ROWS, TRAIN_INFO, CROSS = {}, [], [], []
b0_cache = {}
seeds_to_run = list(range(args.seeds)); si = 0
model0 = km0 = Z0 = None
while si < len(seeds_to_run):
    s = seeds_to_run[si]; t_seed = time.time(); L = {}
    with stage(f"seed{s}:GNN+B4+controls"):
        t = time.time()
        Zg, ig, mg = G.train_gnn(src.X, EI, tr, CFG["kind"], "vgae", DIM, seed=s, epochs=args.epochs, patience=20)
        L["GNN VGAE+K-Means"], kmg = km_labels(Zg, s)
        TRAIN_INFO.append(dict(seed=s, model="GNN", val_link_auc=ig["val_auc"], epochs=ig["epochs"], secs=time.time() - t)); t = time.time()
        Zb, ib, _ = G.train_gnn(src.X, EI, tr, CFG["kind"], "vgae", DIM, seed=s, ablate=True, epochs=args.epochs, patience=20)
        L["B4 A=I+K-Means"], _ = km_labels(Zb, s)
        TRAIN_INFO.append(dict(seed=s, model="B4", val_link_auc=ib["val_auc"], epochs=ib["epochs"], secs=time.time() - t)); t = time.time()
        perm = torch.as_tensor(np.random.default_rng(1000 + s).permutation(n))
        Zs, is_, _ = G.train_gnn(src.X, perm[EI], tr, CFG["kind"], "vgae", DIM, seed=s, epochs=args.epochs, patience=20)
        L["Control: shuffled-edge GNN"], _ = km_labels(Zs, s)
        TRAIN_INFO.append(dict(seed=s, model="shuffled-edge GNN", val_link_auc=is_["val_auc"], epochs=is_["epochs"], secs=time.time() - t))
        L["Control: random Gaussian embedding"], _ = km_labels(
            np.random.default_rng(2000 + s).standard_normal((n, DIM)).astype(np.float32), s)
    with stage(f"seed{s}:B3+B1+B2"):
        Za = G.train_autoencoder(src.X, tr, va, DIM, seed=s, epochs=args.epochs, patience=20)
        L["B3 AE+K-Means"], _ = km_labels(Za, s)
        L["B1 K-Means"], _ = km_labels(src.X, s)
        L["B2 GMM"], _ = cluster(src.X, tr, K, "gmm", s)
    with stage(f"seed{s}:B0"):
        if not b0_cache:   # Ward and DBSCAN do not depend on the seed: fit once
            b0_cache.update(b0_partial(src.X_p1, s, ("ward", "dbscan")))
        L.update(b0_cache); L.update(b0_partial(src.X_p1, s, ("kmeans", "spectral")))
    for m in METHODS:
        LABELS[(m, s)] = L[m].astype(np.int16)
        a = evaluate_tiny(src.X[te], L[m][te], n_total=len(te))
        b = evaluate_tiny(src.X_std[te], L[m][te], n_total=len(te))
        ROWS.append(dict(method=m, seed=s, **a, **{f"std_{k_}": v for k_, v in b.items() if k_.startswith(("silhouette", "dbi", "ch"))}))
    for m in METHODS:
        if m != "GNN VGAE+K-Means":
            CROSS.append(dict(seed=s, other=m, ari=adjusted_rand_score(L["GNN VGAE+K-Means"][te], L[m][te]),
                              nmi=normalized_mutual_info_score(L["GNN VGAE+K-Means"][te], L[m][te])))
    if s == seeds_to_run[0]:
        model0, km0, Z0 = mg, kmg, Zg
        seed_secs = time.time() - t_seed
        np.save(f"{OUT}/embedding_seed0_gnn.npy", Zg)
        # ---------- explanations (seed 0, GNN)
        with stage("explain:surrogate+SHAP"):
            import shap
            lab0 = L["GNN VGAE+K-Means"]
            Xr = pd.DataFrame(src.X_raw, columns=src.features)
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xr.iloc[tr], lab0[tr])
            f1 = f1_score(lab0[te], clf.predict(Xr.iloc[te]), average="macro")
            acc = float((clf.predict(Xr.iloc[te]) == lab0[te]).mean())
            rng = np.random.default_rng(0)
            sel = np.concatenate([rng.choice(te[lab0[te] == c], size=min(args.shap_per_seg, int((lab0[te] == c).sum())), replace=False)
                                  for c in range(K) if (lab0[te] == c).any()])
            bg = Xr.iloc[rng.choice(tr, 20, replace=False)]
            ex = shap.Explainer(clf.predict_proba, shap.maskers.Independent(bg, max_samples=20), algorithm="permutation")
            sv = ex(Xr.iloc[sel], max_evals=2 * len(src.features) * 4 + 1, silent=True).values   # (n, d, K)
            shap_rows = []
            for c in range(K):
                m_ = lab0[sel] == c
                if m_.sum() == 0:
                    continue
                msv = np.abs(sv[m_, :, c]).mean(0)
                for j, f in enumerate(src.features):
                    shap_rows.append(dict(segment=c, n_explained=int(m_.sum()), feature=f, mean_abs_shap=msv[j],
                                          mean_shap=sv[m_, j, c].mean()))
            pd.DataFrame(shap_rows).to_csv(f"{OUT}/shap_per_segment.csv", index=False)
            json.dump(dict(fidelity_macro_f1_test=f1, accuracy_test=acc, n_test=len(te), n_shap=len(sel),
                           shap_method="shap permutation explainer, 20 background train customers, 4 permutations, stratified test subsample (<=100 per segment)",
                           surrogate="sklearn HistGradientBoostingClassifier(max_iter=200)",
                           test_segment_sizes=np.bincount(lab0[te], minlength=K).tolist(),
                           train_segment_sizes=np.bincount(lab0[tr], minlength=K).tolist()),
                      open(f"{OUT}/surrogate_fidelity.json", "w"), indent=2)
            prog(f"surrogate macro-F1 (test) = {f1:.3f}, acc = {acc:.3f}")
        with stage("explain:GNNExplainer"):
            seg_model = SegmentModel(model0, "vgae", km0.cluster_centers_).eval()
            xt = torch.tensor(src.X, dtype=torch.float32)
            expl = Explainer(model=seg_model, algorithm=GNNExplainer(epochs=args.expl_epochs), explanation_type="model",
                             node_mask_type="attributes", edge_mask_type="object",
                             model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"))
            deg = np.bincount(EI[1].numpy(), minlength=n)
            rng = np.random.default_rng(0); erow = []; TOP = 5
            for c in range(K):
                cand = te[lab0[te] == c]
                elig = cand[deg[cand] >= 2 * TOP]
                if len(elig) < args.expl_nodes:   # relax: need strictly more than TOP neighbours so top-5 != all
                    elig = cand[deg[cand] > TOP]
                pick = rng.choice(elig, size=min(args.expl_nodes, len(elig)), replace=False) if len(elig) else []
                for i in pick:
                    subset, sub_ei, mapping, _ = k_hop_subgraph(int(i), 2, EI, relabel_nodes=True)
                    e = expl(xt[subset], sub_ei, index=int(mapping))
                    em = e.edge_mask.detach().numpy()
                    into = (sub_ei[1] == mapping).numpy()
                    nbr = subset[sub_ei[0][into]].numpy(); imp_ = em[into]
                    order = np.argsort(-imp_, kind="stable")[:TOP]
                    same = (lab0[nbr] == c).astype(float)
                    rnd = np.mean([same[rng.choice(len(nbr), TOP, replace=False)].mean() for _ in range(200)])
                    erow.append(dict(segment=c, node=int(i), degree=len(nbr), top5_same=same[order].mean(),
                                     random5_same=rnd, all_same=same.mean(), mask_std=float(imp_.std()),
                                     mask_n_unique=int(len(np.unique(imp_.round(6)))), top5_mask_mean=float(imp_[order].mean())))
                prog(f"  explainer segment {c}: {len(pick)} nodes (eligible {len(elig)} of {len(cand)} test nodes)")
            pd.DataFrame(erow).to_csv(f"{OUT}/gnnexplainer_nodes.csv", index=False)
        # ---------- extrapolation after seed 0
        elapsed = time.time() - T0
        projected = elapsed + seed_secs * (args.seeds - 1) + 120
        meta["seed0_secs"] = seed_secs; meta["projected_total_min_after_seed0"] = projected / 60
        prog(f"seed0 loop = {seed_secs:.0f}s; elapsed = {elapsed / 60:.1f}min; projected total for {args.seeds} seeds = {projected / 60:.1f}min (budget {args.budget_min}min)")
        if projected / 60 > args.budget_min and args.seeds > 5:
            seeds_to_run = seeds_to_run[:5]; meta["seeds_reduced_to"] = 5
            prog("PROJECTION EXCEEDS BUDGET -> REDUCING TO 5 SEEDS (0-4)")
    prog(f"seed {s} done in {time.time() - t_seed:.0f}s")
    si += 1

# ------------------------------------------------------------------ outputs
with stage("write-outputs"):
    pd.DataFrame(ROWS).to_csv(f"{OUT}/results_by_seed.csv", index=False)
    pd.DataFrame(TRAIN_INFO).to_csv(f"{OUT}/training_info.csv", index=False)
    pd.DataFrame(CROSS).to_csv(f"{OUT}/cross_method_agreement.csv", index=False)
    np.savez_compressed(f"{OUT}/labels_all_seeds.npz", **{f"{m}__seed{s}": v for (m, s), v in LABELS.items()},
                        test_idx=te, train_idx=tr, val_idx=va)
    # stability: mean pairwise ARI between seeds (test labels)
    stab = []
    S = seeds_to_run
    for m in METHODS:
        parts = [LABELS[(m, s)][te] for s in S]
        mu, sd = pairwise_ari(parts)
        stab.append(dict(method=m, n_seeds=len(S), n_pairs=len(S) * (len(S) - 1) // 2, mean_pairwise_ari=mu, sd_pairwise_ari=sd))
    pd.DataFrame(stab).to_csv(f"{OUT}/stability.csv", index=False)
    # segment profiles for seed-0 GNN (test customers, original units)
    lab0 = LABELS[("GNN VGAE+K-Means", S[0])].astype(int)
    prof = pd.DataFrame(src.X_raw[te], columns=src.features).assign(segment=lab0[te]).groupby("segment").mean()
    prof.insert(0, "n_test", np.bincount(lab0[te], minlength=K)); prof.insert(1, "share", prof.n_test / len(te))
    prof.reset_index().to_csv(f"{OUT}/segment_profiles_gnn_seed0.csv", index=False)
    meta["seeds_run"] = S; meta["stage_secs"] = times; meta["total_min"] = (time.time() - T0) / 60
    meta["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(meta, open(f"{OUT}/run_meta.json", "w"), indent=2, default=str)
prog(f"ALL DONE in {(time.time() - T0) / 60:.1f} min; seeds run = {seeds_to_run}")
