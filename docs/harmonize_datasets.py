#!/usr/bin/env python3
"""
harmonize_datasets.py
=====================
Loads three banking datasets, cleans them, maps them onto ONE common schema,
assigns a reproducible train/val/test split, and writes unified outputs.

  cc : Credit Card Dataset for Clustering (Kaggle, "CC GENERAL.csv")
  bm : UCI Bank Marketing (bank-full.csv, ';' separated)
  sa : Santander Product Recommendation (train_ver2.csv, one monthly snapshot)

Usage (run from the folder that holds the three datasets):
    python harmonize_datasets.py --data-dir . --out-dir unified

Design rules
  * Statistics (medians, quantile maps, top-categories) are fit on the TRAIN
    split only and then applied to val/test -> no leakage.
  * Splits are customer-level (Santander: one snapshot row per customer).
  * Every table carries: uid, source, orig_id, split, ref_label_type, ref_label.
  * "core" columns = harmonized across sources (NaN where a source cannot
    observe the concept). "<src>_*" columns = source-specific features.
  * Reference labels (bm_y, sa_segmento) are for post-hoc evaluation only and
    are listed in schema.json under reference_labels, NOT under features.
"""
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

SEED = 42

SA_PRODUCTS = [
    "ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1", "ind_cder_fin_ult1",
    "ind_cno_fin_ult1", "ind_ctju_fin_ult1", "ind_ctma_fin_ult1", "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1", "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1", "ind_plan_fin_ult1",
    "ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1", "ind_valo_fin_ult1",
    "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1",
]
SA_FEATURE_COLS = [
    "fecha_dato", "ncodpers", "ind_empleado", "pais_residencia", "sexo", "age", "fecha_alta",
    "ind_nuevo", "antiguedad", "indrel", "ult_fec_cli_1t", "indrel_1mes", "tiprel_1mes",
    "indresi", "indext", "conyuemp", "canal_entrada", "indfall", "tipodom", "cod_prov",
    "nomprov", "ind_actividad_cliente", "renta", "segmento",
]
CORE_NUM = ["age", "tenure_months", "balance", "n_products"]
CORE_BIN = ["is_active", "has_credit_card", "has_loan", "has_mortgage", "has_savings_deposit"]

REPORT = []


def log(msg):
    print(msg)
    REPORT.append(msg)


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def extract_zips(root: Path):
    """Extract zips (skipping the huge Santander archive, which is handled separately)."""
    dest = root / "_extracted"
    for _ in range(3):  # handle nested zips
        found = False
        for z in list(root.rglob("*.zip")):
            if "_extracted" in z.parts[:-1] and (dest / z.stem).exists():
                continue
            if re.search(r"train_ver2|santander", z.name, re.I):
                continue
            target = dest / z.stem
            if target.exists():
                continue
            try:
                with zipfile.ZipFile(z) as zf:
                    members = [m for m in zf.namelist() if "__MACOSX" not in m]
                    zf.extractall(target, members=members)
                found = True
            except zipfile.BadZipFile:
                log(f"  ! could not open {z}")
        if not found:
            break


def find_one(root: Path, regex: str, suffixes=(".csv",)):
    pat = re.compile(regex, re.I)
    hits = [p for p in root.rglob("*") if p.is_file() and pat.search(p.name)
            and p.suffix.lower() in suffixes and "__MACOSX" not in p.parts]
    hits.sort(key=lambda p: (len(p.parts), str(p)))
    return hits[0] if hits else None


def locate_santander(root: Path):
    # 1) plain csv or csv.zip named train_ver2
    p = find_one(root, r"^train_ver2\.csv(\.zip)?$", suffixes=(".csv", ".zip"))
    if p:
        return p
    # 2) a Kaggle download zip that contains train_ver2.csv(.zip)
    for z in root.rglob("*.zip"):
        if re.search(r"santander", z.name, re.I):
            try:
                with zipfile.ZipFile(z) as zf:
                    for m in zf.namelist():
                        if re.search(r"train_ver2\.csv(\.zip)?$", m):
                            out = root / "_extracted" / "santander"
                            zf.extract(m, out)
                            return out / m
            except zipfile.BadZipFile:
                pass
    # 3) any csv whose header has ncodpers and product columns (not the test file)
    for c in root.rglob("*.csv"):
        if "test" in c.name.lower():
            continue
        try:
            head = pd.read_csv(c, nrows=0).columns.str.strip()
            if "ncodpers" in head and "ind_cco_fin_ult1" in head:
                return c
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def assign_split(n, strat, ratios, seed):
    idx = np.arange(n)
    strat = None if strat is None else np.asarray(strat)
    if strat is not None and pd.Series(strat).value_counts().min() < 3:
        log("  ! a stratification class has <3 members -> falling back to a plain random split")
        strat = None
    tr, rest = train_test_split(idx, train_size=ratios[0], stratify=strat, random_state=seed)
    vfrac = ratios[1] / (ratios[1] + ratios[2])
    va, te = train_test_split(rest, train_size=vfrac, random_state=seed,
                              stratify=None if strat is None else strat[rest])
    s = np.empty(n, dtype=object)
    s[tr], s[va], s[te] = "train", "val", "test"
    return s


def num(series):
    return pd.to_numeric(series.astype(str).str.strip().replace({"NA": np.nan, "nan": np.nan, "": np.nan}),
                         errors="coerce")


def impute(df, train_mask, num_cols, cat_cols, group_by=None):
    """Median / 'unknown' imputation with TRAIN statistics + <col>_missing flags."""
    group_by = group_by or {}
    for c in num_cols:
        miss = df[c].isna()
        if not miss.any():
            continue
        flag = f"{c}_missing"
        if flag not in df.columns:
            df[flag] = miss.astype(int)
        glob = df.loc[train_mask, c].median()
        if c in group_by:
            gmap = df.loc[train_mask].groupby(group_by[c])[c].median()
            fill = df[group_by[c]].map(gmap).fillna(glob)
        else:
            fill = pd.Series(glob, index=df.index)
        df[c] = df[c].fillna(fill)
    for c in cat_cols:
        df[c] = df[c].fillna("unknown").astype(str).str.strip().replace({"": "unknown", "nan": "unknown"})
    return df


def bucket_rare(df, train_mask, col, top=15):
    keep = set(df.loc[train_mask, col].value_counts().head(top).index)
    df[col] = np.where(df[col].isin(keep), df[col], "other")
    return df


# --------------------------------------------------------------------------- #
# Loaders  (clean only; no statistics here)
# --------------------------------------------------------------------------- #
def load_cc(path):
    d = pd.read_csv(path)
    d.columns = [c.strip().lower() for c in d.columns]
    log(f"[cc] {path.name}: {d.shape[0]} rows x {d.shape[1]} cols")
    freq = [c for c in d.columns if c.endswith("frequency")]
    n_clip = int((d[freq] > 1).any(axis=1).sum())
    d[freq] = d[freq].clip(0, 1)
    log(f"[cc] clipped impossible frequency values (>1) in {n_clip} rows")
    d["cc_minimum_payments_missing"] = d["minimum_payments"].isna().astype(int)
    zero_fill = d["minimum_payments"].isna() & (d["payments"] == 0)
    d.loc[zero_fill, "minimum_payments"] = 0.0
    log(f"[cc] minimum_payments: {int(zero_fill.sum())} nulls were 'no payment made' -> 0; "
        f"{int(d['minimum_payments'].isna().sum())} remaining nulls -> train median")
    out = pd.DataFrame({"orig_id": d["cust_id"].astype(str)})
    out["uid"] = "CC_" + out["orig_id"]
    out["source"] = "cc"
    num_cols = [c for c in d.columns if c not in ("cust_id", "cc_minimum_payments_missing")]
    for c in num_cols:
        out[f"cc_{c}"] = d[c].values
    out["cc_minimum_payments_missing"] = d["cc_minimum_payments_missing"].values
    out["ref_label_type"], out["ref_label"] = np.nan, np.nan
    meta = {"numeric": [f"cc_{c}" for c in num_cols], "binary": ["cc_minimum_payments_missing"],
            "categorical": [], "process_excluded": [], "reference_labels": []}
    return out, meta, None


def load_bm(path):
    d = pd.read_csv(path, sep=";")
    d.columns = [c.strip().lower() for c in d.columns]
    log(f"[bm] {path.name}: {d.shape[0]} rows x {d.shape[1]} cols")
    out = pd.DataFrame({"orig_id": [f"row{i}" for i in range(len(d))]})
    out["uid"] = "BM_" + out["orig_id"]
    out["source"] = "bm"
    out["bm_age"] = d["age"].values
    out["bm_balance"] = d["balance"].values
    for c in ("default", "housing", "loan"):
        out[f"bm_{c}"] = d[c].map({"yes": 1, "no": 0}).values
    for c in ("job", "marital", "education"):
        out[f"bm_{c}"] = d[c].astype(str).values            # 'unknown' kept as a category
    # campaign-process columns: kept in the table, EXCLUDED from default features
    out["bm_contact"] = d["contact"].astype(str).values
    out["bm_month"] = d["month"].astype(str).values
    out["bm_day"] = d["day"].values
    out["bm_duration"] = d["duration"].values
    out["bm_campaign"] = d["campaign"].values
    out["bm_previous"] = d["previous"].values
    out["bm_prev_contacted"] = (d["pdays"] != -1).astype(int).values   # pdays == -1 -> never contacted
    out["bm_pdays"] = d["pdays"].where(d["pdays"] != -1, np.nan).values
    out["bm_poutcome"] = d["poutcome"].astype(str).values
    out["ref_label_type"] = "term_deposit_response"
    out["ref_label"] = d["y"].values
    process = ["bm_contact", "bm_month", "bm_day", "bm_duration", "bm_campaign", "bm_previous",
               "bm_prev_contacted", "bm_pdays", "bm_poutcome"]
    meta = {"numeric": ["bm_age", "bm_balance"], "binary": ["bm_default", "bm_housing", "bm_loan"],
            "categorical": ["bm_job", "bm_marital", "bm_education"],
            "process_excluded": process, "reference_labels": ["ref_label"]}
    return out, meta, d["y"].values


def load_sa(path, snapshot, sample, seed, nrows=None, chunksize=500_000):
    log(f"[sa] reading {path} in chunks (snapshot {snapshot}) ...")
    wanted = set(SA_FEATURE_COLS) | set(SA_PRODUCTS)
    parts, months, total = [], {}, 0
    for ch in pd.read_csv(path, dtype=str, chunksize=chunksize, nrows=nrows,
                          usecols=lambda c: c.strip() in wanted):
        ch.columns = [c.strip() for c in ch.columns]
        total += len(ch)
        for m, n in ch["fecha_dato"].str.strip().value_counts().items():
            months[m] = months.get(m, 0) + int(n)
        parts.append(ch[ch["fecha_dato"].str.strip() == snapshot])
    d = pd.concat(parts, ignore_index=True)
    log(f"[sa] scanned {total:,} rows; months present: {min(months)} .. {max(months)} ({len(months)} months)")
    if d.empty:
        sys.exit(f"[sa] snapshot {snapshot} not found. Re-run with --sa-snapshot one of: {sorted(months)}")
    d = d.drop_duplicates("ncodpers", keep="first").reset_index(drop=True)
    log(f"[sa] snapshot {snapshot}: {len(d):,} unique customers")
    if sample and len(d) > sample:
        d = d.sample(n=sample, random_state=seed).reset_index(drop=True)
        log(f"[sa] random sample of {sample:,} customers (seed {seed})")

    d["age"] = num(d["age"])
    bad_age = int(((d["age"] < 18) | (d["age"] > 100)).sum())
    d.loc[(d["age"] < 18) | (d["age"] > 100), "age"] = np.nan
    d["antiguedad"] = num(d["antiguedad"])
    bad_ten = int((d["antiguedad"] < 0).sum())
    d.loc[d["antiguedad"] < 0, "antiguedad"] = np.nan
    d["renta"] = num(d["renta"])
    log(f"[sa] invalid ages set to NaN: {bad_age}; invalid seniority (<0) set to NaN: {bad_ten}; "
        f"renta missing: {d['renta'].isna().mean():.1%}")
    d["ind_actividad_cliente"] = num(d["ind_actividad_cliente"])
    d["ind_nuevo"] = num(d["ind_nuevo"])
    d["indrel"] = num(d["indrel"])
    d["indrel_1mes"] = d["indrel_1mes"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    d["segmento"] = d["segmento"].astype(str).str.strip().str[:2]
    for c in SA_PRODUCTS:
        d[c] = num(d[c]).fillna(0).clip(0, 1).astype(int)

    cat_src = ["sexo", "ind_empleado", "pais_residencia", "indrel_1mes", "tiprel_1mes", "indresi",
               "indext", "conyuemp", "canal_entrada", "indfall", "tipodom", "cod_prov"]
    for c in cat_src:
        d[c] = (d[c].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
                .replace({"": np.nan, "nan": np.nan, "NA": np.nan}))
    d["segmento"] = d["segmento"].map({"01": "01_top_vip", "02": "02_individuals", "03": "03_university"})
    log("[sa] columns dropped by design: fecha_dato (constant within a snapshot), fecha_alta (superseded by "
        "antiguedad), nomprov (duplicate of cod_prov), ult_fec_cli_1t (only populated for ex-primary customers)")

    out = pd.DataFrame({"orig_id": d["ncodpers"].astype(str).str.strip()})
    out["uid"] = "SA_" + out["orig_id"]
    out["source"] = "sa"
    out["sa_age"] = d["age"].values
    out["sa_antiguedad"] = d["antiguedad"].values
    out["sa_renta"] = d["renta"].values
    out["sa_ind_actividad_cliente"] = d["ind_actividad_cliente"].values
    out["sa_ind_nuevo"] = d["ind_nuevo"].values
    out["sa_indrel"] = d["indrel"].values
    cat_keep = [c for c in cat_src]
    for c in cat_keep:
        out[f"sa_{c}"] = d[c].values
    for c in SA_PRODUCTS:
        out[f"sa_{c}"] = d[c].values
    out["ref_label_type"] = "bank_segment"
    out["ref_label"] = d["segmento"].values

    meta = {"numeric": ["sa_age", "sa_antiguedad", "sa_renta"],
            "binary": ["sa_ind_actividad_cliente", "sa_ind_nuevo"] + [f"sa_{c}" for c in SA_PRODUCTS],
            "categorical": [f"sa_{c}" for c in cat_keep], "process_excluded": [],
            "reference_labels": ["ref_label"]}
    return out, meta, d["segmento"].fillna("na").values


# --------------------------------------------------------------------------- #
# Source-level post-processing (after split)
# --------------------------------------------------------------------------- #
def finish_source(df, meta, name, ratios, seed, strat):
    df = df.copy()
    df["split"] = assign_split(len(df), strat, ratios, seed)
    tr = (df["split"] == "train").values
    counts = df["split"].value_counts().to_dict()
    log(f"[{name}] split sizes: {counts}")

    if name == "sa":
        # rare/high-cardinality buckets, then drop degenerate columns (data-driven)
        for c in ("sa_canal_entrada", "sa_pais_residencia", "sa_cod_prov"):
            df[c] = df[c].fillna("unknown")
            df = bucket_rare(df, tr, c, top=20 if c == "sa_cod_prov" else 10)
        dropped = []
        for c in list(meta["categorical"]) + list(meta["numeric"]) + list(meta["binary"]):
            s = df.loc[tr, c]
            null_share = s.isna().mean()
            top_share = s.value_counts(normalize=True, dropna=True).head(1).sum() if s.notna().any() else 1.0
            if null_share >= 0.95 or top_share >= 0.995:
                dropped.append((c, round(float(null_share), 3), round(float(top_share), 3)))
        # never auto-drop product flags or the two activity flags (keeps the schema stable)
        keep_always = {"sa_ind_actividad_cliente", "sa_ind_nuevo"}
        drop_cols = [c for c, _, _ in dropped if "_ult1" not in c and c not in keep_always]
        for c in drop_cols:
            for k in ("numeric", "binary", "categorical"):
                if c in meta[k]:
                    meta[k].remove(c)
        df = df.drop(columns=drop_cols)
        log(f"[sa] auto-dropped degenerate columns (null>=95% or top-category>=99.5%): "
            f"{[(c, n, t) for c, n, t in dropped if c in drop_cols]}")
        group_by = {"sa_renta": "sa_cod_prov"}
    else:
        group_by = {}

    num_cols = [c for c in meta["numeric"] + meta["binary"] if c in df.columns and df[c].isna().any()]
    cat_cols = [c for c in meta["categorical"] if c in df.columns]
    before = int(df[num_cols].isna().sum().sum()) if num_cols else 0
    df = impute(df, tr, num_cols, cat_cols, group_by=group_by)
    for c in [c for c in df.columns if c.endswith("_missing") and c not in meta["binary"]]:
        meta["binary"].append(c)
    log(f"[{name}] imputed {before} numeric nulls with TRAIN stats; categorical nulls -> 'unknown'")

    # ---- core harmonized columns ----
    core = pd.DataFrame(index=df.index)
    if name == "cc":
        core["age"] = np.nan
        core["tenure_months"] = df["cc_tenure"]
        core["balance"] = df["cc_balance"]
        core["is_active"] = ((df["cc_purchases_trx"] > 0) | (df["cc_cash_advance_trx"] > 0)).astype(float)
        core["n_products"] = ((df["cc_oneoff_purchases"] > 0).astype(int)
                              + (df["cc_installments_purchases"] > 0).astype(int)
                              + (df["cc_cash_advance"] > 0).astype(int)).astype(float)
        core["has_credit_card"] = 1.0
        core["has_loan"] = core["has_mortgage"] = core["has_savings_deposit"] = np.nan
    elif name == "bm":
        core["age"] = df["bm_age"]
        core["tenure_months"] = np.nan
        core["balance"] = df["bm_balance"]
        core["is_active"] = np.nan
        core["n_products"] = (df["bm_housing"] + df["bm_loan"]).astype(float)
        core["has_credit_card"] = np.nan
        core["has_loan"] = df["bm_loan"].astype(float)
        core["has_mortgage"] = df["bm_housing"].astype(float)
        core["has_savings_deposit"] = np.nan
    else:
        p = lambda k: df[f"sa_{k}"]
        core["age"] = df["sa_age"]
        core["tenure_months"] = df["sa_antiguedad"]
        core["balance"] = np.nan
        core["is_active"] = df["sa_ind_actividad_cliente"].astype(float)
        core["n_products"] = df[[f"sa_{c}" for c in SA_PRODUCTS]].sum(axis=1).astype(float)
        core["has_credit_card"] = p("ind_tjcr_fin_ult1").astype(float)
        core["has_loan"] = p("ind_pres_fin_ult1").astype(float)
        core["has_mortgage"] = p("ind_hip_fin_ult1").astype(float)
        core["has_savings_deposit"] = ((p("ind_ahor_fin_ult1") + p("ind_deco_fin_ult1")
                                        + p("ind_deme_fin_ult1") + p("ind_dela_fin_ult1")) > 0).astype(float)
    for c in CORE_NUM:  # within-source quantile scaling, fit on TRAIN only
        col = core[c]
        if col.loc[tr].notna().sum() < 10:
            core[f"{c}_q"] = np.nan
            continue
        qt = QuantileTransformer(n_quantiles=min(1000, int(tr.sum())), output_distribution="uniform",
                                 subsample=10**9, random_state=seed)
        qt.fit(col.loc[tr].to_frame())
        core[f"{c}_q"] = qt.transform(col.to_frame()).ravel()
    df = pd.concat([df, core.add_prefix("core_")], axis=1)
    return df, meta


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default="unified")
    ap.add_argument("--sources", nargs="+", default=["cc", "bm", "sa"], choices=["cc", "bm", "sa"])
    ap.add_argument("--sa-snapshot", default="2016-05-28")
    ap.add_argument("--sa-sample", type=int, default=50000, help="0 = keep all customers in the snapshot")
    ap.add_argument("--sa-nrows", type=int, default=None, help="dev only: read first N rows of the Santander file")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    ratios = a.ratios
    assert abs(sum(ratios) - 1) < 1e-6, "ratios must sum to 1"
    root, out_dir = Path(a.data_dir).resolve(), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_zips(root)

    frames, schema = {}, {"seed": a.seed, "ratios": ratios, "sources": {}}
    for name in a.sources:
        if name == "cc":
            p = find_one(root, r"^cc[ _-]?general\.csv$") or find_one(root, r"cc.*general|credit.*card.*cluster")
            loader = (lambda p=p: load_cc(p)) if p else None
        elif name == "bm":
            p = find_one(root, r"^bank-full\.csv$")
            loader = (lambda p=p: load_bm(p)) if p else None
        else:
            p = locate_santander(root)
            loader = (lambda p=p: load_sa(p, a.sa_snapshot, a.sa_sample, a.seed, a.sa_nrows)) if p else None
        if loader is None:
            log(f"[{name}] NOT FOUND under {root} -> skipped")
            continue
        df, meta, strat = loader()
        df, meta = finish_source(df, meta, name, ratios, a.seed, strat)
        frames[name] = df
        schema["sources"][name] = {"file": str(p), "rows": int(len(df)), **meta}
        df.to_csv(out_dir / f"{name}_full.csv", index=False)

    if not frames:
        sys.exit("No datasets found.")

    keys = ["uid", "source", "orig_id", "split", "ref_label_type", "ref_label"]
    core_cols = [f"core_{c}" for c in CORE_NUM + CORE_BIN + [f"{c}_q" for c in CORE_NUM]]
    unified = pd.concat([f[keys + core_cols] for f in frames.values()], ignore_index=True)
    unified.columns = [c.replace("core_", "", 1) if c.startswith("core_") else c for c in unified.columns]
    unified.to_csv(out_dir / "unified_core.csv", index=False)
    unified[["uid", "source", "split"]].to_csv(out_dir / "splits.csv", index=False)

    schema["keys"] = keys
    schema["core_features"] = {"numeric": CORE_NUM, "numeric_scaled_within_source": [f"{c}_q" for c in CORE_NUM],
                               "binary": CORE_BIN}
    schema["notes"] = ("core columns are NaN where a source cannot observe the concept (structural missing). "
                       "*_q = within-source quantile rank (fit on train). Build graphs WITHIN a source; "
                       "do not treat cross-source neighbours as meaningful (different banks, no shared customers).")
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2))

    log("\n[unified] rows per source x split")
    log(unified.groupby(["source", "split"]).size().unstack(fill_value=0).to_string())
    log("\n[unified] share of non-null values per core column, by source")
    log(unified.groupby("source")[CORE_NUM + CORE_BIN].agg(lambda s: s.notna().mean()).round(2).to_string())
    (out_dir / "harmonize_report.txt").write_text("\n".join(REPORT))
    log(f"\nDone. Files written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
