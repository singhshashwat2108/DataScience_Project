
# Unified banking dataset: data dictionary

Produced by `harmonize_datasets.py`. Three sources, one schema, one reproducible split (seed 42, 70/15/15).

| Code | Dataset | File used | Rows |
|---|---|---|---|
| `cc` | Credit Card Dataset for Clustering (Kaggle) | `CC GENERAL.csv` | 8,950 |
| `bm` | UCI Bank Marketing | `bank-full.csv` | 45,211 |
| `sa` | Santander Product Recommendation | `train_ver2.csv`, one monthly snapshot (default 2016-05-28), random sample (default 50,000 customers) | configurable |

**Files deliberately not used**
- `bank.csv`: verified to be an exact 10% subset of `bank-full.csv` (4,521 of 4,521 rows found), so it is redundant.
- `bank-additional-*`: has no `balance` column and adds macro-economic indicators (`euribor3m`, `emp.var.rate`, ...) that describe the economy in a given month, not the customer. It also contains 12 duplicate rows.

## Output files

| File | Contents |
|---|---|
| `cc_full.csv`, `bm_full.csv`, `sa_full.csv` | Cleaned, imputed, split. Source-specific features (`cc_*`, `bm_*`, `sa_*`) plus the core columns (`core_*`). |
| `unified_core.csv` | The three sources stacked on the common core schema only. |
| `splits.csv` | `uid, source, split`: single source of truth for the split. |
| `schema.json` | Machine-readable feature lists per source (numeric / binary / categorical), the process-excluded columns and the reference labels. Modeling code should read feature lists from here. |
| `harmonize_report.txt` | Counts, clipped values, imputations and dropped columns from the run. |

Every table carries `uid`, `source`, `orig_id`, `split`, `ref_label_type`, `ref_label`.

## Core schema (the shared structure)

A core column is `NaN` where a source cannot observe the concept (structural missing). It is not imputed.

| Core column | `cc` | `bm` | `sa` |
|---|---|---|---|
| `age` | not available | `age` | `age` (values <18 or >100 set to NaN, then train-median imputed) |
| `tenure_months` | `tenure` (6 to 12) | not available | `antiguedad` (negative sentinel set to NaN) |
| `balance` | `balance` (card balance) | `balance` (average yearly account balance, EUR) | not available |
| `n_products` | count of usage types in {one-off purchases, instalments, cash advance} (0 to 3) | housing loan + personal loan (0 to 2) | sum of the 24 product flags (0 to 24) |
| `is_active` | purchase or cash-advance transactions > 0 | not available | `ind_actividad_cliente` |
| `has_credit_card` | 1 (all customers hold one) | not available | `ind_tjcr_fin_ult1` |
| `has_loan` | not available | `loan` | `ind_pres_fin_ult1` |
| `has_mortgage` | not available | `housing` | `ind_hip_fin_ult1` |
| `has_savings_deposit` | not available | not available | any of savings / short / medium / long-term deposit flags |

`age_q`, `tenure_months_q`, `balance_q`, `n_products_q` are within-source quantile ranks in [0, 1], fitted on the train split of that source only. They make the scales comparable across sources, but not the meaning (see caveat 1).

## Cleaning decisions

| Source | Issue found | Rule applied |
|---|---|---|
| cc | `MINIMUM_PAYMENTS` null in 313 rows; 240 of them have `PAYMENTS = 0` | Those 240 filled with 0 (no payment due); the other 73 filled with the train median; flag column `cc_minimum_payments_missing` kept |
| cc | `CREDIT_LIMIT` null in 1 row | Train median |
| cc | 8 rows with a frequency > 1 (impossible) | Clipped to 1 |
| bm | `pdays = -1` in 36,954 rows (never contacted before) | `bm_pdays` set to NaN, flag `bm_prev_contacted` added |
| bm | `unknown` in job / education / contact / poutcome | Kept as an explicit category |
| sa | Mixed-type fields, `NA` strings, sentinel values (`-999999`), float-formatted codes (`28.0`) | Coerced to numeric or normalized strings |
| sa | `renta` (household income) frequently missing | Train median by province, flag `sa_renta_missing` |
| sa | High-cardinality `canal_entrada`, `cod_prov`, `pais_residencia` | Top categories (fit on train) kept, the rest bucketed to `other` |
| sa | Degenerate columns | Auto-dropped if >=95% null or top category >=99.5% (logged); product flags never dropped |

Imputation values, quantile maps and top-category lists are fitted on the train split only.

## Excluded from the default feature set

- **Bank Marketing campaign-process columns** (`bm_contact`, `bm_month`, `bm_day`, `bm_duration`, `bm_campaign`, `bm_previous`, `bm_prev_contacted`, `bm_pdays`, `bm_poutcome`) describe the marketing call, not the customer. `duration` is only known after the call ends. They stay in the table but are listed under `process_excluded` in `schema.json`.
- **Reference labels** (`ref_label`): `bm_y` (term-deposit response) and Santander `segmento` (the bank's own VIP / individuals / university segmentation). Use them only to evaluate the segments after clustering, never as input features.

## Caveats

1. **These are three different banks with no shared customers.** Pooled clusters would partly reflect which dataset a customer came from. Build the kNN graph within each source; use the pooled `unified_core.csv` for a single reusable pipeline and for comparing results across sources, not for cross-source neighbours.
2. The core columns overlap only partly (see the structural-missing pattern above), so a pooled model on the core alone would use few features per source. Per-source models on the full feature sets are the main experiments.
3. Santander is a monthly snapshot sample. Product history across months (recency-style features) is not included.
