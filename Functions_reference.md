# Project Structure — Function Reference

Technical reference for the What-If Scenario Engine codebase. Lists each file's
core functions and what they do. Pairs with the full write-up (separate document);
this file is for developers (including future-you) who need to find and modify
specific logic quickly.

---

## `features.py`

Shared module. Every other file imports from here — nothing derived should be
recomputed or redefined elsewhere. This is the single source of truth for
feature construction.

- **`load_clean_data(path)`** — loads the cleaned parquet (`v2` by default),
  drops the empty `naew` column, lowercases columns, sorts by
  `product_name`/`month_idx`. Does **not** recompute derived features — trusts
  the parquet as-is.
- **`get_src_cols(d)`** — returns the list of raw district/import column names
  (35 columns), computed by exclusion against a fixed `exclude` list. Used to
  build `feature_cols` and to identify which columns are valid shock targets.
- **`add_derived_features(df, src_cols, imp_cols)`** — recomputes shares
  (`{col}_share`), `import_share`, `domestic_share`, `n_sources`, `herfindahl`,
  `m_sin`/`m_cos` from raw source columns. Defensively drops any stale derived
  columns first, so it's safe to call on already-derived data without creating
  duplicates. **Meant to be called only inside `rebuild()`** — never in the
  normal data-loading path, since `load_clean_data()` already provides a
  parquet with these columns baked in.
- **`add_lags(df, src_cols, imp_cols)`** — adds `avg_price_lag1`,
  `volume_lag1` via `groupby('product_name').shift(1)`. Requires full
  product history in time order; never safe to call on a single hypothetical
  row.
- **`rebuild(df, src_cols, imp_cols)`** — thin wrapper around
  `add_derived_features()`. This is the function the scenario engine calls
  after applying a shock, to recompute shares/`n_sources`/`herfindahl` so they
  reflect the perturbed values rather than stale originals. Does not touch
  lags (a shock doesn't change history).
- **`compute_risk(df, product_vol_stats, scaling_ref)`** — computes
  `volatility`, `concentration` from the row's own data, merges in
  `volume_cv` from `product_vol_stats`, min-max scales all four signals using
  the fixed bounds in `scaling_ref` (not recomputed per call — this is what
  makes it safe to run on a single hypothetical row), averages into
  `risk_score` (skipping missing signals), and bins into Low/Medium/High using
  fixed thresholds (0.177 / 0.319). Formula-based, not a trained model —
  avoids circularity and works identically on real or shocked rows.
- **`cat_cols`** — module-level constant: `['product_name', 'category', 'unit']`.
- **`exclude`** — module-level constant: the column-exclusion list used by
  `get_src_cols()`.

---

## `001feature_ext.py`

One-time data preparation. Run this when the raw source data changes (new
months, corrected values). Produces `market_data_clean_v2.parquet`.

- Loads raw CSV, applies `name_fixes` mapping (product-name typo corrections —
  Dragonfruits→Dragon_Fruits, Gunduruk→Gundruk, Brocauli→Broccoli, etc.).
- Converts `min_price`/`max_price` zeros to NaN (identified as a blanket
  empty-cell fill artifact, not real data) and drops those rows.
- Calls `add_derived_features()` and `add_lags()` **once**, here — this is
  the only place in the normal pipeline where these are computed from scratch.
- Saves the fully-derived result as `v2.parquet`.

## `003risk_features.py`

One-time risk-signal preparation, run after `001`. Produces
`market_data_clean_v3.parquet` plus two small reference files.

- Computes `volatility`, `concentration`, `volume_cv`, `risk_score`, `risk`
  on the full historical dataset using `qcut` (data-driven tertiles — only
  valid for a full batch, not a single row).
- Saves `product_volume_stats.parquet` (per-product mean/std/CV of
  `total_sources`) and `risk_scaling_ref.json` (min/max of each signal,
  used for fixed-bound scaling elsewhere) — these two files are what let
  `compute_risk()` in `features.py` score a single hypothetical row without
  needing the full `v3` dataset.
- `v3` itself is scaffolding for this one-time computation; nothing
  downstream loads it again.

## `002price_surrogate_model_v2.py`

Trains the price model. Time-split (train ≤ month 8, valid = month 9,
test = month 10), Optuna hyperparameter search against valid only, one-time
test evaluation.

- Produces two saved models:
  - `price_surrogate_v1.txt` — train-only, used for all honest evaluation
    (test MAE 24.88 vs persistence 20.33, R²=0.835) and SHAP explanations on
    the test set.
  - `price_surrogate_final.txt` — train+valid+test combined, used inside the
    scenario engine (`run_product_scenario`) for actual predictions, since it
    has learned from the fullest available history.

---

## `scenario_engine004.py`

The core deliverable. Runs hypothetical supply shocks and reports predicted
price/volume/risk response. **To change scenario logic (which columns get
shocked, by how much, which products), edit calls in
`004_scenario_test.ipynb` — the functions below are the stable engine and
shouldn't normally need editing themselves.**

- **`get_product_baseline(d, product, col)`** — returns `(mean_value, total_months, n_sourced)` for a product's history in one source column,
  averaged over only the *nonzero* months. Solves the "0.7 × 0 = 0"
  degenerate-shock problem for products that don't source from every column
  every month. Returns `(None, total_months, 0)` if the product never
  sourced from that column at all.
- **`build_scenario_baseline(d, product, template_month_idx=10)`** — returns
  `(row, month_used)`: the product's month-10 row as the shock template, or
  its most recent available month if month 10 is missing for that product.
- **`confidence_tag(n_sourced)`** — labels a shock result `'reliable'`
  (≥4 sourced months), `'low-confidence'` (2-3), or `'single-observation'`
  (1) — surfaces how much real history backs a given result.
- **`run_product_scenario(d, product, changes, price_booster, feature_cols, cat_cols, product_vol_stats, scaling_ref, src_cols, imp_cols)`** — the
  per-product engine call. Builds the baseline, computes the model's own
  unshocked baseline prediction (`model_baseline_price` — used as the
  denominator for `price_delta_pct`, NOT the real historical price, since
  comparing against actual price would conflate the shock's effect with the
  model's own baseline calibration error), applies the shock, calls
  `rebuild()`, predicts price, computes volume mechanically
  (`total_sources`), computes risk via `compute_risk()`. Returns a dict with
  `status` (`'ok'`, `'no_data_at_all'`, or `'no_applicable_shock'`),
  `coverage`, `confidence`, baseline/shocked price/volume/risk, and deltas.
- **`run_full_scenario(d, changes, **kwargs)` / `run_scenario_report(...)`**
  — loops `run_product_scenario` over all products (or a given subset),
  returns a results DataFrame plus the aggregate summary.
- **`aggregate_summary(results_df)`** — market-wide rollup: count of
  products affected/excluded, mean/median `price_delta_pct`, count of
  products whose risk tier increased (shocked risk > baseline risk).

---

## Notebooks

- **`004_scenario_test.ipynb`** — **the file to edit when running a new or
  modified scenario.** Imports `scenario_engine004.py`, defines the `changes`
  dict (e.g. `{'india': 0.7}`, `{'local': 1.2}`, or combined shocks), calls
  `run_scenario_report(...)`, inspects/plots results. No engine logic lives
  here — only scenario definitions and analysis.
- **`005_shap_analysis.ipynb`** — SHAP artifacts (bar, beeswarm, dependence,
  waterfall, before/after-shock comparison). Uses `price_surrogate_v1` for
  test-set explanations (honest, held-out), `price_surrogate_final` for the
  shock before/after comparison (matches what the scenario engine actually
  uses in production).
- **`006_evaluation_images.ipynb`[REMOVED]** — model evaluation plots (actual vs
  predicted, feature importance, MAE comparison, residuals, error by
  category) and scenario visualizations (sensitivity ranking, import-share
  scatter, shock comparison across scenarios, aggregate comparison).

---

## Known limitations to keep in mind when modifying this code

- **`compute_risk()`'s volatility signal uses the row's *historical*
  min/max/avg price, even for a shocked row** — there's no model that
  predicts shocked min/max, so volatility is intentionally left as a
  slower-moving structural property of the product rather than something
  the shock overrides.
- **The price model shows threshold-driven, not smooth, sensitivity to
  shocks** — because it's tree-based, a shock only changes the prediction if
  it crosses a learned split boundary. Many products show exactly 0% price
  change under a given shock; this is expected model behavior, not a bug.
- **`price_delta_pct` is always model-vs-model** (shocked prediction vs the
  model's own unshocked prediction for the same row) — never model-vs-actual.
  This was a real bug found and fixed mid-project; don't reintroduce a
  comparison against `avg_price` directly when computing deltas.
