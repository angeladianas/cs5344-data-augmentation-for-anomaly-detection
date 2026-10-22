"""CS5344: Export Pure GMM V2 and Pure TabDDPM Submission CSVs with LightGBM Boundary Filtering.

Generates submission-ready CSV files meeting all competition specifications:
- NSL-KDD: Exactly 40,000 rows
- UNSW-NB15: Exactly 70,000 rows
- 'id' column as first column (0 to N-1)
- Exact training feature ordering
- No 'is_anomaly' column
- Zero NaNs, nulls, or infinite values
- Correct data types, decimal precision, and domain invariants

Also evaluates local validation utility (S_val, ECOD, IForest) for comparison.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl

from models.gmm_candidates import infer_source_decimal_precision
from scripts.export_competition_submissions import (
    enforce_domain_invariants,
    fit_and_sample_gmm,
    sample_tabddpm_candidates,
)
from utils.boundary_filter import LightGBMBoundaryFilter
from utils.dataset_configs import NSL_KDD, UNSW_NB15
from utils.decoding import TabularDecoder
from utils.evaluator import evaluate_synthetic_utility

DATASET_CONFIGS = {
    "nsl_kdd": {
        "N": 40000,
        "tau_reject": 0.04,
        "gmm_pool": 55000,
        "tab_pool": 60000,
    },
    "unsw_nb15": {
        "N": 70000,
        "tau_reject": 0.05,
        "gmm_pool": 100000,
        "tab_pool": 105000,
    },
}


def build_and_export_pure_model(
    model_type: str,  # 'gmm' or 'tabddpm'
    dataset_name: str,
    seed: int = 42,
) -> tuple[Path, dict]:
    cfg = DATASET_CONFIGS[dataset_name]
    target_n = cfg["N"]
    tau_reject = cfg["tau_reject"]
    pool_size = cfg["gmm_pool"] if model_type == "gmm" else cfg["tab_pool"]

    model_display = "Protocol-Grouped GMM V2" if model_type == "gmm" else "Deep TabDDPM"
    print(f"\n{'=' * 75}")
    print(f"GENERATING PURE SUBMISSION: {dataset_name.upper()} | Model: {model_display}")
    print(f"Target Rows: {target_n:,} | Candidate Pool: {pool_size:,} | Boundary Filter tau <= {tau_reject}")
    print(f"{'=' * 75}\n")

    input_dir = REPO_ROOT / f"output/{dataset_name}/model_inputs"
    decoder = TabularDecoder.from_directory(input_dir)
    config = NSL_KDD if dataset_name == "nsl_kdd" else UNSW_NB15
    train_csv = REPO_ROOT / f"data/{'NSL-KDD' if dataset_name == 'nsl_kdd' else 'UNSW-NB15'}/train.csv"

    # 1. Load trained LightGBM boundary filter
    filter_path = REPO_ROOT / f"results/models/boundary_filter_{dataset_name}.pkl"
    boundary_filter = LightGBMBoundaryFilter.load(filter_path)
    if hasattr(boundary_filter, "model"):
        boundary_filter.model.set_params(n_jobs=1)

    # 2. Sample candidates from the selected pure generator
    t0 = time.time()
    if model_type == "gmm":
        raw_candidates = fit_and_sample_gmm(
            dataset_name=dataset_name,
            decoder=decoder,
            n_samples=pool_size,
            gmm_version="v2",
            seed=seed,
        )
    else:
        raw_candidates = sample_tabddpm_candidates(
            dataset_name=dataset_name,
            n_samples=pool_size,
            decoder=decoder,
            batch_size=2048,
            seed=seed,
        )

    # 3. Score with LightGBM boundary filter and select target N rows
    print(f"\nScoring {raw_candidates.height:,} candidates with LightGBM boundary filter...")
    probs = boundary_filter.predict_anomaly_prob(raw_candidates.to_pandas())
    mask = probs <= tau_reject
    survivors = raw_candidates.filter(pl.Series("s", mask))
    survivors_p = probs[mask]
    print(f"Survivors (tau <= {tau_reject}): {survivors.height:,} / {raw_candidates.height:,} ({mask.mean()*100:.2f}%)")

    if dataset_name == "nsl_kdd":
        if survivors.height >= target_n:
            selected = survivors.sample(n=target_n, seed=seed)
        else:
            print(f"[WARN] Only {survivors.height:,} survivors < {target_n:,}. Selecting top {target_n:,} by lowest anomaly prob.")
            selected = raw_candidates[np.argsort(probs)[:target_n]]
    else:
        # UNSW-NB15: cleanest-rank selection
        if survivors.height >= target_n:
            selected = survivors[np.argsort(survivors_p)[:target_n]]
        else:
            print(f"[WARN] Only {survivors.height:,} survivors < {target_n:,}. Selecting top {target_n:,} by lowest anomaly prob.")
            selected = raw_candidates[np.argsort(probs)[:target_n]]

    # 4. Enforce domain invariants and decimal precision
    cat_cols = list(config.categorical)
    numeric_cols = [c for c in decoder.features if c not in cat_cols]
    precision = infer_source_decimal_precision(train_csv, decoder.features, numeric_cols)
    final_synthetic = enforce_domain_invariants(selected, dataset_name, precision)

    # 5. Format with 'id' column as the first column
    submission = final_synthetic.with_columns(
        pl.arange(0, target_n, dtype=pl.Int64).alias("id")
    ).select(["id", *decoder.features])

    # Guardrail checks
    assert submission.height == target_n, f"Expected {target_n} rows, got {submission.height}"
    assert "is_anomaly" not in submission.columns, "Found is_anomaly in submission!"
    assert submission.columns[0] == "id", "First column is not 'id'!"
    assert submission.null_count().sum_horizontal().sum() == 0, "Null values detected in submission!"

    # 6. Save submission CSV to results/ablation_benchmarks
    out_dir = REPO_ROOT / "results" / "ablation_benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"submission_{dataset_name}_pure_{model_type}.csv"
    submission.write_csv(out_csv)
    print(f"\n[PASSED] Exported Kaggle submission file: {out_csv}")
    print(f"         Rows: {submission.height:,} | Cols: {len(submission.columns)} | Nulls: 0")

    # 7. Evaluate local validation utility
    print("\nEvaluating downstream local benchmark (ECOD + 10-seed Isolation Forest)...")
    val_path = REPO_ROOT / f"output/{dataset_name}/splits/validation.parquet"
    df_val = pl.read_parquet(val_path)
    metrics = evaluate_synthetic_utility(
        synthetic_df=submission.drop("id"),
        validation_df=df_val,
        categorical_cols=cat_cols,
        n_jobs=1,
    )
    print(f"  -> S_val: {metrics['S_val']:.5f} | ECOD: {metrics['AUPRC_ECOD']:.5f} | IForest: {metrics['AUPRC_IForest_mean']:.5f} ({time.time()-t0:.1f}s total)")

    summary = {
        "model_type": model_type,
        "dataset": dataset_name,
        "submission_file": str(out_csv),
        "rows": target_n,
        "tau_reject": tau_reject,
        "survivor_pct": round(float(mask.mean() * 100), 2),
        "S_val": round(float(metrics["S_val"]), 5),
        "AUPRC_ECOD": round(float(metrics["AUPRC_ECOD"]), 5),
        "AUPRC_IForest": round(float(metrics["AUPRC_IForest_mean"]), 5),
    }
    return out_csv, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", "gmm", "tabddpm"], default="all")
    parser.add_argument("--dataset", choices=["all", "nsl_kdd", "unsw_nb15"], default="all")
    args = parser.parse_args()

    models = ["gmm", "tabddpm"] if args.model == "all" else [args.model]
    datasets = ["nsl_kdd", "unsw_nb15"] if args.dataset == "all" else [args.dataset]

    results = []
    for d in datasets:
        for m in models:
            _csv_path, res = build_and_export_pure_model(m, d)
            results.append(res)

    print("\n" + "=" * 80)
    print("ALL REQUESTED PURE MODEL SUBMISSIONS GENERATED SUCCESSFULLY:")
    print("=" * 80)
    for r in results:
        print(f"Dataset: {r['dataset'].upper():<10} | Model: {r['model_type'].upper():<8} | S_val: {r['S_val']:.5f} (ECOD: {r['AUPRC_ECOD']:.5f}, IF: {r['AUPRC_IForest']:.5f}) -> {r['submission_file']}")


if __name__ == "__main__":
    main()
