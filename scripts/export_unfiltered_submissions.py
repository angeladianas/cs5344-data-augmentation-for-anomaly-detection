"""Export Unfiltered Submissions without LightGBM Filtering.

Isolates the exact impact of the LightGBM negative boundary filtering step
by generating unfiltered counterparts of the three core configurations on UNSW-NB15:
1. Pure Deep TabDDPM (Unfiltered) -> exactly 70,000 rows
2. Pure Protocol-Grouped GMM (Unfiltered) -> exactly 70,000 rows
3. 65:35 Hybrid Ensemble (Unfiltered) -> exactly 70,000 rows (45,500 TabDDPM + 24,500 GMM)

All files strictly adhere to Kaggle submission specifications:
- Exactly 70,000 rows
- 'id' column as first column (0 to 69,999)
- Exact feature column ordering matching training data
- No 'is_anomaly' column
- Zero NaNs, nulls, or infinite values
- Domain invariants and decimal precision rounding applied
- Evaluated on local validation split (S_val, ECOD, IForest)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Thread safety settings
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.ensemble import fit_and_sample_gmm, sample_tabddpm_candidates
from utils.dataset_configs import NSL_KDD, UNSW_NB15
from utils.decoding import TabularDecoder
from utils.domain_rules import enforce_domain_invariants, infer_source_decimal_precision
from utils.evaluator import evaluate_synthetic_utility


def generate_unfiltered_submissions(
    dataset_name: str = "unsw_nb15",
    target_n: int = 70000,
    seed: int = 42,
) -> dict[str, dict[str, object]]:
    r"""Generate unfiltered TabDDPM, GMM, and Ensemble datasets and benchmark downstream utility.

    Args:
        dataset_name: Target dataset ('unsw_nb15' or 'nsl_kdd').
        target_n: Number of synthetic rows to generate (default: 70,000).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping model tags to validation metrics summaries.
    """
    print(f"\n{'=' * 80}")
    print(f"GENERATING UNFILTERED SUBMISSIONS (WITHOUT LIGHTGBM FILTERING): {dataset_name.upper()}")
    print(f"Target Rows: {target_n:,} | Seed: {seed}")
    print(f"{'=' * 80}\n")

    input_dir = REPO_ROOT / f"output/{dataset_name}/model_inputs"
    decoder = TabularDecoder.from_directory(input_dir)
    config = NSL_KDD if dataset_name == "nsl_kdd" else UNSW_NB15
    train_csv = (
        REPO_ROOT / f"data/{'NSL-KDD' if dataset_name == 'nsl_kdd' else 'UNSW-NB15'}/train.csv"
    )

    cat_cols = list(config.categorical)
    numeric_cols = [c for c in decoder.features if c not in cat_cols]
    precision = infer_source_decimal_precision(train_csv, decoder.features, numeric_cols)

    val_path = REPO_ROOT / f"output/{dataset_name}/splits/validation.parquet"
    df_val = pl.read_parquet(val_path)

    # -------------------------------------------------------------
    # 1. Sample 70,000 raw candidates from TabDDPM
    # -------------------------------------------------------------
    print(f"\n[1/4] Sampling {target_n:,} candidates from Deep TabDDPM (batch_size=2048)...")
    t0_tab = time.time()
    raw_tab = sample_tabddpm_candidates(
        dataset_name=dataset_name,
        n_samples=target_n,
        decoder=decoder,
        batch_size=2048,
        seed=seed,
    )
    print(f"TabDDPM generation complete: {raw_tab.height:,} rows in {time.time() - t0_tab:.1f}s.")

    # -------------------------------------------------------------
    # 2. Sample 70,000 raw candidates from Protocol-Grouped GMM
    # -------------------------------------------------------------
    print("\n[2/4] Sampling raw candidates from Protocol-Grouped GMM...")
    t0_gmm = time.time()
    raw_gmm = fit_and_sample_gmm(
        dataset_name=dataset_name,
        decoder=decoder,
        n_samples=target_n,
        gmm_version="v2",
        seed=seed,
    )
    print(
        f"Protocol-Grouped GMM generation complete: {raw_gmm.height:,} rows in {time.time() - t0_gmm:.1f}s."
    )

    # -------------------------------------------------------------
    # 3. Create three datasets without LightGBM filtering
    # -------------------------------------------------------------
    n_tab_ensemble = round(0.65 * target_n)  # 45,500
    n_gmm_ensemble = target_n - n_tab_ensemble  # 24,500

    print("\n[3/4] Assembling candidate sets:")
    print(f"  - Pure TabDDPM (Unfiltered): {target_n:,} rows")
    print(f"  - Pure Protocol-Grouped GMM (Unfiltered): {target_n:,} rows")
    print(
        f"  - 65:35 Ensemble (Unfiltered): {n_tab_ensemble:,} TabDDPM + {n_gmm_ensemble:,} GMM = {target_n:,} rows"
    )

    df_pure_tab = raw_tab.slice(0, target_n)
    df_pure_gmm = raw_gmm.slice(0, target_n)

    ensemble_slices = [
        raw_tab.slice(0, n_tab_ensemble),
        raw_gmm.slice(0, n_gmm_ensemble),
    ]
    df_ensemble = pl.concat(ensemble_slices)
    df_ensemble = df_ensemble.sample(fraction=1.0, shuffle=True, seed=seed)

    datasets_to_process = {
        "pure_tabddpm_unfiltered": ("Pure TabDDPM (No Filter)", df_pure_tab),
        "pure_gmm_unfiltered": ("Pure Protocol-Grouped GMM (No Filter)", df_pure_gmm),
        "ensemble_unfiltered": ("65:35 Hybrid Ensemble (No Filter)", df_ensemble),
    }

    # -------------------------------------------------------------
    # 4. Enforce Invariants, Format, Validate, Evaluate, and Save
    # -------------------------------------------------------------
    print(
        "\n[4/4] Applying domain invariants, validating formats, evaluating utility, and exporting CSVs..."
    )
    ablation_dir = REPO_ROOT / "results" / "ablation_benchmarks"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    results_summary: dict[str, dict[str, object]] = {}

    for tag, (display_name, raw_df) in datasets_to_process.items():
        print(f"\n--- Processing: {display_name} ---")
        t_start = time.time()

        clean_df = enforce_domain_invariants(raw_df, dataset_name, precision)

        submission = clean_df.with_columns(
            pl.arange(0, target_n, dtype=pl.Int64).alias("id")
        ).select(["id", *decoder.features])

        # Strict validation assertions
        assert submission.height == target_n, f"Expected {target_n} rows, got {submission.height}"
        assert len(submission.columns) == len(decoder.features) + 1, "Column count mismatch!"
        assert submission.columns[0] == "id", "First column is not 'id'!"
        assert submission.columns[1:] == decoder.features, "Feature column ordering mismatch!"
        assert "is_anomaly" not in submission.columns, "Found is_anomaly in submission!"
        assert submission.null_count().sum_horizontal().sum() == 0, (
            "Null values detected in submission!"
        )

        out_csv_abl = ablation_dir / f"submission_{dataset_name}_{tag}.csv"
        submission.write_csv(out_csv_abl)

        print(f"  [OK] Exported benchmark file: {out_csv_abl}")
        print(
            f"       Rows: {submission.height:,} | Cols: {len(submission.columns)} | Nulls: 0 | Size: {out_csv_abl.stat().st_size / (1024 * 1024):.2f} MB"
        )

        # Evaluate local validation utility
        print("  Evaluating downstream local benchmark (ECOD + 10-seed Isolation Forest)...")
        metrics = evaluate_synthetic_utility(
            synthetic_df=submission.drop("id"),
            validation_df=df_val,
            categorical_cols=cat_cols,
            n_jobs=1,
        )
        elapsed = time.time() - t_start
        print(
            f"  -> S_val: {metrics['S_val']:.5f} | ECOD: {metrics['AUPRC_ECOD']:.5f} | "
            f"IForest: {metrics['AUPRC_IForest_mean']:.5f} ({elapsed:.1f}s)"
        )

        results_summary[tag] = {
            "name": display_name,
            "file": str(out_csv_abl),
            "rows": target_n,
            "filter": "None",
            "S_val": round(float(metrics["S_val"]), 5),
            "ECOD": round(float(metrics["AUPRC_ECOD"]), 5),
            "IForest": round(float(metrics["AUPRC_IForest_mean"]), 5),
        }

    return results_summary


def main() -> None:
    r"""Parse command line arguments and execute unfiltered submissions generation."""
    parser = argparse.ArgumentParser(
        description="Generate unfiltered TabDDPM, GMM, and Ensemble submissions."
    )
    parser.add_argument(
        "--dataset",
        choices=["unsw_nb15", "nsl_kdd"],
        default="unsw_nb15",
        help="Dataset name (default: unsw_nb15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    results = generate_unfiltered_submissions(
        dataset_name=args.dataset,
        seed=args.seed,
    )

    print("\n" + "=" * 80)
    print("UNFILTERED BENCHMARKS COMPLETE:")
    print("=" * 80)
    for tag, res in results.items():
        print(
            f"{res['name']:<40} | S_val: {float(res['S_val']):.5f} | "
            f"ECOD: {float(res['ECOD']):.5f} | IForest: {float(res['IForest']):.5f}"
        )


if __name__ == "__main__":
    main()
