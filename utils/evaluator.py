"""Downstream Anomaly Detection Utility Evaluator (CS5344 Benchmark).

Emulates the official Kaggle evaluation scheme:
1. One-hot encodes all categorical columns.
2. Trains ECOD (deterministic tail detector) strictly on synthetic normal data.
3. Trains 10-seed Isolation Forest (500 trees each) strictly on synthetic normal data.
4. Scores validation samples and computes Area Under the Precision-Recall Curve (AUPRC).
5. Returns the primary composite leaderboard score S = 0.5 * (AUPRC_ECOD + AUPRC_IForest).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder

from utils.dataset_configs import NSL_KDD, UNSW_NB15, DatasetConfig

logger = logging.getLogger(__name__)

DEFAULT_IFOREST_SEEDS: tuple[int, ...] = (
    42,
    101,
    202,
    303,
    404,
    505,
    606,
    707,
    808,
    909,
)


def _to_pandas(df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame is converted to Pandas without requiring pyarrow."""
    if isinstance(df, pl.DataFrame):
        try:
            return df.to_pandas()
        except ImportError:
            return pd.DataFrame(df.to_dict(as_series=False))
    return df.copy()


def prepare_detector_features(
    df: pd.DataFrame | pl.DataFrame,
    categorical_cols: list[str] | tuple[str, ...],
    encoder: OneHotEncoder | None = None,
) -> tuple[np.ndarray, OneHotEncoder]:
    r"""Prepare one-hot encoded and numerical feature matrices for PyOD detectors.

    Args:
        df: Input DataFrame (synthetic candidates or validation set).
        categorical_cols: Names of categorical columns to encode.
        encoder: Fitted OneHotEncoder. If None, fits a new encoder on df[categorical_cols].

    Returns:
        tuple of (X_matrix as float64 np.ndarray, fitted OneHotEncoder).
    """
    pdf = _to_pandas(df)
    drop_cols = [c for c in ("id", "is_anomaly", "source_row") if c in pdf.columns]
    if drop_cols:
        pdf = pdf.drop(columns=drop_cols)

    cat_cols = list(categorical_cols)
    num_cols = [c for c in pdf.columns if c not in cat_cols]

    # Handle categorical columns
    if cat_cols:
        cat_data = pdf[cat_cols].astype(str)
        if encoder is None:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            cat_encoded = encoder.fit_transform(cat_data)
        else:
            cat_encoded = encoder.transform(cat_data)
    else:
        if encoder is None:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_encoded = np.empty((len(pdf), 0), dtype=np.float64)

    # Handle numerical columns
    if num_cols:
        num_encoded = pdf[num_cols].to_numpy(dtype=np.float64)
    else:
        num_encoded = np.empty((len(pdf), 0), dtype=np.float64)

    if cat_encoded.shape[1] > 0 and num_encoded.shape[1] > 0:
        X = np.ascontiguousarray(np.hstack([num_encoded, cat_encoded]), dtype=np.float64)
    elif cat_encoded.shape[1] > 0:
        X = np.ascontiguousarray(cat_encoded, dtype=np.float64)
    else:
        X = np.ascontiguousarray(num_encoded, dtype=np.float64)

    return X, encoder


def evaluate_ecod(
    X_syn: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    contamination: float = 0.10,
    n_jobs: int = -1,
) -> tuple[float, np.ndarray]:
    r"""Train deterministic ECOD detector on synthetic normal data and score validation set.

    Args:
        X_syn: One-hot encoded feature matrix of synthetic normal records.
        X_val: One-hot encoded feature matrix of validation records.
        y_val: Ground truth binary anomaly labels for validation records (1 = attack).
        contamination: ECOD expected contamination fraction.
        n_jobs: Number of CPU workers for parallel execution.

    Returns:
        tuple of (auprc_ecod, validation_anomaly_scores).
    """
    ecod = ECOD(contamination=contamination, n_jobs=n_jobs)
    ecod.fit(X_syn)
    scores = ecod.decision_function(X_val)
    auprc = float(average_precision_score(y_val, scores))
    return auprc, scores


def evaluate_iforest(
    X_syn: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seeds: tuple[int, ...] | list[int] = DEFAULT_IFOREST_SEEDS,
    n_estimators: int = 500,
    contamination: float = 0.10,
    n_jobs: int = -1,
) -> tuple[float, list[float], np.ndarray]:
    r"""Train 10-seed Isolation Forest ensemble on synthetic data and score validation set.

    Args:
        X_syn: One-hot encoded feature matrix of synthetic normal records.
        X_val: One-hot encoded feature matrix of validation records.
        y_val: Ground truth binary anomaly labels for validation records (1 = attack).
        seeds: Collection of random seeds for the 10 fixed Isolation Forests.
        n_estimators: Number of trees per Isolation Forest (500 trees).
        contamination: Contamination parameter for PyOD IForest.
        n_jobs: Number of parallel jobs.

    Returns:
        tuple of (mean_auprc, list_of_seed_auprcs, mean_validation_anomaly_scores).
    """
    seed_auprcs: list[float] = []
    all_scores: list[np.ndarray] = []

    for seed in seeds:
        iforest = IForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=seed,
            behaviour="new",
            n_jobs=n_jobs,
        )
        iforest.fit(X_syn)
        scores = iforest.decision_function(X_val)
        auprc = float(average_precision_score(y_val, scores))
        seed_auprcs.append(auprc)
        all_scores.append(scores)

    mean_auprc = float(np.mean(seed_auprcs))
    mean_scores = np.mean(all_scores, axis=0)
    return mean_auprc, seed_auprcs, mean_scores


def evaluate_synthetic_utility(
    synthetic_df: pd.DataFrame | pl.DataFrame,
    validation_df: pd.DataFrame | pl.DataFrame,
    categorical_cols: list[str] | tuple[str, ...],
    target_col: str = "is_anomaly",
    seeds: tuple[int, ...] | list[int] = DEFAULT_IFOREST_SEEDS,
    contamination: float = 0.10,
    n_jobs: int = -1,
) -> dict[str, Any]:
    r"""Calculate downstream anomaly detection utility of synthetic normal data.

    Evaluates candidate normal data using the exact Kaggle competition scoring protocol:
    S = 0.5 * (AUPRC_ECOD + AUPRC_IForest).

    Args:
        synthetic_df: Generated normal dataset (must NOT contain target_col).
        validation_df: Ground truth labeled evaluation split (contains target_col).
        categorical_cols: Categorical features to one-hot encode.
        target_col: Binary label column in validation_df (0 = normal, 1 = attack).
        seeds: Random seeds for the 10-seed Isolation Forest ensemble.
        contamination: Prior contamination assumption for PyOD detectors.
        n_jobs: CPU workers for parallel execution.

    Returns:
        dict containing:
            - S_val: Composite utility score.
            - AUPRC_ECOD: Score achieved by ECOD.
            - AUPRC_IForest_mean: Average score achieved by 10-seed Isolation Forest.
            - AUPRC_IForest_seeds: Individual scores per random seed.
            - n_synthetic_rows: Row count of evaluated synthetic data.
            - n_validation_rows: Row count of validation data.
            - feature_dim: Dimensionality of one-hot encoded detector input.
            - duration_seconds: Total benchmark evaluation wall-clock time.
    """
    start_time = time.time()
    val_pdf = _to_pandas(validation_df)

    if target_col not in val_pdf.columns:
        raise ValueError(f"Target column '{target_col}' not found in validation_df")

    y_val = val_pdf[target_col].to_numpy(dtype=np.int64)
    unique_labels = set(np.unique(y_val))
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Expected binary labels {{0, 1}}, got {unique_labels}")

    # Step 1: Feature encoding
    # Encoder is fitted strictly on synthetic data, matching Kaggle protocol
    X_syn, encoder = prepare_detector_features(synthetic_df, categorical_cols, encoder=None)
    X_val, _ = prepare_detector_features(val_pdf, categorical_cols, encoder=encoder)

    if not np.isfinite(X_syn).all() or not np.isfinite(X_val).all():
        raise ValueError("Non-finite values encountered in detector feature matrices")

    # Step 2: Evaluate ECOD (deterministic)
    auprc_ecod, _scores_ecod = evaluate_ecod(
        X_syn, X_val, y_val, contamination=contamination, n_jobs=n_jobs
    )

    # Step 3: Evaluate Isolation Forest (10 fixed seeds)
    auprc_iforest, seed_auprcs, _scores_iforest = evaluate_iforest(
        X_syn,
        X_val,
        y_val,
        seeds=seeds,
        contamination=contamination,
        n_jobs=n_jobs,
    )

    # Step 4: Compute composite leaderboard score
    s_val = 0.5 * (auprc_ecod + auprc_iforest)
    elapsed = time.time() - start_time

    return {
        "S_val": s_val,
        "AUPRC_ECOD": auprc_ecod,
        "AUPRC_IForest_mean": auprc_iforest,
        "AUPRC_IForest_seeds": seed_auprcs,
        "n_synthetic_rows": len(synthetic_df),
        "n_validation_rows": len(validation_df),
        "feature_dim": X_syn.shape[1],
        "duration_seconds": round(elapsed, 2),
    }


def format_evaluation_summary(results: dict[str, Any], title: str = "Benchmark Results") -> str:
    """Format evaluation metrics into an ASCII summary report."""
    border = "=" * 55
    lines = [
        border,
        f"  CS5344 Downstream Utility: {title}",
        border,
        f"  Composite Score (S_val) : {results['S_val']:.5f}",
        "-" * 55,
        f"  ECOD AUPRC              : {results['AUPRC_ECOD']:.5f}",
        f"  IForest AUPRC (10 seeds): {results['AUPRC_IForest_mean']:.5f}",
        f"  IForest Min / Max Seed  : {min(results['AUPRC_IForest_seeds']):.5f} / {max(results['AUPRC_IForest_seeds']):.5f}",
        "-" * 55,
        f"  Synthetic Rows Evaluated: {results['n_synthetic_rows']:,}",
        f"  Validation Rows Scored  : {results['n_validation_rows']:,}",
        f"  Encoded Feature Dim     : {results['feature_dim']}",
        f"  Evaluation Time         : {results['duration_seconds']}s",
        border,
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI runner to evaluate any candidate synthetic CSV against validation data."""
    parser = argparse.ArgumentParser(
        description="Evaluate downstream anomaly detection utility on synthetic data."
    )
    parser.add_argument(
        "--synthetic", type=Path, required=True, help="Path to synthetic normal CSV"
    )
    parser.add_argument(
        "--validation", type=Path, required=True, help="Path to labeled validation CSV/Parquet"
    )
    parser.add_argument(
        "--dataset",
        choices=["nsl_kdd", "unsw_nb15"],
        required=True,
        help="Dataset identifier (nsl_kdd or unsw_nb15)",
    )
    args = parser.parse_args()

    config_map: dict[str, DatasetConfig] = {
        "nsl_kdd": NSL_KDD,
        "unsw_nb15": UNSW_NB15,
    }
    config = config_map[args.dataset]

    print(f"Loading synthetic candidate: {args.synthetic}")
    synthetic_df = (
        pl.read_parquet(args.synthetic)
        if args.synthetic.suffix == ".parquet"
        else pl.read_csv(args.synthetic)
    )

    print(f"Loading validation set: {args.validation}")
    validation_df = (
        pl.read_parquet(args.validation)
        if args.validation.suffix == ".parquet"
        else pl.read_csv(args.validation)
    )

    print("Running downstream detector benchmark (ECOD + 10-seed Isolation Forest)...")
    results = evaluate_synthetic_utility(
        synthetic_df=synthetic_df,
        validation_df=validation_df,
        categorical_cols=config.categorical,
        target_col=config.label,
    )
    print(format_evaluation_summary(results, title=f"{config.name.upper()} Local Benchmark"))


if __name__ == "__main__":
    main()
