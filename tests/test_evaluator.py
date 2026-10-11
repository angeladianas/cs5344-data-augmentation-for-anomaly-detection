"""Unit tests for downstream anomaly detection evaluator (utils/evaluator.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from utils.evaluator import (
    evaluate_ecod,
    evaluate_iforest,
    evaluate_synthetic_utility,
    format_evaluation_summary,
    prepare_detector_features,
)


@pytest.fixture
def sample_evaluation_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate toy synthetic normal data and labeled validation data."""
    np.random.seed(42)
    n_syn = 200
    n_val_norm = 100
    n_val_att = 30

    # Normal distribution around (0, 0)
    syn_df = pd.DataFrame(
        {
            "num1": np.random.normal(0.0, 1.0, n_syn),
            "num2": np.random.normal(0.0, 1.0, n_syn),
            "proto": np.random.choice(["tcp", "udp"], size=n_syn),
        }
    )

    val_norm_df = pd.DataFrame(
        {
            "num1": np.random.normal(0.0, 1.0, n_val_norm),
            "num2": np.random.normal(0.0, 1.0, n_val_norm),
            "proto": np.random.choice(["tcp", "udp"], size=n_val_norm),
            "is_anomaly": [0] * n_val_norm,
        }
    )

    # Attack distribution shifted to (4, 4)
    val_att_df = pd.DataFrame(
        {
            "num1": np.random.normal(4.0, 1.0, n_val_att),
            "num2": np.random.normal(4.0, 1.0, n_val_att),
            "proto": np.random.choice(["tcp", "icmp"], size=n_val_att),
            "is_anomaly": [1] * n_val_att,
        }
    )

    val_df = pd.concat([val_norm_df, val_att_df], ignore_index=True)
    return syn_df, val_df


def test_prepare_detector_features(
    sample_evaluation_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verifies feature extraction, one-hot encoding, and label exclusion."""
    syn_df, val_df = sample_evaluation_data

    # Test fitting on synthetic data
    X_syn, encoder = prepare_detector_features(syn_df, categorical_cols=["proto"])
    assert X_syn.shape[0] == len(syn_df)
    # 2 numerical + 2 one-hot categories (tcp, udp) = 4 columns
    assert X_syn.shape[1] == 4

    # Test transforming validation set using fitted encoder
    # Note: validation contains 'icmp' which should be ignored (all zeros) without error
    X_val, _ = prepare_detector_features(val_df, categorical_cols=["proto"], encoder=encoder)
    assert X_val.shape[0] == len(val_df)
    assert X_val.shape[1] == 4


def test_prepare_detector_features_polars(
    sample_evaluation_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verifies prepare_detector_features works with Polars DataFrames."""
    syn_df, _ = sample_evaluation_data
    syn_pl = pl.DataFrame(syn_df.to_dict(orient="list"))

    X_syn, _ = prepare_detector_features(syn_pl, categorical_cols=["proto"])
    assert X_syn.shape[0] == len(syn_df)
    assert X_syn.shape[1] == 4


def test_evaluate_ecod(sample_evaluation_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Verifies ECOD scoring on synthetic normal vs validation data."""
    syn_df, val_df = sample_evaluation_data
    X_syn, encoder = prepare_detector_features(syn_df, categorical_cols=["proto"])
    X_val, _ = prepare_detector_features(val_df, categorical_cols=["proto"], encoder=encoder)
    y_val = val_df["is_anomaly"].to_numpy()

    auprc, scores = evaluate_ecod(X_syn, X_val, y_val, contamination=0.10)
    assert 0.0 <= auprc <= 1.0
    assert len(scores) == len(val_df)
    # Attacks at (4, 4) should yield high AUPRC
    assert auprc > 0.50


def test_evaluate_iforest(sample_evaluation_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Verifies Isolation Forest multi-seed scoring."""
    syn_df, val_df = sample_evaluation_data
    X_syn, encoder = prepare_detector_features(syn_df, categorical_cols=["proto"])
    X_val, _ = prepare_detector_features(val_df, categorical_cols=["proto"], encoder=encoder)
    y_val = val_df["is_anomaly"].to_numpy()

    test_seeds = (42, 101)
    mean_auprc, seed_auprcs, scores = evaluate_iforest(
        X_syn, X_val, y_val, seeds=test_seeds, n_estimators=50
    )
    assert 0.0 <= mean_auprc <= 1.0
    assert len(seed_auprcs) == len(test_seeds)
    assert len(scores) == len(val_df)
    assert mean_auprc > 0.50


def test_evaluate_synthetic_utility_end_to_end(
    sample_evaluation_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verifies end-to-end evaluation pipeline calculating S_val."""
    syn_df, val_df = sample_evaluation_data
    test_seeds = (42, 101)

    results = evaluate_synthetic_utility(
        synthetic_df=syn_df,
        validation_df=val_df,
        categorical_cols=["proto"],
        target_col="is_anomaly",
        seeds=test_seeds,
    )

    assert "S_val" in results
    assert "AUPRC_ECOD" in results
    assert "AUPRC_IForest_mean" in results
    assert "AUPRC_IForest_seeds" in results
    assert len(results["AUPRC_IForest_seeds"]) == len(test_seeds)
    assert 0.0 <= results["S_val"] <= 1.0
    assert results["n_synthetic_rows"] == len(syn_df)
    assert results["n_validation_rows"] == len(val_df)

    summary = format_evaluation_summary(results, title="Toy Test")
    assert "Composite Score (S_val)" in summary
    assert "ECOD AUPRC" in summary
