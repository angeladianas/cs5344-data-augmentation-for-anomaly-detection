r"""Submission Output Integrity & Bitwise Invariance Test Suite.

Validates the final competition deliverables:
- NSL-KDD (40,000 rows, 42 columns)
- UNSW-NB15 (70,000 rows, 43 columns)

Guarantees zero nulls, exact feature schema alignment with training data,
proper 'id' index sequence, adherence to domain protocol constraints,
and exact bitwise SHA-256 hash preservation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from utils.dataset_configs import NSL_KDD, UNSW_NB15, DatasetConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

EXPECTED_SUBMISSIONS = {
    "nsl_kdd": {
        "filename": "submission_nsl_kdd.csv",
        "expected_rows": 40_000,
        "config": NSL_KDD,
        "sha256": "bbb59bb172020f7b2d4617a6f13f54e16401cfe6f45cb5f804fa29eaf1f219a7",
    },
    "unsw_nb15": {
        "filename": "submission_unsw_nb15.csv",
        "expected_rows": 70_000,
        "config": UNSW_NB15,
        "sha256": "cb283a31bf8ce4c17b616cd721319d0a6ab577f551f10b5c034b741019017cc8",
    },
}


@pytest.mark.parametrize("dataset_key", ["nsl_kdd", "unsw_nb15"])
def test_submission_files_exist(dataset_key: str) -> None:
    r"""Verify that submission CSV files exist in the submissions directory."""
    meta = EXPECTED_SUBMISSIONS[dataset_key]
    csv_path = SUBMISSIONS_DIR / meta["filename"]
    assert csv_path.exists(), f"Submission file {csv_path} does not exist!"
    assert csv_path.stat().st_size > 0, f"Submission file {csv_path} is empty!"


@pytest.mark.parametrize("dataset_key", ["nsl_kdd", "unsw_nb15"])
def test_submission_row_count_and_columns(dataset_key: str) -> None:
    r"""Verify row counts, column count, 'id' sequencing, and omission of 'is_anomaly'."""
    meta = EXPECTED_SUBMISSIONS[dataset_key]
    csv_path = SUBMISSIONS_DIR / meta["filename"]
    df = pl.read_csv(csv_path)

    expected_n = meta["expected_rows"]
    config: DatasetConfig = meta["config"]

    # 1. Exact row count
    assert (
        df.height == expected_n
    ), f"{dataset_key}: Expected {expected_n} rows, found {df.height}"

    # 2. First column must be 'id'
    assert (
        df.columns[0] == "id"
    ), f"{dataset_key}: First column must be 'id', found {df.columns[0]}"
    ids = df["id"].to_list()
    assert ids == list(
        range(expected_n)
    ), f"{dataset_key}: 'id' column is not contiguous range(0, {expected_n})"

    # 3. Label column must not be present
    assert (
        "is_anomaly" not in df.columns
    ), f"{dataset_key}: Found forbidden 'is_anomaly' column!"

    # 4. Feature columns must match dataset specification
    expected_features = set(config.features)
    actual_features = set(df.columns[1:])
    assert (
        actual_features == expected_features
    ), f"{dataset_key}: Features do not match specification! Diff: {actual_features ^ expected_features}"


@pytest.mark.parametrize("dataset_key", ["nsl_kdd", "unsw_nb15"])
def test_submission_null_integrity(dataset_key: str) -> None:
    r"""Verify that submission files contain zero null, NaN, or missing values."""
    meta = EXPECTED_SUBMISSIONS[dataset_key]
    csv_path = SUBMISSIONS_DIR / meta["filename"]
    df = pl.read_csv(csv_path)

    total_nulls = df.null_count().sum_horizontal().item()
    assert (
        total_nulls == 0
    ), f"{dataset_key}: Detected {total_nulls} null values in submission!"


def test_unsw_domain_invariants() -> None:
    r"""Verify domain protocol invariants specific to UNSW-NB15 synthetic data."""
    meta = EXPECTED_SUBMISSIONS["unsw_nb15"]
    csv_path = SUBMISSIONS_DIR / meta["filename"]
    df = pl.read_csv(csv_path)

    # 1. Non-TCP flows must have 0 for TCP-only fields
    non_tcp_df = df.filter(pl.col("proto").str.to_lowercase() != "tcp")
    for field in ["swin", "stcpb", "dtcpb", "dwin"]:
        if field in non_tcp_df.columns:
            non_zero_count = non_tcp_df.filter(pl.col(field) != 0).height
            assert (
                non_zero_count == 0
            ), f"Found {non_zero_count} non-zero values for {field} in non-TCP flows!"

    # 2. synack + ackdat <= 3.5 seconds
    if all(c in df.columns for c in ["synack", "ackdat"]):
        timing_sum = df["synack"] + df["ackdat"]
        violations = (timing_sum > 3.5001).sum()
        assert (
            violations == 0
        ), f"Found {violations} flows where synack + ackdat exceeds 3.5s bound!"


@pytest.mark.parametrize("dataset_key", ["nsl_kdd", "unsw_nb15"])
def test_submission_sha256_invariance(dataset_key: str) -> None:
    r"""Verify that submission files maintain exact bitwise SHA-256 reproducibility."""
    meta = EXPECTED_SUBMISSIONS[dataset_key]
    csv_path = SUBMISSIONS_DIR / meta["filename"]
    with open(csv_path, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()

    expected_sha256 = meta["sha256"]
    assert actual_sha256 == expected_sha256, (
        f"{dataset_key}: Bitwise hash mismatch! "
        f"Expected {expected_sha256}, got {actual_sha256}. Output has been modified!"
    )
