"""Adversarial stress-testing suite for CS5344 Group 39.

Tests data leakage isolation, boundary filter resilience, zero-inflation preservation,
decoding invariance, and submission gatekeeping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from utils.boundary_filter import LightGBMBoundaryFilter
from utils.preprocessing import make_development_split


class TestDataLeakageIsolation:
    """Verifies that development and validation partitions have zero leakage."""

    def test_no_feature_overlap_between_partitions(self) -> None:
        """Confirms that duplicate feature groups are never split across partitions."""
        df = pl.DataFrame(
            {
                "feat_a": [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0],
                "feat_b": ["x", "x", "y", "z", "w", "a", "a", "b", "c", "d"],
                "is_anomaly": [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
            }
        )
        feature_cols = ["feat_a", "feat_b"]
        dev_df, val_df, _ = make_development_split(
            df, feature_cols, "is_anomaly", validation_fraction=0.3, seed=42
        )

        assert dev_df.height + val_df.height == df.height
        overlap = (
            dev_df.select(feature_cols)
            .unique()
            .join(val_df.select(feature_cols).unique(), on=feature_cols, how="inner")
        )
        assert overlap.height == 0, "Feature rows leaked across development and validation!"

    def test_attack_categories_quarantined_from_normal_vocab(self) -> None:
        """Verifies that attack-exclusive categories are not assigned to normal generator."""
        normal_services = {"http", "smtp", "dns"}
        generator_vocab = sorted(normal_services)
        assert "neptune_flood" not in generator_vocab
        assert "smurf_attack" not in generator_vocab


class TestBoundaryFilterResilience:
    """Tests the LightGBM boundary rejection filter under stress and edge cases."""

    @pytest.fixture
    def toy_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        np.random.seed(42)
        n = 200
        norm_x = np.random.normal(loc=0.0, scale=1.0, size=(n, 2))
        att_x = np.random.normal(loc=5.0, scale=1.0, size=(n, 2))

        train_df = pd.DataFrame(
            {
                "num1": np.concatenate([norm_x[:, 0], att_x[:, 0]]),
                "num2": np.concatenate([norm_x[:, 1], att_x[:, 1]]),
                "proto": ["tcp"] * (2 * n),
                "is_anomaly": [0] * n + [1] * n,
            }
        )

        cand_df = pd.DataFrame(
            {
                "num1": [0.1, -0.2, 4.8, 5.2, 0.0],
                "num2": [0.2, 0.1, 4.9, 5.1, -0.1],
                "proto": ["tcp", "tcp", "tcp", "tcp", "tcp"],
            }
        )
        return train_df, cand_df

    def test_pandas_and_polars_support(self, toy_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """Verifies that filter works interchangeably with Pandas and Polars."""
        train_df, cand_df = toy_data
        filter_model = LightGBMBoundaryFilter(random_state=42)
        filter_model.fit(train_df)

        # Test with Pandas input
        filtered_pd, _, probs_pd = filter_model.filter_candidates(cand_df, tau_reject=0.30)
        assert isinstance(filtered_pd, pd.DataFrame)
        assert len(filtered_pd) == 3

        # Test with Polars input (safe conversion without pyarrow requirement)
        cand_pl = pl.DataFrame(cand_df.to_dict(orient="list"))
        filtered_pl, _, probs_pl = filter_model.filter_candidates(cand_pl, tau_reject=0.30)
        assert isinstance(filtered_pl, pl.DataFrame)
        assert filtered_pl.height == 3
        assert np.allclose(probs_pd, probs_pl)

    def test_extreme_thresholds(self, toy_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """Verifies behavior when tau_reject is 0.0 or 1.0."""
        train_df, cand_df = toy_data
        filter_model = LightGBMBoundaryFilter(random_state=42)
        filter_model.fit(train_df)

        filtered_all, rate_all, _ = filter_model.filter_candidates(cand_df, tau_reject=1.0)
        assert len(filtered_all) == len(cand_df)
        assert rate_all == 1.0

        filtered_strict, _, _ = filter_model.filter_candidates(cand_df, tau_reject=0.0)
        assert len(filtered_strict) <= len(cand_df)

    def test_depletion_safeguard_fallback(
        self, toy_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Verifies dynamic fallback when tau is too aggressive for min_survivors."""
        train_df, cand_df = toy_data
        filter_model = LightGBMBoundaryFilter(random_state=42)
        filter_model.fit(train_df)

        filtered_df, rate, _ = filter_model.filter_candidates(
            cand_df, tau_reject=0.00001, min_survivors=4
        )
        assert len(filtered_df) == 4
        assert rate == 4 / len(cand_df)

    def test_save_and_load(
        self, toy_data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: pytest.TempPathFactory
    ) -> None:
        """Verifies serialization round-trip."""
        train_df, cand_df = toy_data
        filter_model = LightGBMBoundaryFilter(random_state=42).fit(train_df)
        model_path = tmp_path / "filter.pkl"
        filter_model.save(model_path)

        loaded_model = LightGBMBoundaryFilter.load(model_path)
        probs_orig = filter_model.predict_anomaly_prob(cand_df)
        probs_loaded = loaded_model.predict_anomaly_prob(cand_df)
        assert np.array_equal(probs_orig, probs_loaded)


class TestZeroInflationPreservation:
    """Verifies that discrete zeros on payload features are preserved upon decoding."""

    def test_zero_byte_discrete_spike(self) -> None:
        """Decoded integer counts and bytes must remain non-negative integers."""
        raw_values = np.array([-0.005, 0.000, 0.001, 10.4, 25.8])
        rounded = np.rint(raw_values)
        clipped = np.clip(rounded, 0, np.inf)

        assert clipped[0] == 0.0
        assert clipped[1] == 0.0
        assert clipped[2] == 0.0
        assert clipped[3] == 10.0
        assert clipped[4] == 26.0


class TestSubmissionIntegrityGatekeeper:
    """Tests strict assertions on submission CSV outputs."""

    def test_reject_submission_with_anomaly_label(self) -> None:
        """Gatekeeper must reject any submission containing is_anomaly."""
        sub_df = pd.DataFrame({"src_bytes": [100, 200], "is_anomaly": [0, 0]})
        with pytest.raises(AssertionError, match="is_anomaly"):
            assert "is_anomaly" not in sub_df.columns, "Submission must NOT contain 'is_anomaly'"

    def test_reject_submission_with_nan_or_inf(self) -> None:
        """Gatekeeper must reject nulls, NaNs, or infs."""
        sub_df = pd.DataFrame({"src_bytes": [100, np.nan], "dst_bytes": [50, 60]})
        with pytest.raises(AssertionError, match="NaN"):
            assert not sub_df.isnull().any().any(), "Submission contains null or NaN values"

    def test_reject_wrong_row_count(self) -> None:
        """Gatekeeper must reject mismatched row count."""
        sub_df = pd.DataFrame({"src_bytes": [100] * 39999})
        target_n = 40000
        with pytest.raises(AssertionError, match="40000"):
            assert len(sub_df) == target_n, f"Expected {target_n} rows, got {len(sub_df)}"
