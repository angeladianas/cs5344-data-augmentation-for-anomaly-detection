r"""Unit and integration test suite for GMM candidate generation and post-processing.

Tests component-conditional donor sampling, protocol-stratified allocation,
marginal calibration, precision alignment, and domain invariant enforcement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

import models.gmm_candidates as gmm_candidates
from models.gmm_candidates import (
    align_candidate_precision,
    allocate_group_sample_counts,
    apply_boundary_filter,
    calibrate_gmm_feature_marginal,
    calibrate_unsw_response_body_len,
    derive_unsw_packet_means,
    derive_unsw_tcprtt,
    generation_group_masks,
    infer_source_decimal_precision,
    protocol_group_masks,
    reconstruct_rare_pair,
    reconstruct_unsw_ct_state_ttl,
    sample_component_conditional_donors,
    zero_unsw_tcp_fields_for_non_tcp,
)


def test_component_conditional_sampling_preserves_discrete_tuples() -> None:
    r"""Verify that GMM discrete donor sampling preserves joint categorical associations."""
    donors = pl.DataFrame(
        {
            "protocol": ["tcp", "udp", "icmp", "tcp"],
            "state": ["S0", "SF", "OTH", "REJ"],
        }
    )
    responsibilities = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    sampled = sample_component_conditional_donors(
        np.array([0, 1, 0, 1]),
        responsibilities,
        donors,
        random_state=42,
    )

    assert sampled.height == 4
    assert set(sampled.filter(pl.col("protocol") == "tcp")["state"].to_list()) <= {
        "S0",
        "REJ",
    }
    assert sampled.row(0) in {donors.row(0), donors.row(1)}
    assert sampled.row(1) in {donors.row(2), donors.row(3)}
    assert sampled.row(2) in {donors.row(0), donors.row(1)}
    assert sampled.row(3) in {donors.row(2), donors.row(3)}


def test_component_conditional_sampling_rejects_missing_component_mass() -> None:
    r"""Verify that components with zero probability mass raise a ValueError."""
    donors = pl.DataFrame({"category": ["a", "b"]})

    with pytest.raises(ValueError, match="probability mass"):
        sample_component_conditional_donors(
            np.array([1]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            donors,
            random_state=42,
        )


@pytest.mark.parametrize(
    ("dataset", "protocols", "expected_sizes", "expected_samples"),
    [
        (
            "nsl_kdd",
            ["tcp"] * 3 + ["udp"] * 2 + ["icmp"],
            {"tcp": 3, "non_tcp": 3},
            {"tcp": 10, "non_tcp": 10},
        ),
        (
            "unsw_nb15",
            ["tcp"] * 5 + ["udp"] * 3 + ["arp"] * 2,
            {"tcp": 5, "udp": 3, "other": 2},
            {"tcp": 11, "udp": 6, "other": 4},
        ),
    ],
)
def test_protocol_group_sample_counts_follow_training_proportions(
    dataset: str,
    protocols: list[str],
    expected_sizes: dict[str, int],
    expected_samples: dict[str, int],
) -> None:
    r"""Verify protocol group mask sizes and proportional sample count allocations."""
    groups = protocol_group_masks(dataset, protocols)
    assert {name: int(mask.sum()) for name, mask in groups.items()} == expected_sizes
    n_samples = 21 if dataset == "unsw_nb15" else 20
    assert allocate_group_sample_counts(groups, n_samples) == expected_samples
    assert sum(allocate_group_sample_counts(groups, n_samples).values()) == n_samples


def test_generation_versions_select_ungrouped_and_protocol_grouped_masks() -> None:
    r"""Verify GMM generation group mask selection for v1 (monolithic) and v2 (grouped)."""
    protocols = ["tcp", "udp", "arp"]

    v1_groups = generation_group_masks("unsw_nb15", "v1", protocols)
    v2_groups = generation_group_masks("unsw_nb15", "v2", protocols)

    assert list(v1_groups) == ["all"]
    assert v1_groups["all"].all()
    assert {name: int(mask.sum()) for name, mask in v2_groups.items()} == {
        "tcp": 1,
        "udp": 1,
        "other": 1,
    }
    with pytest.raises(ValueError, match="Unknown GMM generation version"):
        generation_group_masks("unsw_nb15", "v3", protocols)


def test_gmm_marginal_calibration_matches_reference_tail() -> None:
    r"""Verify that marginal empirical quantile calibration matches reference heavy tails."""
    generated = np.random.default_rng(42).normal(size=10_000)
    reference = np.sort(np.concatenate((np.arange(9_900), np.arange(1_000, 101_000, 1_000))))

    calibrated = calibrate_gmm_feature_marginal(
        generated,
        reference,
    )

    assert np.isclose(np.quantile(calibrated, 0.99), np.quantile(reference, 0.99), rtol=0.01)
    assert np.isclose(np.quantile(calibrated, 0.999), np.quantile(reference, 0.999), rtol=0.01)
    assert calibrated.max() <= reference.max()


def test_gmm_marginal_calibration_preserves_rare_zero_rate() -> None:
    r"""Verify that rare zero-point masses in reference data are preserved after calibration."""
    reference = np.concatenate(([0.0], np.arange(1.0, 20_000.0)))
    generated = np.random.default_rng(42).normal(size=24_000)

    calibrated = calibrate_gmm_feature_marginal(generated, reference)

    assert np.count_nonzero(np.rint(calibrated) == 0) == 1


def test_gmm_marginal_calibration_preserves_common_zero_rate() -> None:
    r"""Verify that dominant zero-inflation frequencies are preserved under marginal mapping."""
    reference = np.concatenate((np.zeros(2_600), np.arange(1.0, 13_401.0)))
    generated = np.random.default_rng(42).normal(size=24_000)

    calibrated = calibrate_gmm_feature_marginal(generated, reference)

    assert abs(np.count_nonzero(calibrated == 0) - 24_000 * 2_600 / 16_000) <= 1


def test_source_decimal_precision_and_candidate_rounding(tmp_path: pytest.TempPathFactory) -> None:
    r"""Verify inference of column decimal precisions and rounding alignment on candidates."""
    source = tmp_path / "source.csv"
    source.write_text(
        "rate,count,category\n0.00,1,normal\n0.01,2.0,attack\n1.0,3,normal\n",
        encoding="utf-8",
    )

    precision = infer_source_decimal_precision(
        source,
        ["rate", "count", "category"],
        ["rate", "count"],
    )
    aligned = align_candidate_precision(
        pl.DataFrame({"rate": [0.12345, 0.98765], "count": [1.0, 2.0]}),
        precision,
    )

    assert precision == {"rate": 2, "count": 0}
    assert aligned["rate"].to_list() == [0.12, 0.99]
    assert aligned["count"].to_list() == [1.0, 2.0]


def test_derive_unsw_tcprtt_from_components() -> None:
    r"""Verify deterministic mathematical constraint tcprtt = synack + ackdat with bounds."""
    candidates = pl.DataFrame(
        {
            "tcprtt": [9.0, 9.0, 9.0],
            "synack": [0.123456, 0.0, 4.0],
            "ackdat": [0.654321, 0.000001, 3.0],
        }
    )

    derived, violations = derive_unsw_tcprtt(candidates)

    assert derived["tcprtt"].to_list() == [0.777777, 0.000001, 3.5]
    assert derived["synack"].to_list() == [0.123456, 0.0, 2.0]
    assert derived["ackdat"].to_list() == [0.654321, 0.000001, 1.5]
    assert derived["tcprtt"].max() <= 3.5
    assert violations == 0


def test_zero_unsw_tcp_fields_for_non_tcp_protocols() -> None:
    r"""Verify that TCP-specific header window/sequence fields are zeroed for non-TCP traffic."""
    candidates = pl.DataFrame(
        {
            "proto": ["tcp", "udp", "icmp"],
            "swin": [255, 100, 0],
            "stcpb": [1234, 5678, 0],
            "dtcpb": [4321, 8765, 0],
            "dwin": [255, 100, 0],
        }
    )

    result, zeroed_counts = zero_unsw_tcp_fields_for_non_tcp(candidates)

    for feature in ("swin", "stcpb", "dtcpb", "dwin"):
        assert result[feature].to_list()[1:] == [0, 0]
        assert result[feature].to_list()[0] == candidates[feature].to_list()[0]
    assert zeroed_counts == {"swin": 1, "stcpb": 1, "dtcpb": 1, "dwin": 1}


def test_unsw_response_body_length_uses_http_conditional_reference() -> None:
    r"""Verify HTTP-conditional calibration of response_body_len and non-HTTP zeroing."""
    synthetic = pl.DataFrame(
        {
            "service": ["http", "http", "dns", "dns"],
            "ct_flw_http_mthd": [0, 1, 0, 0],
            "response_body_len": [1_000_000, 2_000_000, 10_000, 20_000],
        }
    )
    real_normal = pl.DataFrame(
        {
            "service": ["http", "http", "http", "http", "dns", "dns"],
            "ct_flw_http_mthd": [0, 0, 1, 1, 0, 0],
            "response_body_len": [0, 0, 10, 20, 0, 0],
        }
    )

    calibrated, metrics = calibrate_unsw_response_body_len(synthetic, real_normal)

    assert calibrated["response_body_len"].to_list() == [0, 13, 0, 0]
    assert np.isclose(metrics["http"]["reference_p99"], 19.7)
    assert metrics["http"]["calibrated_p99"] <= metrics["http"]["reference_p99"]
    assert metrics["non_http"]["calibrated_p99"] == 0.0


def test_unsw_ct_state_ttl_matches_conditional_zero_rates() -> None:
    r"""Verify conditional zero rate matching and donor value sampling for ct_state_ttl."""
    real_normal = pl.DataFrame(
        {
            "proto": ["tcp", "tcp", "tcp", "tcp", "udp", "udp", "tcp", "tcp"],
            "state": ["FIN", "FIN", "FIN", "FIN", "CON", "CON", "REQ", "REQ"],
            "ct_state_ttl": [0, 0, 2, 4, 0, 0, 2, 2],
        }
    )
    synthetic = pl.DataFrame(
        {
            "proto": ["tcp"] * 20 + ["udp"] * 6 + ["tcp"] * 4 + ["ip"] * 20,
            "state": ["FIN"] * 20 + ["CON"] * 6 + ["REQ"] * 4 + ["INT"] * 20,
            "ct_state_ttl": [10] * 20 + [1] * 6 + [0] * 4 + [1] * 20,
        }
    )

    reconstructed, counts = reconstruct_unsw_ct_state_ttl(
        synthetic,
        real_normal,
        seed=42,
    )

    assert reconstructed[:20]["ct_state_ttl"].eq(0).sum() == 10
    assert reconstructed[20:26]["ct_state_ttl"].eq(0).sum() == 6
    assert reconstructed[26:30]["ct_state_ttl"].eq(0).sum() == 0
    assert reconstructed[30:]["ct_state_ttl"].eq(0).sum() == 10
    assert reconstructed["ct_state_ttl"].min() == 0
    tcp_fin_nonzero = [
        value for value in reconstructed[:20]["ct_state_ttl"].to_list() if value != 0
    ]
    global_nonzero = [value for value in reconstructed[30:]["ct_state_ttl"].to_list() if value != 0]
    assert tcp_fin_nonzero == [10] * 10
    assert set(global_nonzero) <= {1, 2, 4}
    assert counts["protocol"] == 0
    assert counts["global"] == 20


def test_derive_unsw_packet_means_and_zero_packet_behavior() -> None:
    r"""Verify packet mean quotient relationships and handling of zero packet counts."""
    candidates = pl.DataFrame(
        {
            "sbytes": [10, 1],
            "spkts": [3, 2],
            "smean": [0, 0],
            "dbytes": [12, 8],
            "dpkts": [4, 0],
            "dmean": [0, 0],
        }
    )

    derived, violations, zero_packet_repairs = derive_unsw_packet_means(candidates)

    assert derived["smean"].to_list() == [3, 1]
    assert derived["dmean"].to_list() == [3, 0]
    assert derived["dbytes"].to_list() == [12, 0]
    assert derived["smean"].dtype == pl.Int64
    for byte_column, packet_column, mean_column in (
        ("sbytes", "spkts", "smean"),
        ("dbytes", "dpkts", "dmean"),
    ):
        byte_counts = derived[byte_column].to_numpy()
        packet_counts = derived[packet_column].to_numpy()
        means = derived[mean_column].to_numpy()
        assert np.all(np.abs(byte_counts - packet_counts * means) <= packet_counts / 2 + 1e-9)
    assert violations == {"smean": 0, "dmean": 0}
    assert zero_packet_repairs == {"sbytes": 0, "dbytes": 1}


def test_rare_pair_reconstruction_uses_context_and_fallbacks() -> None:
    r"""Verify context-conditional donor sampling and fallback tiers for rare discrete pairs."""
    real_normal = pd.DataFrame(
        {
            "protocol_type": ["tcp", "tcp", "udp", "icmp"],
            "service": ["http", "ftp", "dns", "echo"],
            "flag": ["SF", "S0", "SF", "OTH"],
            "logged_in": [1, 0, 1, 0],
            "num_compromised": [2, 3, 5, 7],
            "num_root": [11, 13, 17, 19],
        }
    )
    synthetic = pd.DataFrame(
        {
            "protocol_type": ["tcp", "tcp", "unknown"],
            "service": ["http", "smtp", "none"],
            "flag": ["SF", "REJ", "unknown"],
            "logged_in": [1, 1, 1],
            "num_compromised": [100, 200, 300],
            "num_root": [1000, 2000, 3000],
        }
    )

    reconstructed = reconstruct_rare_pair(synthetic, real_normal, seed=3)

    assert tuple(reconstructed.loc[0, ["num_compromised", "num_root"]]) == (2, 11)
    assert tuple(reconstructed.loc[1, ["num_compromised", "num_root"]]) in {
        (2, 11),
        (3, 13),
    }
    assert tuple(reconstructed.loc[2, ["num_compromised", "num_root"]]) in {
        (2, 11),
        (3, 13),
        (5, 17),
        (7, 19),
    }
    assert reconstructed.loc[1, "num_compromised"] != 200


def test_boundary_filter_writes_filtered_candidates_and_scores(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""Verify boundary filter output serialization and score column matching."""

    class FakeBoundaryFilter:
        r"""Mock boundary filter classifier for isolated unit testing."""

        def __init__(self) -> None:
            r"""Initialize mock boundary filter with dummy feature column names."""
            self.feature_names = ["feature_a", "feature_b"]

        def filter_candidates(
            self, candidates: pl.DataFrame, *, tau_reject: float
        ) -> tuple[pl.DataFrame, float, np.ndarray]:
            r"""Filter candidates based on fixed mock anomaly posterior probabilities."""
            probabilities = np.array([0.01, 0.11, 0.10])
            accepted = probabilities <= tau_reject
            return candidates[accepted], float(accepted.mean()), probabilities

    model_path = tmp_path / "boundary_filter_nsl_kdd.pkl"
    model_path.touch()
    monkeypatch.setattr(gmm_candidates, "BOUNDARY_FILTER_DIR", tmp_path)

    def load_fake_boundary_filter(path: str | Path) -> FakeBoundaryFilter:
        r"""Return mock filter instance asserting expected model path."""
        assert path == model_path
        return FakeBoundaryFilter()

    monkeypatch.setattr(gmm_candidates.LightGBMBoundaryFilter, "load", load_fake_boundary_filter)
    candidates = pl.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": ["a", "b", "c"],
        }
    )

    result = apply_boundary_filter(
        "nsl_kdd",
        candidates,
        tmp_path,
        tau_reject=0.10,
    )

    assert result["input_rows"] == 3
    assert result["accepted_rows"] == 2
    assert result["rejected_rows"] == 1
    assert pl.read_csv(tmp_path / "filtered_samples.csv").height == 2
    assert pl.read_parquet(tmp_path / "filtered_samples.parquet").height == 2
    scores = pl.read_csv(tmp_path / "boundary_filter_scores.csv")
    assert scores["accepted"].to_list() == [True, False, True]
