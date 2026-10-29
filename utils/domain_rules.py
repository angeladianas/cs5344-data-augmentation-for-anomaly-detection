r"""Network Protocol Domain Rules and Physical Invariant Enforcement.

Provides pure functional transforms to enforce RFC physical invariants,
packet/byte count identities, TCP window and sequence constraints,
and dataset precision alignment across all synthetic candidate pools.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import polars as pl

UNSW_TCPRTT_MAX_SECONDS = 3.5
UNSW_TCP_ONLY_FEATURES = ("swin", "stcpb", "dtcpb", "dwin")
UNSW_PACKET_BYTE_PAIRS = (("spkts", "sbytes", "smean"), ("dpkts", "dbytes", "dmean"))


def infer_source_decimal_precision(
    source_path: str | Path,
    features: list[str],
    numeric_columns: list[str],
) -> dict[str, int]:
    r"""Infer maximum decimal places for numeric columns from the source CSV file."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source dataset not found: {path}")

    numeric_set = set(numeric_columns)
    precision: dict[str, int] = {column: 0 for column in numeric_columns}

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for column in numeric_set:
                value = row.get(column)
                if not value:
                    continue
                stripped = value.strip()
                decimals = len(stripped.split(".", 1)[1]) if "." in stripped else 0
                if decimals > precision[column]:
                    precision[column] = decimals

    return precision


def align_candidate_precision(
    candidates: pl.DataFrame,
    decimal_places: dict[str, int],
) -> pl.DataFrame:
    r"""Round numerical output columns to the source dataset's decimal precision."""
    missing = set(decimal_places) - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidates are missing numeric columns: {sorted(missing)}")
    if any(not isinstance(places, int) or places < 0 for places in decimal_places.values()):
        raise ValueError("Decimal-place precision must be a nonnegative integer")
    return candidates.with_columns(
        pl.col(column).round(places) for column, places in decimal_places.items()
    )


def derive_unsw_tcprtt(synthetic: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    r"""Derive bounded UNSW TCP round-trip time: tcprtt = synack + ackdat (capped at 3.5s)."""
    required = {"tcprtt", "synack", "ackdat"}
    missing = required - set(synthetic.columns)
    if missing:
        raise ValueError(f"UNSW candidates are missing timing columns: {sorted(missing)}")

    synack = synthetic["synack"].to_numpy().astype(float)
    ackdat = synthetic["ackdat"].to_numpy().astype(float)
    if not np.isfinite(synack).all() or not np.isfinite(ackdat).all():
        raise ValueError("UNSW timing components must be finite")
    if (synack < 0).any() or (ackdat < 0).any():
        raise ValueError("UNSW timing components must be nonnegative")

    total = synack + ackdat
    over_bound = total > UNSW_TCPRTT_MAX_SECONDS
    scale = np.ones_like(total)
    np.divide(
        UNSW_TCPRTT_MAX_SECONDS,
        total,
        out=scale,
        where=over_bound,
    )
    synack *= scale
    ackdat *= scale
    capped_total = synack + ackdat
    floating_point_overflow = capped_total > UNSW_TCPRTT_MAX_SECONDS
    ackdat[floating_point_overflow] = UNSW_TCPRTT_MAX_SECONDS - synack[floating_point_overflow]

    synthetic = synthetic.with_columns(
        pl.Series("synack", synack).cast(synthetic.schema["synack"]),
        pl.Series("ackdat", ackdat).cast(synthetic.schema["ackdat"]),
    ).with_columns((pl.col("synack") + pl.col("ackdat")).alias("tcprtt"))
    violations = (
        synthetic.select((pl.col("tcprtt") - pl.col("synack") - pl.col("ackdat")).abs())
        .to_series()
        .gt(1e-5)
        .sum()
    )
    return synthetic, int(violations)


def zero_unsw_tcp_fields_for_non_tcp(
    synthetic: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    r"""Zero TCP window and sequence fields for non-TCP protocols in UNSW-NB15."""
    required = {"proto", *UNSW_TCP_ONLY_FEATURES}
    missing = required - set(synthetic.columns)
    if missing:
        raise ValueError(f"UNSW candidates are missing TCP-only columns: {sorted(missing)}")

    non_tcp = pl.col("proto").str.to_lowercase() != "tcp"
    zeroed_counts = {
        feature: int(synthetic.filter(non_tcp)[feature].ne(0).sum())
        for feature in UNSW_TCP_ONLY_FEATURES
    }
    result = synthetic.with_columns(
        pl.when(non_tcp).then(pl.lit(0)).otherwise(pl.col(feature)).alias(feature)
        for feature in UNSW_TCP_ONLY_FEATURES
    )
    return result, zeroed_counts


def derive_unsw_packet_means(
    synthetic: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int], dict[str, int]]:
    r"""Derive smean and dmean from packet and byte counts with integer consistency."""
    required = {
        item
        for packet_column, byte_column, mean_column in UNSW_PACKET_BYTE_PAIRS
        for item in (packet_column, byte_column, mean_column)
    }
    missing = required - set(synthetic.columns)
    if missing:
        raise ValueError(f"UNSW candidates are missing packet-mean columns: {sorted(missing)}")

    result = synthetic.clone()
    violations: dict[str, int] = {}
    zero_packet_repairs: dict[str, int] = {}

    for packet_column, byte_column, mean_column in UNSW_PACKET_BYTE_PAIRS:
        packet_counts = result[packet_column].to_numpy().astype(float)
        byte_counts = result[byte_column].to_numpy().astype(float)

        if not np.isfinite(packet_counts).all() or not np.isfinite(byte_counts).all():
            raise ValueError(f"Packet and byte counts for {mean_column} must be finite")
        if (packet_counts < 0).any() or (byte_counts < 0).any():
            raise ValueError(f"Packet and byte counts for {mean_column} must be nonnegative")

        zero_packet_rows = packet_counts == 0
        zero_packet_repairs[byte_column] = int(np.count_nonzero(byte_counts[zero_packet_rows] != 0))
        byte_counts[zero_packet_rows] = 0

        positive_packets = packet_counts > 0
        derived_means = np.zeros_like(packet_counts)
        np.divide(
            byte_counts,
            packet_counts,
            out=derived_means,
            where=positive_packets,
        )
        rounded_means = np.rint(derived_means)

        expected_bounds = np.zeros_like(derived_means)
        expected_bounds[positive_packets] = 0.5 * packet_counts[positive_packets] + 1e-5
        observed_bounds = np.abs(byte_counts - rounded_means * packet_counts)
        pair_violations = int(
            np.count_nonzero(positive_packets & (observed_bounds > expected_bounds))
        )
        violations[mean_column] = pair_violations

        result = result.with_columns(
            pl.Series(byte_column, byte_counts).cast(result.schema[byte_column]),
            pl.Series(mean_column, rounded_means).cast(result.schema[mean_column]),
        )

    return result, violations, zero_packet_repairs


def enforce_domain_invariants(
    df: pl.DataFrame,
    dataset_name: str,
    decimal_precision: dict[str, int] | None = None,
) -> pl.DataFrame:
    r"""Clean up synthetic candidate rows to enforce physical network invariants."""
    result = df.clone()

    # 1. Round to observed source decimal precision
    if decimal_precision:
        round_exprs = [
            pl.col(c).round(places)
            for c, places in decimal_precision.items()
            if c in result.columns
        ]
        if round_exprs:
            result = result.with_columns(round_exprs)

    # 2. Dataset-specific physical invariants
    if dataset_name == "unsw_nb15":
        # TCP-only field zeroing for non-TCP protocols
        if "proto" in result.columns:
            result, _ = zero_unsw_tcp_fields_for_non_tcp(result)

        # Round-trip timing constraint (tcprtt = synack + ackdat <= 3.5s)
        if all(c in result.columns for c in ["synack", "ackdat", "tcprtt"]):
            result, _ = derive_unsw_tcprtt(result)

        # Packet mean derivation consistency
        packet_mean_cols = [item for pair in UNSW_PACKET_BYTE_PAIRS for item in pair]
        if all(c in result.columns for c in packet_mean_cols):
            result, _, _ = derive_unsw_packet_means(result)

    return result
