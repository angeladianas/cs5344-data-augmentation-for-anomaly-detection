"""Fit dataset-specific GMMs and generate the proposal's GMM candidate pools."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

# Support both `python -m models.gmm_candidates` and direct script execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.gmm_baseline import DEFAULT_CANDIDATE_COMPONENTS, fit_gmm_baseline
from utils.boundary_filter import LightGBMBoundaryFilter
from utils.dataset_configs import NSL_KDD, UNSW_NB15
from utils.decoding import TabularDecoder

REPO_ROOT = Path(__file__).resolve().parents[1]
GMM_POOL_SIZES = {"nsl_kdd": 50_000, "unsw_nb15": 100_000}
DATASET_CONFIGS = {"nsl_kdd": NSL_KDD, "unsw_nb15": UNSW_NB15}
BOUNDARY_FILTER_DIR = REPO_ROOT / "results" / "models"
DEFAULT_REJECTION_THRESHOLD = 0.10
UNSW_TCPRTT_MAX_SECONDS = 3.5
UNSW_TCP_ONLY_FEATURES = ("swin", "stcpb", "dtcpb", "dwin")
UNSW_HTTP_FEATURES = ("service", "ct_flw_http_mthd", "response_body_len")
UNSW_STATE_FEATURES = ("proto", "state", "ct_state_ttl")
NSL_KDD_CALIBRATED_FEATURES = (
    "src_bytes",
    "dst_bytes",
    "dst_host_count",
    "dst_host_srv_count",
)
DATASET_SOURCE_DIRS = {"nsl_kdd": "NSL-KDD", "unsw_nb15": "UNSW-NB15"}
RARE_PAIR_FEATURES = ("num_compromised", "num_root")
RARE_PAIR_CONTEXT = ("protocol_type", "service", "flag", "logged_in")


def sample_component_conditional_donors(
    component_ids: np.ndarray,
    responsibilities: np.ndarray,
    donors: pl.DataFrame,
    *,
    random_state: int,
) -> pl.DataFrame:
    """Sample intact discrete tuples according to training-row component responsibilities."""
    component_ids = np.asarray(component_ids)
    responsibilities = np.asarray(responsibilities, dtype=float)
    if component_ids.ndim != 1 or responsibilities.ndim != 2:
        raise ValueError("Component IDs and responsibilities must be one- and two-dimensional")
    if responsibilities.shape[0] != donors.height:
        raise ValueError("Responsibilities and donor rows must have the same row count")
    if (
        not np.isfinite(responsibilities).all()
        or (responsibilities < 0).any()
        or (component_ids < 0).any()
        or (component_ids >= responsibilities.shape[1]).any()
    ):
        raise ValueError("Invalid component IDs or responsibilities")

    rng = np.random.default_rng(random_state)
    selected_rows = np.empty(len(component_ids), dtype=np.int64)
    for component in np.unique(component_ids):
        candidate_positions = np.flatnonzero(component_ids == component)
        probabilities = responsibilities[:, component]
        total = probabilities.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"No training donor probability mass for component {component}")
        selected_rows[candidate_positions] = rng.choice(
            donors.height,
            size=len(candidate_positions),
            replace=True,
            p=probabilities / total,
        )
    return donors[selected_rows]


def calibrate_gmm_feature_marginal(
    values: np.ndarray,
    reference_values: np.ndarray,
) -> np.ndarray:
    """Map generated feature ranks to the empirical fit-data marginal."""
    values = np.asarray(values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    if values.ndim != 1 or reference_values.ndim != 1 or not len(reference_values):
        raise ValueError("Feature values must be vectors with a nonempty reference")
    if not np.isfinite(values).all() or not np.isfinite(reference_values).all():
        raise ValueError("Generated and reference feature values must be finite")
    if not len(values):
        return values.copy()

    generated_order = np.argsort(values, kind="stable")
    quantiles = (np.arange(len(values)) + 0.5) / len(values)
    calibrated = np.empty_like(values)
    calibrated[generated_order] = np.quantile(reference_values, quantiles, method="linear")
    return calibrated


def protocol_group_masks(dataset: str, protocols: list[str]) -> dict[str, np.ndarray]:
    """Partition protocols into the dataset's requested generation groups."""
    normalized = np.char.lower(np.asarray(protocols, dtype=str))
    if dataset == "nsl_kdd":
        groups = {
            "tcp": normalized == "tcp",
            "non_tcp": normalized != "tcp",
        }
    elif dataset == "unsw_nb15":
        groups = {
            "tcp": normalized == "tcp",
            "udp": normalized == "udp",
            "other": (normalized != "tcp") & (normalized != "udp"),
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    assigned = np.zeros(len(normalized), dtype=np.int8)
    for mask in groups.values():
        assigned += mask
    if not np.all(assigned == 1):
        raise ValueError("Protocol groups must partition all training rows exactly once")
    if any(not mask.any() for mask in groups.values()):
        groups = {name: mask for name, mask in groups.items() if mask.any()}
    return groups


def generation_group_masks(
    dataset: str,
    generation_version: str,
    protocols: list[str],
) -> dict[str, np.ndarray]:
    """Return the single V1 group or protocol-specific V2 groups."""
    if generation_version == "v1":
        return {"all": np.ones(len(protocols), dtype=bool)}
    if generation_version == "v2":
        return protocol_group_masks(dataset, protocols)
    raise ValueError(f"Unknown GMM generation version: {generation_version}")


def allocate_group_sample_counts(
    group_masks: dict[str, np.ndarray],
    n_samples: int,
) -> dict[str, int]:
    """Allocate samples proportionally to group sizes using largest remainders."""
    if n_samples < 0:
        raise ValueError("The sample count must be nonnegative")
    if not group_masks:
        raise ValueError("At least one protocol group is required")
    sizes = {name: int(np.count_nonzero(mask)) for name, mask in group_masks.items()}
    if any(size <= 0 for size in sizes.values()):
        raise ValueError("Protocol groups must contain at least one training row")

    total_rows = sum(sizes.values())
    exact_allocations = {name: n_samples * size / total_rows for name, size in sizes.items()}
    allocations = {
        name: int(np.floor(exact_count)) for name, exact_count in exact_allocations.items()
    }
    remaining = n_samples - sum(allocations.values())
    remainders = sorted(
        sizes,
        key=lambda name: exact_allocations[name] - allocations[name],
        reverse=True,
    )
    for name in remainders[:remaining]:
        allocations[name] += 1
    return allocations


def infer_source_decimal_precision(
    source_path: str | Path,
    feature_columns: list[str],
    numeric_columns: list[str],
) -> dict[str, int]:
    """Infer maximum meaningful decimal places for numeric features in the source CSV."""
    precision = {column: 0 for column in numeric_columns}
    with Path(source_path).open(newline="", encoding="utf-8-sig") as source_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None:
            raise ValueError(f"Source dataset has no CSV header: {source_path}")
        missing = set([*feature_columns, *numeric_columns]) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Source dataset is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            for column in numeric_columns:
                value = row[column]
                if value is None or not value.strip():
                    raise ValueError(f"Missing numeric value for {column} at row {row_number}")
                try:
                    decimal_value = Decimal(value.strip())
                except InvalidOperation as error:
                    raise ValueError(
                        f"Invalid numeric value for {column} at row {row_number}: {value}"
                    ) from error
                if not decimal_value.is_finite():
                    raise ValueError(f"Nonfinite numeric value for {column} at row {row_number}")

                digits = list(decimal_value.as_tuple().digits)
                exponent = decimal_value.as_tuple().exponent
                while exponent < 0 and digits and digits[-1] == 0:
                    digits.pop()
                    exponent += 1
                precision[column] = max(precision[column], max(0, -exponent))

    return precision


def align_candidate_precision(
    candidates: pl.DataFrame,
    decimal_places: dict[str, int],
) -> pl.DataFrame:
    """Round numerical output columns to the source dataset's decimal precision."""
    missing = set(decimal_places) - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidates are missing numeric columns: {sorted(missing)}")
    if any(not isinstance(places, int) or places < 0 for places in decimal_places.values()):
        raise ValueError("Decimal-place precision must be a nonnegative integer")
    return candidates.with_columns(
        pl.col(column).round(places) for column, places in decimal_places.items()
    )


def derive_unsw_tcprtt(synthetic: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Derive bounded UNSW TCP round-trip time and count identity violations."""
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
    """Zero TCP window and sequence fields for rows using other protocols."""
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


def calibrate_unsw_response_body_len(
    synthetic: pl.DataFrame,
    real_normal: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, dict[str, float | int]]]:
    """Calibrate response-body lengths separately for HTTP and non-HTTP traffic."""
    for name, frame in (("synthetic", synthetic), ("real_normal", real_normal)):
        missing = set(UNSW_HTTP_FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} data is missing HTTP calibration columns: {sorted(missing)}")
        if frame.height == 0:
            raise ValueError(f"{name} data must not be empty")

    def http_mask(frame: pl.DataFrame) -> np.ndarray:
        r"""Identify rows representing HTTP traffic based on service name or method counts."""
        service_is_http = np.char.lower(np.asarray(frame["service"].to_list(), dtype=str)) == "http"
        method_counts = frame["ct_flw_http_mthd"].to_numpy().astype(float)
        if not np.isfinite(method_counts).all():
            raise ValueError("UNSW HTTP method counts must be finite")
        return service_is_http | (method_counts > 0)

    generated_http = http_mask(synthetic)
    reference_http = http_mask(real_normal)
    generated_values = synthetic["response_body_len"].to_numpy().astype(float)
    reference_values = real_normal["response_body_len"].to_numpy().astype(float)
    if (
        not np.isfinite(generated_values).all()
        or not np.isfinite(reference_values).all()
        or (generated_values < 0).any()
        or (reference_values < 0).any()
    ):
        raise ValueError("UNSW response body lengths must be finite and nonnegative")

    calibrated = generated_values.copy()
    metrics: dict[str, dict[str, float | int]] = {}
    for label, generated_mask, reference_mask in (
        ("http", generated_http, reference_http),
        ("non_http", ~generated_http, ~reference_http),
    ):
        generated_indices = np.flatnonzero(generated_mask)
        reference_group = reference_values[reference_mask]
        if len(generated_indices) and not len(reference_group):
            raise ValueError(f"No development-normal response body lengths for {label} traffic")
        if len(generated_indices):
            calibrated[generated_indices] = calibrate_gmm_feature_marginal(
                generated_values[generated_indices],
                reference_group,
            )
            calibrated[generated_indices] = np.floor(calibrated[generated_indices] + 0.5)
        metrics[label] = {
            "generated_rows": len(generated_indices),
            "reference_rows": len(reference_group),
            "reference_p99": float(np.quantile(reference_group, 0.99)),
            "calibrated_p99": (
                float(np.quantile(calibrated[generated_indices], 0.99))
                if len(generated_indices)
                else 0.0
            ),
        }

    result = synthetic.with_columns(
        pl.Series("response_body_len", calibrated).cast(synthetic.schema["response_body_len"])
    )
    return result, metrics


def reconstruct_unsw_ct_state_ttl(
    synthetic: pl.DataFrame,
    real_normal: pl.DataFrame,
    *,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Match normal zero rates by protocol/state while retaining generated positives."""
    for name, frame in (("synthetic", synthetic), ("real_normal", real_normal)):
        missing = set(UNSW_STATE_FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} data is missing state-count columns: {sorted(missing)}")
        if frame.height == 0:
            raise ValueError(f"{name} data must not be empty")

    zero_values = real_normal["ct_state_ttl"].to_numpy().astype(float)
    if not np.isfinite(zero_values).all() or (zero_values < 0).any():
        raise ValueError("UNSW ct_state_ttl values must be finite and nonnegative")

    exact_lookup = {
        (protocol, state): values
        for protocol, state, values in real_normal.group_by("proto", "state", maintain_order=True)
        .agg(pl.col("ct_state_ttl"))
        .iter_rows()
    }
    protocol_lookup = {
        protocol: values
        for protocol, values in real_normal.group_by("proto", maintain_order=True)
        .agg(pl.col("ct_state_ttl"))
        .iter_rows()
    }
    all_values = zero_values
    result_values = synthetic["ct_state_ttl"].to_numpy().astype(float)
    if not np.isfinite(result_values).all() or (result_values < 0).any():
        raise ValueError("Generated UNSW ct_state_ttl values must be finite and nonnegative")

    group_indices: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(zip(synthetic["proto"].to_list(), synthetic["state"].to_list())):
        group_indices.setdefault(key, []).append(index)

    rng = np.random.default_rng(seed)
    fallback_counts = {"exact_context": 0, "protocol": 0, "global": 0}
    zero_count_before = int(np.count_nonzero(result_values == 0))
    for key, indices_list in group_indices.items():
        indices = np.asarray(indices_list, dtype=np.int64)
        if key in exact_lookup:
            donor_values = np.asarray(exact_lookup[key], dtype=float)
            fallback_counts["exact_context"] += len(indices)
        elif key[0] in protocol_lookup:
            donor_values = np.asarray(protocol_lookup[key[0]], dtype=float)
            fallback_counts["protocol"] += len(indices)
        else:
            donor_values = all_values
            fallback_counts["global"] += len(indices)

        zero_rate = np.count_nonzero(donor_values == 0) / len(donor_values)
        zero_count = int(np.floor(zero_rate * len(indices) + 0.5))
        shuffled = rng.permutation(indices)
        zero_indices = shuffled[:zero_count]
        nonzero_indices = shuffled[zero_count:]
        result_values[zero_indices] = 0

        needs_positive = nonzero_indices[result_values[nonzero_indices] == 0]
        positive_donors = donor_values[donor_values > 0]
        if len(needs_positive):
            if not len(positive_donors):
                raise ValueError(f"No positive ct_state_ttl donor values for context {key}")
            result_values[needs_positive] = positive_donors[
                rng.integers(0, len(positive_donors), size=len(needs_positive))
            ]

    fallback_counts["zero_values_before"] = zero_count_before
    fallback_counts["zero_values_after"] = int(np.count_nonzero(result_values == 0))
    result = synthetic.with_columns(
        pl.Series("ct_state_ttl", result_values).cast(synthetic.schema["ct_state_ttl"])
    )
    return result, fallback_counts


def derive_unsw_packet_means(
    synthetic: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int], dict[str, int]]:
    """Derive integer mean bytes per packet and repair zero-packet byte counts."""
    constraints = (
        ("smean", "sbytes", "spkts"),
        ("dmean", "dbytes", "dpkts"),
    )
    missing = {
        column
        for mean_column, byte_column, packet_column in constraints
        for column in (mean_column, byte_column, packet_column)
        if column not in synthetic.columns
    }
    if missing:
        raise ValueError(f"UNSW candidates are missing packet-mean columns: {sorted(missing)}")

    result = synthetic
    violations = {}
    zero_packet_repairs = {}
    for mean_column, byte_column, packet_column in constraints:
        byte_counts = synthetic[byte_column].to_numpy().astype(float)
        packet_counts = synthetic[packet_column].to_numpy().astype(float)
        if not np.isfinite(byte_counts).all() or not np.isfinite(packet_counts).all():
            raise ValueError(f"Nonfinite byte or packet counts for {mean_column}")
        if (packet_counts < 0).any() or (byte_counts < 0).any():
            raise ValueError(f"Negative byte or packet counts for {mean_column}")
        zero_packet_rows = packet_counts == 0
        zero_packet_repairs[byte_column] = int(np.count_nonzero(byte_counts[zero_packet_rows] != 0))
        byte_counts[zero_packet_rows] = 0

        ratios = np.divide(
            byte_counts,
            packet_counts,
            out=np.zeros_like(byte_counts),
            where=~zero_packet_rows,
        )
        means = np.floor(ratios + 0.5)
        result = result.with_columns(
            pl.Series(byte_column, byte_counts).cast(synthetic.schema[byte_column]),
            pl.Series(mean_column, means.astype(np.int64)),
        )
        normalized_residual = np.divide(
            np.abs(byte_counts - means * packet_counts),
            packet_counts,
            out=np.zeros_like(byte_counts),
            where=~zero_packet_rows,
        )
        violations[mean_column] = int(np.count_nonzero(normalized_residual > 0.5 + 1e-12))
    return result, violations, zero_packet_repairs


def reconstruct_rare_pair(
    synthetic: pd.DataFrame,
    real_normal: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Resample rare feature pairs jointly, conditioned on synthetic context."""
    required = {*RARE_PAIR_FEATURES, *RARE_PAIR_CONTEXT}
    for name, frame in (("synthetic", synthetic), ("real_normal", real_normal)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} data is missing rare-pair columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} data must not be empty")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{name} rare-pair/context columns must not contain missing values")

    lookup = {
        key: group[list(RARE_PAIR_FEATURES)].to_numpy()
        for key, group in real_normal.groupby(list(RARE_PAIR_CONTEXT), sort=False)
    }
    protocol_lookup = {
        key: group[list(RARE_PAIR_FEATURES)].to_numpy()
        for key, group in real_normal.groupby("protocol_type", sort=False)
    }
    global_pairs = real_normal[list(RARE_PAIR_FEATURES)].to_numpy()

    rng = np.random.default_rng(seed)
    result = synthetic.copy()
    for key, indices in result.groupby(list(RARE_PAIR_CONTEXT), sort=False).groups.items():
        pool = lookup.get(key)
        if pool is None:
            pool = protocol_lookup.get(key[0], global_pairs)
        sampled = pool[rng.integers(0, len(pool), size=len(indices))]
        result.loc[indices, list(RARE_PAIR_FEATURES)] = sampled
    return result


def apply_boundary_filter(
    dataset: str,
    candidates: pl.DataFrame,
    output_dir: str | Path,
    *,
    tau_reject: float = DEFAULT_REJECTION_THRESHOLD,
) -> dict:
    """Reject GMM candidates predicted to lie in the anomaly region."""
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not 0 <= tau_reject <= 1:
        raise ValueError("tau_reject must be between 0 and 1")

    model_path = BOUNDARY_FILTER_DIR / f"boundary_filter_{dataset}.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Boundary filter model not found: {model_path}")
    boundary_filter = LightGBMBoundaryFilter.load(model_path)

    candidates_pd = pd.DataFrame(candidates.to_dict(as_series=False))
    if candidates_pd.columns.tolist() != boundary_filter.feature_names:
        raise ValueError("GMM candidate columns do not match the boundary-filter feature order")

    filtered_pd, survival_rate, probabilities = boundary_filter.filter_candidates(
        candidates_pd,
        tau_reject=tau_reject,
    )
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (candidates.height,) or not np.isfinite(probabilities).all():
        raise ValueError("Boundary filter returned invalid candidate probabilities")
    accepted = probabilities <= tau_reject
    if len(filtered_pd) != int(accepted.sum()):
        raise ValueError("Boundary filter result does not match its rejection threshold")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_candidates = candidates.filter(pl.Series("accepted", accepted))
    filtered_candidates.write_csv(output_dir / "filtered_samples.csv")
    filtered_candidates.write_parquet(output_dir / "filtered_samples.parquet")
    pd.DataFrame(
        {
            "candidate_row": np.arange(candidates.height),
            "anomaly_probability": probabilities,
            "accepted": accepted,
        }
    ).to_csv(output_dir / "boundary_filter_scores.csv", index=False)
    return {
        "tau_reject": tau_reject,
        "input_rows": candidates.height,
        "accepted_rows": filtered_candidates.height,
        "rejected_rows": candidates.height - filtered_candidates.height,
        "survival_rate": survival_rate,
    }


def generate_gmm_candidate_pool(
    dataset: str,
    *,
    output_root: str | Path = REPO_ROOT / "output",
    candidate_components: tuple[int, ...] = DEFAULT_CANDIDATE_COMPONENTS,
    random_state: int = 42,
    tau_reject: float = DEFAULT_REJECTION_THRESHOLD,
    generation_version: str = "v2",
) -> dict:
    """Fit versioned GMMs and export the decoded candidate pool."""
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if generation_version not in {"v1", "v2"}:
        raise ValueError(f"Unknown GMM generation version: {generation_version}")

    dataset_dir = Path(output_root) / dataset
    input_dir = dataset_dir / "model_inputs"
    input_metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    if input_metadata["dataset"] != dataset:
        raise ValueError(f"Model-input metadata does not match dataset {dataset}")

    numerical_frame = pd.read_csv(input_dir / input_metadata["csv_files"]["gmm_inputs"])
    feature_columns = input_metadata["numerical_columns"]
    if [c for c in numerical_frame.columns if c != "source_row"] != feature_columns:
        raise ValueError("GMM input columns differ from the stored numerical feature order")
    if len(numerical_frame) != input_metadata["rows"]:
        raise ValueError("GMM input row count differs from its metadata")

    donors = pl.read_parquet(input_dir / input_metadata["gmm"]["donors"])
    manifest = pl.read_parquet(input_dir / input_metadata["provenance"]["manifest"])
    if donors.height != len(numerical_frame) or manifest.height != len(numerical_frame):
        raise ValueError("GMM inputs, discrete donors, and provenance rows are misaligned")
    if not np.array_equal(
        numerical_frame["source_row"].to_numpy(),
        manifest["source_row"].to_numpy(),
    ):
        raise ValueError("GMM input rows do not match the saved provenance order")

    n_samples = GMM_POOL_SIZES[dataset]
    protocol_column = "protocol_type" if dataset == "nsl_kdd" else "proto"
    group_masks = generation_group_masks(
        dataset,
        generation_version,
        donors[protocol_column].to_list(),
    )
    group_sample_counts = allocate_group_sample_counts(group_masks, n_samples)
    calibrated_features = NSL_KDD_CALIBRATED_FEATURES if dataset == "nsl_kdd" else ()
    decoder = TabularDecoder.from_directory(input_dir)
    features = input_metadata["feature_order"]
    generated_frames = []
    generated_group_labels = []
    generated_component_ids = []
    models = {}
    group_info = {}
    for group_index, (group_name, group_mask) in enumerate(group_masks.items()):
        group_numerical = numerical_frame.loc[group_mask].reset_index(drop=True)
        group_donors = donors.filter(pl.Series("in_group", group_mask))
        if len(group_numerical) != group_donors.height:
            raise ValueError(f"Numerical and donor rows are misaligned for group {group_name}")

        model, info = fit_gmm_baseline(
            group_numerical,
            candidate_components=candidate_components,
            random_state=random_state + group_index,
        )
        group_n_samples = group_sample_counts[group_name]
        synthetic_numeric, component_ids = model.sample(n_samples=group_n_samples)
        for feature in calibrated_features:
            feature_index = info["features"].index(feature)
            synthetic_numeric[:, feature_index] = calibrate_gmm_feature_marginal(
                synthetic_numeric[:, feature_index],
                group_numerical[feature].to_numpy(dtype=float),
            )

        responsibilities = model.predict_proba(
            group_numerical[info["features"]].to_numpy(dtype=float)
        )
        synthetic_donors = sample_component_conditional_donors(
            component_ids,
            responsibilities,
            group_donors,
            random_state=random_state + group_index,
        )
        group_decoded = decoder.decode_gmm(synthetic_numeric, synthetic_donors)
        if group_decoded.columns != features or group_decoded.height != group_n_samples:
            raise ValueError(
                f"Decoded GMM samples for {group_name} do not match the expected schema"
            )
        generated_frames.append(group_decoded)
        generated_group_labels.extend([group_name] * group_n_samples)
        generated_component_ids.extend(component_ids.tolist())
        models[group_name] = model
        group_info[group_name] = {
            "training_rows": int(group_mask.sum()),
            "training_proportion": float(group_mask.mean()),
            "generated_rows": group_n_samples,
            "n_components": info["n_components"],
            "best_bic": info["bic"],
            "best_aic": info["aic"],
            "converged": info["converged"],
            "n_iter": info["n_iter"],
            "criteria": info["criteria"],
        }

    decoded = pl.concat(generated_frames, how="vertical")
    component_ids = np.asarray(generated_component_ids, dtype=np.int64)
    if decoded.columns != features or decoded.height != n_samples:
        raise ValueError("Decoded GMM samples do not match the expected dataset schema")
    if decoded.null_count().sum_horizontal().sum() != 0:
        raise ValueError("Decoded GMM samples contain missing values")

    source_path = REPO_ROOT / "data" / DATASET_SOURCE_DIRS[dataset] / "train.csv"
    numeric_columns = [
        column for column in features if column not in DATASET_CONFIGS[dataset].categorical
    ]
    decimal_places = infer_source_decimal_precision(source_path, features, numeric_columns)
    decoded = align_candidate_precision(decoded, decimal_places)
    timing_violations = None
    tcp_only_zeroed = None
    response_body_calibration = None
    ct_state_ttl_reconstruction = None
    packet_mean_violations = None
    zero_packet_repairs = None
    if dataset == "unsw_nb15":
        source_data = pl.read_csv(source_path).with_row_index("source_row")
        real_normal = source_data.filter(
            pl.col("source_row").is_in(manifest["source_row"].to_list())
        )
        if real_normal.height != manifest.height or real_normal["is_anomaly"].ne(0).any():
            raise ValueError("UNSW reconstruction donors must be development-normal records")
        decoded, response_body_calibration = calibrate_unsw_response_body_len(
            decoded,
            real_normal,
        )
        decoded, ct_state_ttl_reconstruction = reconstruct_unsw_ct_state_ttl(
            decoded,
            real_normal,
            seed=random_state,
        )
        decoded, tcp_only_zeroed = zero_unsw_tcp_fields_for_non_tcp(decoded)
        print(f"Non-TCP TCP-only fields zeroed: {sum(tcp_only_zeroed.values())}")
        decoded, timing_violations = derive_unsw_tcprtt(decoded)
        print(f"Timing violations: {timing_violations}")
        decoded, packet_mean_violations, zero_packet_repairs = derive_unsw_packet_means(decoded)
        for feature, count in packet_mean_violations.items():
            print(f"{feature} integer byte/packet consistency violations: {count}")
        print(f"Zero-packet byte-count repairs: {sum(zero_packet_repairs.values())}")

    rare_pair_info = None
    if dataset == "nsl_kdd":
        source_data = pl.read_csv(source_path).with_row_index("source_row")
        real_normal = source_data.filter(
            pl.col("source_row").is_in(manifest["source_row"].to_list())
        )
        if real_normal.height != manifest.height or real_normal["is_anomaly"].ne(0).any():
            raise ValueError("Rare-pair donor rows must be aligned development-normal records")
        synthetic_pd = pd.DataFrame(decoded.to_dict(as_series=False))
        real_normal_pd = pd.DataFrame(
            real_normal.select([*RARE_PAIR_CONTEXT, *RARE_PAIR_FEATURES]).to_dict(as_series=False)
        )
        reconstructed = reconstruct_rare_pair(
            synthetic_pd,
            real_normal_pd,
            seed=random_state,
        )
        decoded = pl.DataFrame(
            [
                pl.Series(column, reconstructed[column].tolist()).cast(decoded.schema[column])
                for column in decoded.columns
            ]
        )
        rare_pair_info = {
            "features": list(RARE_PAIR_FEATURES),
            "context": list(RARE_PAIR_CONTEXT),
            "donors": "development-normal source rows",
            "fallback": "same protocol, then global development normals",
            "seed": random_state,
        }

    output_dir = dataset_dir / f"gmm_candidates_{generation_version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    decoded.write_csv(output_dir / "synthetic_samples.csv")
    decoded.write_parquet(output_dir / "synthetic_samples.parquet")
    pd.DataFrame(
        {
            "generation_group": generated_group_labels,
            "component": component_ids,
        }
    ).to_csv(output_dir / "component_assignments.csv", index=False)
    joblib.dump(
        models["all"] if generation_version == "v1" else models,
        output_dir / "gmm_model.joblib",
    )
    boundary_filter_info = apply_boundary_filter(
        dataset,
        decoded,
        output_dir,
        tau_reject=tau_reject,
    )

    metadata = {
        "dataset": dataset,
        "generation_version": generation_version,
        "fit_partition": input_metadata["fit_partition"],
        "fit_rows": len(numerical_frame),
        "feature_order": features,
        "numerical_features": feature_columns,
        "selection_criterion": (
            "lowest BIC among converged candidate fits"
            if generation_version == "v1"
            else "lowest BIC among converged candidate fits within each protocol group"
        ),
        "generation_groups": group_info,
        "n_components": {name: values["n_components"] for name, values in group_info.items()},
        "best_bic": {name: values["best_bic"] for name, values in group_info.items()},
        "best_aic": {name: values["best_aic"] for name, values in group_info.items()},
        "converged": all(values["converged"] for values in group_info.values()),
        "n_iter": {name: values["n_iter"] for name, values in group_info.items()},
        "candidate_components": candidate_components,
        "criteria": {name: values["criteria"] for name, values in group_info.items()},
        "random_state": random_state,
        "synthetic_rows": n_samples,
        "candidate_pool_fraction": 0.30,
        "donor_sampling": (
            "protocol-group-local intact discrete tuples sampled with probabilities "
            "proportional to training-row posterior responsibility for the generated component"
        ),
        "protocol_group_sampling": {
            "protocol_column": protocol_column,
            "version": generation_version,
            "groups": group_info,
            "allocation": (
                "all samples use the single ungrouped model"
                if generation_version == "v1"
                else "generated row counts use largest-remainder proportional allocation from development-normal training group sizes"
            ),
            "component_assignments_include_group": True,
        },
        "numerical_decoding": "inverse-transformed, rounded where required, and clipped to normal bounds",
        "marginal_calibration": (
            {
                "features": list(calibrated_features),
                "reference": "development-normal GMM input marginal",
                "method": "generated-row ranks mapped to empirical fit-data quantiles",
            }
            if calibrated_features
            else None
        ),
        "source_decimal_places": decimal_places,
        "derived_feature_constraints": (
            {
                "tcprtt": "synack + ackdat",
                "tcprtt_max_seconds": UNSW_TCPRTT_MAX_SECONDS,
                "tcprtt_bound_policy": (
                    "scale synack and ackdat proportionally when their sum exceeds "
                    "the maximum, preserving their sum identity"
                ),
                "timing_violations": timing_violations,
                "violation_tolerance": 1e-5,
                "smean": "nearest integer to sbytes / spkts",
                "dmean": "nearest integer to dbytes / dpkts; zero when both are zero",
                "packet_mean_violations": packet_mean_violations,
                "packet_mean_violation_tolerance": 0.5,
                "derived_means_rounding": "nearest integer, matching source representation",
                "packet_mean_consistency": (
                    "absolute bytes-minus-mean-times-packets residual is at most "
                    "half the packet count"
                ),
                "zero_packet_byte_count_repairs": zero_packet_repairs,
                "zero_packet_byte_count_policy": "set bytes to zero when packet count is zero",
            }
            if dataset == "unsw_nb15"
            else None
        ),
        "response_body_len_calibration": (
            {
                "context": "HTTP service or positive ct_flw_http_mthd; separately calibrated from non-HTTP traffic",
                "reference": "development-normal source rows",
                "method": "generated-rank empirical quantile mapping within HTTP/non-HTTP groups, rounded to integer",
                "groups": response_body_calibration,
            }
            if dataset == "unsw_nb15"
            else None
        ),
        "ct_state_ttl_reconstruction": (
            {
                "context": ["proto", "state"],
                "reference": "development-normal source rows",
                "method": "match the nearest representable normal zero count per context; preserve generated nonzero values and donor-sample any newly positive values",
                "fallback": "same protocol, then global development normals",
                "counts": ct_state_ttl_reconstruction,
            }
            if dataset == "unsw_nb15"
            else None
        ),
        "tcp_only_fields": (
            {
                "protocol_column": "proto",
                "tcp_protocol_value": "tcp",
                "fields": list(UNSW_TCP_ONLY_FEATURES),
                "non_tcp_nonzero_values_zeroed": tcp_only_zeroed,
            }
            if dataset == "unsw_nb15"
            else None
        ),
        "rare_pair_reconstruction": rare_pair_info,
        "boundary_filter": {
            **boundary_filter_info,
            "model": f"results/models/boundary_filter_{dataset}.pkl",
            "rejection_rule": "reject candidates with predicted anomaly probability > tau_reject",
            "outputs": {
                "filtered_samples_csv": "filtered_samples.csv",
                "filtered_samples_parquet": "filtered_samples.parquet",
                "candidate_scores_csv": "boundary_filter_scores.csv",
            },
        },
        "outputs": {
            "samples_csv": "synthetic_samples.csv",
            "samples_parquet": "synthetic_samples.parquet",
            "filtered_samples_csv": "filtered_samples.csv",
            "filtered_samples_parquet": "filtered_samples.parquet",
            "boundary_filter_scores_csv": "boundary_filter_scores.csv",
            "component_assignments_csv": "component_assignments.csv",
            "model": "gmm_model.joblib",
        },
        "final_submission": False,
        "note": "GMM candidates only; combine with TabDDPM candidates and apply boundary filtering before submission.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "model": models["all"] if generation_version == "v1" else models,
        "metadata": metadata,
        "output_dir": output_dir,
        "boundary_filter": boundary_filter_info,
    }


def main(*, generation_version: str | None = None) -> None:
    """Run the selected generation version for the requested datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default=generation_version or "v2",
        help="GMM generation version: 'v1' (ungrouped) or 'v2' (protocol-grouped, default: v2).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_CONFIGS),
        default=tuple(DATASET_CONFIGS),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--tau-reject",
        type=float,
        default=DEFAULT_REJECTION_THRESHOLD,
        help="Reject candidates with predicted anomaly probability above this threshold.",
    )
    args = parser.parse_args()
    chosen_version = args.version or generation_version or "v2"

    for dataset in args.datasets:
        result = generate_gmm_candidate_pool(
            dataset,
            random_state=args.random_state,
            tau_reject=args.tau_reject,
            generation_version=chosen_version,
        )
        metadata = result["metadata"]
        group_summaries = ", ".join(
            f"{name}:K={group['n_components']},BIC={group['best_bic']:.3f},"
            f"AIC={group['best_aic']:.3f},n={group['generated_rows']}"
            for name, group in metadata["generation_groups"].items()
        )
        print(
            f"{dataset} {generation_version}: groups=[{group_summaries}], "
            f"samples={metadata['synthetic_rows']}, "
            f"boundary_filter_survivors="
            f"{metadata['boundary_filter']['accepted_rows']}/"
            f"{metadata['boundary_filter']['input_rows']}, "
            f"saved={result['output_dir']}"
        )


if __name__ == "__main__":
    main()
