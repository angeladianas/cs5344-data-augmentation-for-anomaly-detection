"""Shared preprocessing; no generator or classifier training occurs here."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler

# Direct execution puts utils/ on sys.path; package imports need its parent.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dataset_configs import NSL_KDD, UNSW_NB15, DatasetConfig
from utils.decoding import TabularDecoder


def make_development_split(frame, feature_columns, label_column, validation_fraction, seed):
    """Split exact feature groups; return original rows and a separate row manifest."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    reserved = {"source_row", "feature_group", "partition"}
    if reserved.intersection(frame.columns):
        raise ValueError("Input contains reserved split metadata columns")
    grouped = (
        frame.with_row_index("source_row")
        .group_by(feature_columns)
        .agg(
            pl.col("source_row").min().alias("feature_group"),
            pl.col(label_column).first().alias("group_label"),
            pl.col(label_column).n_unique().alias("n_labels"),
        )
        .sort("feature_group")
    )
    if grouped.filter(pl.col("n_labels") != 1).height:
        raise ValueError("Identical feature rows have conflicting labels; review before splitting")
    development_ids, validation_ids = train_test_split(
        grouped["feature_group"].to_numpy(),
        test_size=validation_fraction,
        random_state=seed,
        stratify=grouped["group_label"].to_numpy(),
    )
    group_assignments = grouped.select([*feature_columns, "feature_group"]).with_columns(
        pl.when(pl.col("feature_group").is_in(validation_ids.tolist()))
        .then(pl.lit("validation"))
        .otherwise(pl.lit("development"))
        .alias("partition")
    )
    assigned = (
        frame.with_row_index("source_row")
        .join(group_assignments, on=feature_columns, how="left", validate="m:1")
        .sort("source_row")
    )
    manifest = assigned.select("source_row", "feature_group", label_column, "partition")
    development = assigned.filter(pl.col("partition") == "development").select(frame.columns)
    validation = assigned.filter(pl.col("partition") == "validation").select(frame.columns)
    assert set(development_ids).isdisjoint(validation_ids)
    return (development, validation, manifest)


def encode_categories(frame, vocabularies, allow_unknown=False):
    """Return category codes without fitting or extending a vocabulary."""
    encoded = []
    for c, vocabulary in vocabularies.items():
        mapping = {value: index for index, value in enumerate(vocabulary)}
        unknown = sorted(set(frame[c].to_list()) - set(vocabulary))
        if unknown and (not allow_unknown):
            raise ValueError(f"Unknown generator categories in {c}: {unknown}")
        encoded.append(frame[c].replace_strict(mapping, default=-1, return_dtype=pl.Int64).alias(c))
    return pl.DataFrame(encoded)


def float32_inverse_error_bound(transformer, original, integer_required):
    """Bound inverse error by adjacent representable float32 coordinates.

    Quantile tails and large-offset scalers can amplify float32 rounding.
    Compare per row rather than applying one broad tolerance to a column.
    """
    original = np.asarray(original, dtype=np.float64)
    coordinates = transformer.transform(original.reshape(-1, 1)).astype(np.float32)
    lower = np.nextafter(coordinates, np.float32(-np.inf)).astype(np.float64)
    upper = np.nextafter(coordinates, np.float32(np.inf)).astype(np.float64)
    low_values = transformer.inverse_transform(lower).ravel()
    high_values = transformer.inverse_transform(upper).ravel()
    bound = np.maximum(np.abs(low_values - original), np.abs(high_values - original))
    return bound + (1.0 if integer_required else 1e-8)


def run_preprocessing(
    input_path: str | Path,
    config: DatasetConfig,
    output_root: str | Path = "output",
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
    skew_threshold: float = 2.0,
    max_quantiles: int = 1000,
):
    """Audit, split, fit, adapt, decode, and export a configured dataset.

    Return tables, arrays, metadata, and paths for interactive inspection.
    Re-running overwrites only this dataset's output directory. Generator rules
    use development normals; boundary vocabularies use all development rows.
    Validation is never fitted. Zero indicators remain an optional later experiment.
    """
    train_path = Path(input_path).resolve()
    output_dir = Path(output_root).resolve() / config.name
    train_df = pl.read_csv(train_path, has_header=True)
    LABEL = config.label
    CATEGORICAL = list(config.categorical)
    BINARY = list(config.binary)
    DISCRETE = config.discrete
    COUNTS = list(config.counts)
    RATES = list(config.bounded_rates)
    features = [c for c in train_df.columns if c != LABEL]
    numerical = [c for c in features if c not in CATEGORICAL]
    semantic_type = config.semantic_type
    if len(config.features) != len(set(config.features)):
        raise ValueError("Configuration assigns a feature more than once")
    if set(train_df.columns) != set([*config.features, LABEL]):
        raise ValueError("CSV columns do not match the configured schema")
    if (
        not train_df.height
        or train_df[LABEL].null_count()
        or set(train_df[LABEL].to_list()) != {0, 1}
    ):
        raise ValueError("Expected a nonempty dataset with binary labels 0 and 1")
    if not all(train_df.schema[c] == pl.String for c in CATEGORICAL):
        raise ValueError("Categorical columns must contain strings")
    if not all(train_df.schema[c].is_numeric() for c in numerical):
        raise ValueError("Numerical columns have unexpected dtypes")
    if max_quantiles < 2 or not np.isfinite(skew_threshold) or skew_threshold < 0:
        raise ValueError("Invalid numerical transformation settings")
    normal_df = train_df.filter(pl.col(LABEL) == 0)
    attack_df = train_df.filter(pl.col(LABEL) == 1)
    feature_catalogue = pl.DataFrame(
        [
            {
                "position": i,
                "feature": c,
                "dtype": str(train_df.schema[c]),
                "semantic_type": semantic_type(c),
                "n_unique_all": train_df[c].n_unique(),
                "n_unique_normal": normal_df[c].n_unique(),
                "constant_normal": normal_df[c].n_unique() == 1,
                "normal_constant_value": str(normal_df[c][0])
                if normal_df[c].n_unique() == 1
                else None,
            }
            for i, c in enumerate(features)
        ]
    )
    quality_rows = []
    for c in train_df.columns:
        s = train_df[c]
        nonfinite = int((~s.is_finite()).fill_null(False).sum()) if s.dtype.is_numeric() else 0
        blank = (
            int((s.str.strip_chars() == "").fill_null(False).sum()) if s.dtype == pl.String else 0
        )
        violations = 0
        if c in BINARY:
            violations = int((~s.is_in([0, 1])).fill_null(False).sum())
        elif c in DISCRETE:
            violations = int((~s.is_in(DISCRETE[c])).fill_null(False).sum())
        elif c in RATES:
            violations = int((~s.is_between(0, 1)).fill_null(False).sum())
        elif c in config.continuous:
            violations = int((s < 0).fill_null(False).sum())
        elif c in COUNTS:
            violations = int(((s < 0) | (s != s.floor())).fill_null(False).sum())
        quality_rows.append(
            {
                "column": c,
                "nulls": s.null_count(),
                "nonfinite": nonfinite,
                "blank_strings": blank,
                "domain_violations": violations,
            }
        )
    quality_report = pl.DataFrame(quality_rows)
    assert (
        quality_report.select(
            pl.sum_horizontal("nulls", "nonfinite", "blank_strings", "domain_violations").sum()
        ).item()
        == 0
    ), "Resolve data quality issues before proceeding"
    duplicate_groups = (
        train_df.group_by(features)
        .agg(pl.len().alias("n_rows"), pl.col(LABEL).n_unique().alias("n_labels"))
        .filter(pl.col("n_rows") > 1)
    )
    duplicate_summary = {
        "exact_duplicate_extra_rows": train_df.height - train_df.unique().height,
        "feature_duplicate_extra_rows": train_df.height - train_df.select(features).unique().height,
        "feature_duplicate_groups": duplicate_groups.height,
        "conflicting_label_groups": duplicate_groups.filter(pl.col("n_labels") > 1).height,
    }
    if duplicate_groups.height:
        pass
    numeric_rows = []
    for name, subset in [("normal", normal_df), ("attack", attack_df)]:
        for c in numerical:
            s = subset[c]
            numeric_rows.append(
                {
                    "class": name,
                    "feature": c,
                    "semantic_type": semantic_type(c),
                    "n_unique": s.n_unique(),
                    "min": float(s.min()),
                    "p01": float(s.quantile(0.01)),
                    "p25": float(s.quantile(0.25)),
                    "median": float(s.median()),
                    "p75": float(s.quantile(0.75)),
                    "p99": float(s.quantile(0.99)),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "skewness": float(s.skew()) if s.n_unique() > 1 else None,
                    "zero_fraction": float((s == 0).mean()),
                }
            )
    numerical_summary = pl.DataFrame(numeric_rows)
    category_rows = []
    for c in CATEGORICAL:
        normal_counts = dict(normal_df.group_by(c).len().iter_rows())
        attack_counts = dict(attack_df.group_by(c).len().iter_rows())
        for value in sorted(set(normal_counts) | set(attack_counts)):
            n = normal_counts.get(value, 0)
            a = attack_counts.get(value, 0)
            category_rows.append(
                {
                    "feature": c,
                    "category": value,
                    "normal_count": n,
                    "attack_count": a,
                    "normal_fraction": n / normal_df.height,
                    "attack_fraction": a / attack_df.height,
                    "support": "shared" if n and a else "normal_only" if n else "attack_only",
                }
            )
    category_frequencies = pl.DataFrame(category_rows)
    category_support = (
        category_frequencies.group_by("feature", "support").len().sort("feature", "support")
    )
    print(f"Normal: {normal_df.height:,}; attack: {attack_df.height:,}")
    print(
        "Constant normal features:",
        feature_catalogue.filter(pl.col("constant_normal"))["feature"].to_list(),
    )
    print("Duplicate audit:", duplicate_summary)
    SPLIT_SEED = seed
    VALIDATION_FRACTION = validation_fraction
    development_df, validation_df, split_manifest = make_development_split(
        train_df, features, LABEL, VALIDATION_FRACTION, SPLIT_SEED
    )
    development_normal_df = development_df.filter(pl.col(LABEL) == 0)
    development_attack_df = development_df.filter(pl.col(LABEL) == 1)
    validation_normal_df = validation_df.filter(pl.col(LABEL) == 0)
    validation_attack_df = validation_df.filter(pl.col(LABEL) == 1)
    X_generator_raw = development_normal_df.select(features)
    X_boundary_raw = development_df.select(features)
    y_boundary = development_df[LABEL]
    X_validation_raw = validation_df.select(features)
    y_validation = validation_df[LABEL]
    split_summary = (
        split_manifest.group_by("partition", LABEL)
        .len()
        .sort("partition", LABEL)
        .with_columns(
            (pl.col("len") / pl.col("len").sum().over("partition")).alias("class_fraction"),
            (pl.col("len") / pl.col("len").sum().over(LABEL)).alias("fraction_of_class"),
        )
    )
    print(f"Generator fitting rows: {X_generator_raw.height:,} (normal only)")
    print(f"Boundary fitting rows: {X_boundary_raw.height:,} (both classes)")
    print(f"Validation rows: {X_validation_raw.height:,} (both classes)")
    assert development_df.height + validation_df.height == train_df.height
    assert development_df.schema == validation_df.schema == train_df.schema
    assert split_manifest["source_row"].n_unique() == train_df.height
    assert split_manifest["partition"].null_count() == 0
    assert (
        split_manifest.group_by("feature_group")
        .agg(pl.col("partition").n_unique().alias("partitions"))["partitions"]
        .max()
        == 1
    )
    assert (
        development_df.select(features)
        .unique()
        .join(validation_df.select(features).unique(), on=features, how="inner")
        .height
        == 0
    ), "Feature rows overlap across partitions"
    assert set(development_df[LABEL].unique().to_list()) == {0, 1}
    assert set(validation_df[LABEL].unique().to_list()) == {0, 1}
    assert X_generator_raw.columns == X_boundary_raw.columns == X_validation_raw.columns == features
    assert development_normal_df[LABEL].eq(0).all()
    _, _, repeated_manifest = make_development_split(
        train_df, features, LABEL, VALIDATION_FRACTION, SPLIT_SEED
    )
    assert split_manifest.equals(repeated_manifest), "Split is not reproducible"
    support_rows = []
    for c in CATEGORICAL:
        generator_support = set(development_normal_df[c].to_list())
        boundary_support = set(development_df[c].to_list())
        support_rows.append(
            {
                "feature": c,
                "development_normal_categories": len(generator_support),
                "development_all_categories": len(boundary_support),
                "validation_normal_absent_from_generator": sorted(
                    set(validation_normal_df[c]) - generator_support
                ),
                "validation_all_absent_from_boundary": sorted(
                    set(validation_df[c]) - boundary_support
                ),
            }
        )
    split_category_support = pl.DataFrame(support_rows)
    print("Passed coverage, leakage, class, schema, and reproducibility checks.")
    split_output = output_dir / "splits"
    split_output.mkdir(parents=True, exist_ok=True)
    raw_partitions = {
        "development": development_df,
        "validation": validation_df,
        "development_normal": development_normal_df,
        "development_attack": development_attack_df,
    }
    for name, frame in raw_partitions.items():
        frame.write_parquet(split_output / f"{name}.parquet")
        frame.write_csv(split_output / f"{name}.csv")
    split_manifest.write_parquet(split_output / "row_manifest.parquet")
    split_manifest.write_csv(split_output / "row_manifest.csv")
    split_metadata = {
        "source": str(train_path),
        "source_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "source_rows": train_df.height,
        "seed": SPLIT_SEED,
        "requested_validation_fraction": VALIDATION_FRACTION,
        "actual_validation_fraction": validation_df.height / train_df.height,
        "method": "class-stratified train_test_split over exact feature groups",
        "group_id": "minimum zero-based source data-row index in each exact feature group",
        "feature_order": features,
        "label": LABEL,
        "versions": {"polars": pl.__version__, "scikit_learn": sklearn.__version__},
        "counts": split_summary.to_dicts(),
        "files": {name: f"{name}.parquet" for name in raw_partitions},
        "manifest": "row_manifest.parquet",
        "csv_files": {name: f"{name}.csv" for name in raw_partitions},
        "csv_manifest": "row_manifest.csv",
    }
    (split_output / "split_metadata.json").write_text(
        json.dumps(split_metadata, indent=2) + "\n", encoding="utf-8"
    )
    for name, frame in raw_partitions.items():
        assert pl.read_parquet(split_output / f"{name}.parquet").equals(frame)
    assert pl.read_parquet(split_output / "row_manifest.parquet").equals(split_manifest)
    print(f"Saved and verified split artifacts: {split_output}")
    SKEW_THRESHOLD = skew_threshold
    MAX_QUANTILES = max_quantiles
    numeric_transformers = {}
    column_rules = {}
    generator_vocabularies = {}
    boundary_vocabularies = {}
    for c in features:
        s = development_normal_df[c]
        kind = semantic_type(c)
        constant = s.n_unique() == 1
        rule = {
            "dtype": str(train_df.schema[c]),
            "semantic_type": kind,
            "constant_normal": constant,
            "constant_value": s[0] if constant else None,
            "n_unique_normal": s.n_unique(),
        }
        if c in CATEGORICAL:
            vocabulary = sorted(s.unique().to_list())
            generator_vocabularies[c] = vocabulary
            boundary_vocabularies[c] = sorted(development_df[c].unique().to_list())
            rule.update(
                transform="restore_constant" if constant else "categorical",
                vocabulary=vocabulary,
                cardinality=len(vocabulary),
                frequencies=dict(s.value_counts().iter_rows()),
                domain={"allowed_values": vocabulary, "basis": "development_normal_support"},
            )
        else:
            values = s.to_numpy().astype(np.float64)
            skewness = float(s.skew()) if not constant else None
            if skewness is not None and (not np.isfinite(skewness)):
                skewness = None
            domain = {"min": 0}
            if c in BINARY:
                domain = {"allowed_values": [0, 1]}
            elif c in DISCRETE:
                domain = {"allowed_values": DISCRETE[c]}
            elif c in RATES:
                domain = {"min": 0, "max": 1}
            rule.update(
                empirical_normal_bounds={"min": float(s.min()), "max": float(s.max())},
                domain=domain,
                integer_required=c in COUNTS or c in BINARY or c in DISCRETE,
                zero_fraction=float((s == 0).mean()),
                skewness=skewness,
                zero_indicator_candidate=not constant
                and kind in {"count", "continuous", "bounded_rate"}
                and (0.5 <= float((s == 0).mean()) < 1),
            )
            if constant:
                rule["transform"] = "restore_constant"
            elif c in BINARY or c in DISCRETE:
                rule["transform"] = "identity_discrete"
            else:
                use_quantile = (
                    skewness is not None and abs(skewness) > SKEW_THRESHOLD and (s.n_unique() > 2)
                )
                transformer = (
                    QuantileTransformer(
                        n_quantiles=min(MAX_QUANTILES, len(values)),
                        output_distribution="normal",
                        subsample=len(values),
                        random_state=SPLIT_SEED,
                    )
                    if use_quantile
                    else StandardScaler()
                )
                transformer.fit(values.reshape(-1, 1))
                numeric_transformers[c] = transformer
                rule["transform"] = "quantile_normal" if use_quantile else "standard"
        column_rules[c] = rule
    constant_features = [c for c in features if column_rules[c]["constant_normal"]]
    active_numerical_features = [c for c in numerical if c in numeric_transformers]
    active_discrete_features = [
        c for c in features if column_rules[c]["transform"] == "identity_discrete"
    ]
    active_categorical_features = [c for c in CATEGORICAL if c not in constant_features]
    preprocessing_summary = pl.DataFrame(
        [
            {
                "feature": c,
                "semantic_type": column_rules[c]["semantic_type"],
                "transform": column_rules[c]["transform"],
                "constant_normal": column_rules[c]["constant_normal"],
                "zero_fraction": column_rules[c].get("zero_fraction"),
                "skewness": column_rules[c].get("skewness"),
            }
            for c in features
        ]
    )
    print("Normal fitting rows:", development_normal_df.height)
    print("Constant features:", constant_features)
    print("Fitted numerical transformers:", len(numeric_transformers))

    def decode_generator_categories(encoded):
        decoded = []
        for c, vocabulary in generator_vocabularies.items():
            s = encoded[c]
            if (
                s.null_count()
                or not s.dtype.is_integer()
                or (not s.is_between(0, len(vocabulary) - 1).all())
            ):
                raise ValueError(f"Invalid generator category codes in {c}")
            decoded.append(
                s.replace_strict(dict(enumerate(vocabulary)), return_dtype=pl.String).alias(c)
            )
        return pl.DataFrame(decoded)

    unknown_category_report = pl.DataFrame(
        [
            {
                "feature": c,
                "generator_cardinality": len(generator_vocabularies[c]),
                "boundary_cardinality": len(boundary_vocabularies[c]),
                "validation_unknown_boundary_rows": int(
                    (~validation_df[c].is_in(boundary_vocabularies[c])).sum()
                ),
                "validation_unknown_normal_rows": int(
                    (~validation_normal_df[c].is_in(generator_vocabularies[c])).sum()
                ),
            }
            for c in CATEGORICAL
        ]
    )
    assert development_normal_df[LABEL].eq(0).all()
    assert set(numeric_transformers) == set(active_numerical_features)
    for c, vocabulary in generator_vocabularies.items():
        assert vocabulary == sorted(development_normal_df[c].unique().to_list())
        assert boundary_vocabularies[c] == sorted(development_df[c].unique().to_list())
    encoded_normal = encode_categories(development_normal_df, generator_vocabularies)
    assert decode_generator_categories(encoded_normal).equals(
        development_normal_df.select(CATEGORICAL)
    )
    for c, transformer in numeric_transformers.items():
        original = development_normal_df[c].to_numpy().astype(np.float64).reshape(-1, 1)
        encoded = transformer.transform(original)
        restored = transformer.inverse_transform(encoded)
        assert np.isfinite(encoded).all() and np.isfinite(restored).all(), c
        if column_rules[c]["integer_required"]:
            restored = np.rint(restored)
        assert np.allclose(restored, original, rtol=1e-07, atol=1e-07), f"Round-trip mismatch: {c}"
        if isinstance(transformer, StandardScaler):
            assert transformer.n_samples_seen_ == development_normal_df.height
        else:
            assert transformer.quantiles_[0, 0] == original.min()
            assert transformer.quantiles_[-1, 0] == original.max()
    for c in constant_features:
        assert (development_normal_df[c] == column_rules[c]["constant_value"]).all()
    probe = (
        development_normal_df.select(CATEGORICAL)
        .head(1)
        .with_columns(pl.lit("__UNSEEN_PROTOCOL_TEST__").alias(CATEGORICAL[0]))
    )
    assert "__UNSEEN_PROTOCOL_TEST__" not in generator_vocabularies[CATEGORICAL[0]]
    assert (
        encode_categories(probe, boundary_vocabularies, allow_unknown=True)[CATEGORICAL[0]][0] == -1
    )
    try:
        encode_categories(probe, generator_vocabularies)
    except ValueError:
        pass
    else:
        raise AssertionError("Generator accepted an unseen category")
    print("Passed fitted-support, numerical/category round-trip, and unknown-category checks.")
    preprocessing_output = output_dir / "preprocessing"
    preprocessing_output.mkdir(parents=True, exist_ok=True)
    preprocessing_metadata = {
        "dataset": config.name,
        "artifact_version": 1,
        "fit_partition": "development_normal",
        "fit_rows": development_normal_df.height,
        "boundary_fit_partition": "development",
        "boundary_fit_rows": development_df.height,
        "source_sha256": split_metadata["source_sha256"],
        "split_manifest_sha256": hashlib.sha256(
            (split_output / "row_manifest.parquet").read_bytes()
        ).hexdigest(),
        "split_seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "feature_order": features,
        "label": LABEL,
        "constant_features": constant_features,
        "active_numerical_features": active_numerical_features,
        "active_discrete_features": active_discrete_features,
        "active_categorical_features": active_categorical_features,
        "columns": column_rules,
        "generator_vocabularies": generator_vocabularies,
        "boundary_vocabularies": boundary_vocabularies,
        "category_encoding": "zero-based index in stored vocabulary order",
        "unknown_policy": {"generator": "error", "boundary": "code -1; treat as missing"},
        "transform_policy": {
            "absolute_skew_threshold": SKEW_THRESHOLD,
            "max_quantiles": MAX_QUANTILES,
            "quantile_subsample": development_normal_df.height,
            "zero_indicators": "candidates recorded only; not applied in this baseline",
            "empirical_bounds": "observed normal support; not hard physical constraints",
        },
        "versions": {
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    metadata_path = preprocessing_output / "metadata.json"
    transformers_path = preprocessing_output / "numeric_transformers.joblib"
    metadata_path.write_text(
        json.dumps(preprocessing_metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    joblib.dump(numeric_transformers, transformers_path)
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == preprocessing_metadata
    loaded_transformers = joblib.load(transformers_path)
    assert set(loaded_transformers) == set(numeric_transformers)
    for c, transformer in loaded_transformers.items():
        sample = development_normal_df[c].head(10).to_numpy().reshape(-1, 1)
        assert np.array_equal(
            transformer.transform(sample), numeric_transformers[c].transform(sample)
        )
    print(f"Saved and verified preprocessing artifacts: {preprocessing_output}")
    input_rules = json.loads(metadata_path.read_text(encoding="utf-8"))
    input_transformers = joblib.load(transformers_path)
    assert input_rules["source_sha256"] == hashlib.sha256(train_path.read_bytes()).hexdigest()
    assert (
        input_rules["split_manifest_sha256"]
        == hashlib.sha256((split_output / "row_manifest.parquet").read_bytes()).hexdigest()
    )
    assert input_rules["feature_order"] == features
    assert input_rules["fit_partition"] == "development_normal"
    assert input_rules["fit_rows"] == development_normal_df.height
    model_numeric_columns = input_rules["active_numerical_features"]
    model_discrete_columns = [
        c
        for c in features
        if c in input_rules["active_categorical_features"]
        or c in input_rules["active_discrete_features"]
    ]
    model_constant_columns = input_rules["constant_features"]
    assert set(model_numeric_columns).isdisjoint(model_discrete_columns)
    assert set(model_numeric_columns).isdisjoint(model_constant_columns)
    assert set(model_discrete_columns).isdisjoint(model_constant_columns)
    assert set([*model_numeric_columns, *model_discrete_columns, *model_constant_columns]) == set(
        features
    )
    assert set(input_transformers) == set(model_numeric_columns)
    generator_row_manifest = (
        split_manifest.filter((pl.col("partition") == "development") & (pl.col(LABEL) == 0))
        .sort("source_row")
        .select("source_row", "feature_group")
    )
    source_rows = generator_row_manifest["source_row"].to_numpy().astype(np.int64)
    assert train_df[source_rows.tolist()].equals(development_normal_df)
    numerical_matrix = np.ascontiguousarray(
        np.column_stack(
            [
                input_transformers[c]
                .transform(development_normal_df[c].to_numpy().astype(np.float64).reshape(-1, 1))
                .ravel()
                for c in model_numeric_columns
            ]
        ),
        dtype=np.float64,
    )
    gmm_X = numerical_matrix
    tabddpm_X_num = np.ascontiguousarray(numerical_matrix, dtype=np.float32)
    gmm_donors = development_normal_df.select(model_discrete_columns)
    model_discrete_vocabularies = {
        c: input_rules["generator_vocabularies"][c]
        if c in input_rules["generator_vocabularies"]
        else sorted(development_normal_df[c].unique().to_list())
        for c in model_discrete_columns
    }
    encoded_discrete = encode_categories(development_normal_df, model_discrete_vocabularies)
    tabddpm_X_cat = np.ascontiguousarray(encoded_discrete.to_numpy(), dtype=np.int64)
    tabddpm_cardinalities = np.array(
        [len(model_discrete_vocabularies[c]) for c in model_discrete_columns], dtype=np.int64
    )
    model_input_summary = pl.DataFrame(
        [
            {
                "input": "GMM numerical",
                "rows": gmm_X.shape[0],
                "columns": gmm_X.shape[1],
                "dtype": str(gmm_X.dtype),
            },
            {
                "input": "GMM discrete donors",
                "rows": gmm_donors.height,
                "columns": gmm_donors.width,
                "dtype": "original column types",
            },
            {
                "input": "TabDDPM numerical",
                "rows": tabddpm_X_num.shape[0],
                "columns": tabddpm_X_num.shape[1],
                "dtype": str(tabddpm_X_num.dtype),
            },
            {
                "input": "TabDDPM categorical",
                "rows": tabddpm_X_cat.shape[0],
                "columns": tabddpm_X_cat.shape[1],
                "dtype": str(tabddpm_X_cat.dtype),
            },
        ]
    )
    assert (
        gmm_X.shape
        == tabddpm_X_num.shape
        == (development_normal_df.height, len(model_numeric_columns))
    )
    assert tabddpm_X_cat.shape == (development_normal_df.height, len(model_discrete_columns))
    assert gmm_donors.height == len(source_rows) == development_normal_df.height
    assert gmm_X.dtype == np.float64 and tabddpm_X_num.dtype == np.float32
    assert tabddpm_X_cat.dtype == np.int64
    assert np.isfinite(gmm_X).all() and np.isfinite(tabddpm_X_num).all()
    assert np.array_equal(tabddpm_X_num, gmm_X.astype(np.float32))
    for i, c in enumerate(model_numeric_columns):
        expected = (
            input_transformers[c]
            .transform(development_normal_df[c].to_numpy().astype(np.float64).reshape(-1, 1))
            .ravel()
        )
        assert np.array_equal(gmm_X[:, i], expected), c
    for i, c in enumerate(model_discrete_columns):
        codes = tabddpm_X_cat[:, i]
        vocabulary = model_discrete_vocabularies[c]
        assert (codes >= 0).all() and (codes < tabddpm_cardinalities[i]).all(), c
        decoded = pl.Series(
            c, [vocabulary[int(code)] for code in codes], dtype=development_normal_df.schema[c]
        )
        assert decoded.equals(development_normal_df[c]), f"Discrete round-trip mismatch: {c}"
    assert gmm_donors.equals(train_df[source_rows.tolist()].select(model_discrete_columns))
    assert LABEL not in model_numeric_columns and LABEL not in model_discrete_columns
    assert source_rows.size == np.unique(source_rows).size
    print("Passed model-input dtype, finite-value, code-range, decoding, and alignment checks.")
    model_input_output = output_dir / "model_inputs"
    model_input_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_input_output / "gmm_inputs.npz", X=gmm_X, source_row=source_rows)
    np.savez_compressed(
        model_input_output / "tabddpm_inputs.npz",
        X_num=tabddpm_X_num,
        X_cat=tabddpm_X_cat,
        categorical_cardinalities=tabddpm_cardinalities,
        source_row=source_rows,
    )
    gmm_donors.write_parquet(model_input_output / "gmm_donors.parquet")
    generator_row_manifest.write_parquet(model_input_output / "generator_rows.parquet")
    source_row_series = pl.Series("source_row", source_rows)
    csv_model_frames = {
        "gmm_inputs": pl.DataFrame(gmm_X, schema=model_numeric_columns, orient="row").insert_column(
            0, source_row_series
        ),
        "tabddpm_numerical": pl.DataFrame(
            tabddpm_X_num, schema=model_numeric_columns, orient="row"
        ).insert_column(0, source_row_series),
        "tabddpm_categorical": pl.DataFrame(
            tabddpm_X_cat, schema=model_discrete_columns, orient="row"
        ).insert_column(0, source_row_series),
        "gmm_donors": gmm_donors.clone().insert_column(0, source_row_series),
        "generator_rows": generator_row_manifest,
    }
    for name, frame in csv_model_frames.items():
        frame.write_csv(model_input_output / f"{name}.csv")
    model_input_metadata = {
        "dataset": config.name,
        "artifact_version": 1,
        "fit_partition": "development_normal",
        "rows": development_normal_df.height,
        "source_sha256": input_rules["source_sha256"],
        "split_manifest_sha256": input_rules["split_manifest_sha256"],
        "preprocessing_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "numeric_transformers_sha256": hashlib.sha256(transformers_path.read_bytes()).hexdigest(),
        "preprocessing_metadata": "../preprocessing/metadata.json",
        "numeric_transformers": "../preprocessing/numeric_transformers.joblib",
        "feature_order": features,
        "numerical_columns": model_numeric_columns,
        "discrete_columns": model_discrete_columns,
        "discrete_vocabularies": model_discrete_vocabularies,
        "categorical_cardinalities": tabddpm_cardinalities.tolist(),
        "constant_values": {
            c: input_rules["columns"][c]["constant_value"] for c in model_constant_columns
        },
        "zero_indicator_policy": "baseline; not applied",
        "gmm": {
            "file": "gmm_inputs.npz",
            "training_key": "X",
            "shape": list(gmm_X.shape),
            "dtype": str(gmm_X.dtype),
            "donors": "gmm_donors.parquet",
            "donor_sampling": "whole discrete tuples conditional on fitted component; not yet fitted",
        },
        "tabddpm": {
            "file": "tabddpm_inputs.npz",
            "numerical_key": "X_num",
            "categorical_key": "X_cat",
            "numerical_shape": list(tabddpm_X_num.shape),
            "categorical_shape": list(tabddpm_X_cat.shape),
            "numerical_dtype": str(tabddpm_X_num.dtype),
            "categorical_dtype": str(tabddpm_X_cat.dtype),
            "unknown_categories": "error; no unknown category trained",
        },
        "provenance": {
            "manifest": "generator_rows.parquet",
            "npz_key": "source_row",
            "row_order": "ascending original CSV source row",
            "model_feature": False,
        },
        "versions": input_rules["versions"],
        "csv_files": {name: f"{name}.csv" for name in csv_model_frames},
        "csv_source_row_policy": "provenance only; exclude source_row from training features",
    }
    input_metadata_path = model_input_output / "metadata.json"
    input_metadata_path.write_text(
        json.dumps(model_input_metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    assert json.loads(input_metadata_path.read_text(encoding="utf-8")) == model_input_metadata
    with np.load(model_input_output / "gmm_inputs.npz", allow_pickle=False) as stored:
        assert np.array_equal(stored["X"], gmm_X)
        assert np.array_equal(stored["source_row"], source_rows)
    with np.load(model_input_output / "tabddpm_inputs.npz", allow_pickle=False) as stored:
        assert np.array_equal(stored["X_num"], tabddpm_X_num)
        assert np.array_equal(stored["X_cat"], tabddpm_X_cat)
        assert np.array_equal(stored["categorical_cardinalities"], tabddpm_cardinalities)
        assert np.array_equal(stored["source_row"], source_rows)
    assert pl.read_parquet(model_input_output / "gmm_donors.parquet").equals(gmm_donors)
    assert pl.read_parquet(model_input_output / "generator_rows.parquet").equals(
        generator_row_manifest
    )
    print(f"Saved and verified model inputs: {model_input_output}")
    decoder = TabularDecoder.from_directory(model_input_output)
    decoded_gmm = decoder.decode_gmm(gmm_X, gmm_donors)
    decoded_tabddpm = decoder.decode_tabddpm(tabddpm_X_num, tabddpm_X_cat)
    assert decoded_gmm.schema == decoded_tabddpm.schema == X_generator_raw.schema
    assert decoded_gmm.columns == features
    for c in features:
        if c in CATEGORICAL or c in model_discrete_columns or c in model_constant_columns:
            assert decoded_gmm[c].equals(X_generator_raw[c]), c
            assert decoded_tabddpm[c].equals(X_generator_raw[c]), c
        else:
            original = X_generator_raw[c].to_numpy()
            assert np.allclose(decoded_gmm[c].to_numpy(), original, rtol=1e-07, atol=1e-07), c
            tolerance = float32_inverse_error_bound(
                input_transformers[c], original, column_rules[c]["integer_required"]
            )
            assert (np.abs(decoded_tabddpm[c].to_numpy() - original) <= tolerance).all(), c

    print("Both reusable decoders passed reconstruction and schema checks.")
    saved_development = pl.read_parquet(split_output / "development.parquet")
    saved_validation = pl.read_parquet(split_output / "validation.parquet")
    saved_normal = pl.read_parquet(split_output / "development_normal.parquet")
    saved_attack = pl.read_parquet(split_output / "development_attack.parquet")
    saved_manifest = pl.read_parquet(split_output / "row_manifest.parquet")
    saved_preprocessing = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_input_metadata = json.loads(input_metadata_path.read_text(encoding="utf-8"))
    saved_decoder = TabularDecoder.from_directory(model_input_output)
    saved_donors = pl.read_parquet(model_input_output / "gmm_donors.parquet")
    saved_generator_rows = pl.read_parquet(model_input_output / "generator_rows.parquet")
    with np.load(model_input_output / "gmm_inputs.npz", allow_pickle=False) as arrays:
        saved_gmm = arrays["X"].copy()
        saved_gmm_rows = arrays["source_row"].copy()
    with np.load(model_input_output / "tabddpm_inputs.npz", allow_pickle=False) as arrays:
        saved_num = arrays["X_num"].copy()
        saved_cat = arrays["X_cat"].copy()
        saved_cardinalities = arrays["categorical_cardinalities"].copy()
        saved_tab_rows = arrays["source_row"].copy()
    artifact_checks = []

    def record_check(name, passed):
        artifact_checks.append({"check": name, "passed": bool(passed)})

    record_check(
        "source_checksum",
        saved_preprocessing["source_sha256"] == hashlib.sha256(train_path.read_bytes()).hexdigest(),
    )
    record_check(
        "split_checksum",
        saved_preprocessing["split_manifest_sha256"]
        == hashlib.sha256((split_output / "row_manifest.parquet").read_bytes()).hexdigest(),
    )
    record_check(
        "manifest_coverage",
        saved_manifest["source_row"].sort().to_list() == list(range(train_df.height)),
    )
    record_check(
        "manifest_partitions",
        set(saved_manifest["partition"].to_list()) == {"development", "validation"},
    )
    for partition, frame in [("development", saved_development), ("validation", saved_validation)]:
        row_ids = (
            saved_manifest.filter(pl.col("partition") == partition)
            .sort("source_row")["source_row"]
            .to_list()
        )
        record_check(f"{partition}_source_alignment", frame.equals(train_df[row_ids]))
        record_check(f"{partition}_both_classes", set(frame[LABEL].to_list()) == {0, 1})
    record_check(
        "duplicate_groups_kept_together",
        saved_manifest.group_by("feature_group")
        .agg(pl.col("partition").n_unique())["partition"]
        .max()
        == 1,
    )
    record_check(
        "no_feature_overlap",
        saved_development.select(features)
        .unique()
        .join(saved_validation.select(features).unique(), on=features, how="inner")
        .height
        == 0,
    )
    record_check("normal_subset", saved_normal.equals(saved_development.filter(pl.col(LABEL) == 0)))
    record_check("attack_subset", saved_attack.equals(saved_development.filter(pl.col(LABEL) == 1)))
    record_check(
        "normal_fit_provenance",
        saved_preprocessing["fit_partition"] == "development_normal"
        and saved_preprocessing["fit_rows"] == saved_normal.height,
    )
    record_check(
        "schema_and_feature_order",
        saved_normal.schema == train_df.schema
        and saved_input_metadata["feature_order"] == features,
    )
    record_check(
        "model_row_ids",
        np.array_equal(saved_gmm_rows, saved_tab_rows)
        and np.array_equal(saved_gmm_rows, saved_generator_rows["source_row"].to_numpy()),
    )
    record_check("model_source_alignment", saved_normal.equals(train_df[saved_gmm_rows.tolist()]))
    record_check(
        "gmm_donors_aligned",
        saved_donors.equals(saved_normal.select(saved_input_metadata["discrete_columns"])),
    )
    record_check(
        "array_shapes",
        saved_gmm.shape
        == saved_num.shape
        == (saved_normal.height, len(saved_input_metadata["numerical_columns"]))
        and saved_cat.shape == (saved_normal.height, len(saved_input_metadata["discrete_columns"])),
    )
    record_check(
        "array_dtypes",
        saved_gmm.dtype == np.float64
        and saved_num.dtype == np.float32
        and (saved_cat.dtype == np.int64),
    )
    record_check(
        "finite_model_inputs", np.isfinite(saved_gmm).all() and np.isfinite(saved_num).all()
    )
    record_check(
        "shared_numeric_representation", np.array_equal(saved_num, saved_gmm.astype(np.float32))
    )
    record_check(
        "cardinalities",
        saved_cardinalities.tolist() == saved_input_metadata["categorical_cardinalities"],
    )
    for i, c in enumerate(saved_input_metadata["numerical_columns"]):
        expected = (
            saved_decoder.transformers[c]
            .transform(saved_normal[c].to_numpy().astype(np.float64).reshape(-1, 1))
            .ravel()
        )
        record_check(f"saved_transform:{c}", np.array_equal(saved_gmm[:, i], expected))
    for i, c in enumerate(saved_input_metadata["discrete_columns"]):
        vocabulary = saved_input_metadata["discrete_vocabularies"][c]
        record_check(
            f"normal_vocabulary:{c}", vocabulary == sorted(saved_normal[c].unique().to_list())
        )
        record_check(
            f"valid_codes:{c}",
            (saved_cat[:, i] >= 0).all() and (saved_cat[:, i] < len(vocabulary)).all(),
        )
    reconstruction_rows = []
    for engine, decoded in [
        ("gmm", saved_decoder.decode_gmm(saved_gmm, saved_donors)),
        ("tabddpm", saved_decoder.decode_tabddpm(saved_num, saved_cat)),
    ]:
        record_check(
            f"decoded_schema:{engine}", decoded.schema == saved_normal.select(features).schema
        )
        for c in features:
            if c in numerical:
                original = saved_normal[c].to_numpy().astype(np.float64)
                restored = decoded[c].to_numpy().astype(np.float64)
                error = np.abs(restored - original)
                integer = saved_preprocessing["columns"][c]["integer_required"]
                atol = (2.0 if integer else 1e-05) if engine == "tabddpm" else 1e-07
                rtol = 0.0 if engine == "tabddpm" else 1e-7
                precision_bound = (
                    float32_inverse_error_bound(saved_decoder.transformers[c], original, integer)
                    if engine == "tabddpm" and c in saved_decoder.transformers
                    else np.full(len(original), atol)
                )
                atol = float(precision_bound.max())
                reconstruction_rows.append(
                    {
                        "engine": engine,
                        "feature": c,
                        "max_absolute_error": float(error.max()),
                        "mean_absolute_error": float(error.mean()),
                        "atol": atol,
                        "rtol": rtol,
                    }
                )
                record_check(
                    f"reconstruction:{engine}:{c}",
                    (
                        np.abs(restored - original) <= precision_bound + rtol * np.abs(original)
                    ).all(),
                )
            else:
                record_check(f"reconstruction:{engine}:{c}", decoded[c].equals(saved_normal[c]))
    for folder, frames in [
        (split_output, {**raw_partitions, "row_manifest": saved_manifest}),
        (model_input_output, csv_model_frames),
    ]:
        for name, frame in frames.items():
            loaded_csv = pl.read_csv(folder / f"{name}.csv", schema_overrides=frame.schema)
            record_check(f"csv_roundtrip:{folder.name}:{name}", loaded_csv.equals(frame))
    artifact_check_table = pl.DataFrame(artifact_checks)
    reconstruction_report = pl.DataFrame(reconstruction_rows)
    audit_output = output_dir / "audit"
    audit_output.mkdir(parents=True, exist_ok=True)
    audit_tables = {
        "feature_catalogue": feature_catalogue,
        "data_quality": quality_report,
        "numerical_distributions": numerical_summary,
        "category_frequencies": category_frequencies,
        "duplicate_summary": pl.DataFrame([duplicate_summary]),
        "split_summary": split_summary,
        "unknown_categories": unknown_category_report,
        "preprocessing_rules": preprocessing_summary,
        "model_input_summary": model_input_summary,
        "artifact_checks": artifact_check_table,
        "reconstruction_errors": reconstruction_report,
    }
    for name, table in audit_tables.items():
        table.write_csv(audit_output / f"{name}.csv")
        table.write_parquet(audit_output / f"{name}.parquet")
    artifact_inventory = []
    for folder in [split_output, preprocessing_output, model_input_output]:
        for artifact in sorted(folder.iterdir()):
            if artifact.is_file():
                artifact_inventory.append(
                    {
                        "path": artifact.relative_to(output_dir).as_posix(),
                        "bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                )
    pl.DataFrame(artifact_inventory).write_csv(audit_output / "artifact_inventory.csv")
    audit_report = {
        "dataset": config.name,
        "artifact_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(row["passed"] for row in artifact_checks) else "failed",
        "source_sha256": saved_preprocessing["source_sha256"],
        "split_seed": SPLIT_SEED,
        "normal_generator_rows": saved_normal.height,
        "checks": artifact_checks,
        "tables": {
            name: {"csv": f"{name}.csv", "parquet": f"{name}.parquet"} for name in audit_tables
        },
        "artifacts": artifact_inventory,
        "notes": [
            "Exploratory tables use the full released data; generator fitting uses development normals only.",
            "Float32 inverse errors are bounded per row using adjacent representable coordinates; reconstruction_errors.csv records measured errors and maximum bounds.",
            "Zero-indicator modelling is not applied in the baseline.",
            "These checks do not establish generation quality, unseen-attack performance, or downstream AUPRC.",
        ],
    }
    (audit_output / "audit_report.json").write_text(
        json.dumps(audit_report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    failed_checks = artifact_check_table.filter(~pl.col("passed"))
    if failed_checks.height:
        pass
        raise ValueError(f"{failed_checks.height} artifact checks failed; see {audit_output}")
    print(f"Passed {len(artifact_checks)} artifact checks. Audit package: {audit_output}")
    return {
        "category_support": category_support,
        "split_category_support": split_category_support,
        "validation_attack_df": validation_attack_df,
        "y_boundary": y_boundary,
        "y_validation": y_validation,
        "config": config,
        "train_df": train_df,
        "features": features,
        "development_df": development_df,
        "validation_df": validation_df,
        "development_normal_df": development_normal_df,
        "development_attack_df": development_attack_df,
        "split_manifest": split_manifest,
        "split_metadata": split_metadata,
        "preprocessing_metadata": preprocessing_metadata,
        "model_input_metadata": model_input_metadata,
        "gmm_X": gmm_X,
        "gmm_donors": gmm_donors,
        "tabddpm_X_num": tabddpm_X_num,
        "tabddpm_X_cat": tabddpm_X_cat,
        "decoder": saved_decoder,
        "decoded_gmm": decoded_gmm,
        "decoded_tabddpm": decoded_tabddpm,
        "audit_tables": audit_tables,
        "audit_report": audit_report,
        "output_dir": output_dir,
        "model_input_output": model_input_output,
        "audit_output": audit_output,
    }


def plot_distributions(result):
    """Plot configured features without transforming the source data."""
    import matplotlib.pyplot as plt

    config = result["config"]
    data = result["train_df"]
    normal = data.filter(pl.col(config.label) == 0)
    attack = data.filter(pl.col(config.label) == 1)
    columns = config.plot_features
    if not columns:
        raise ValueError("No plot features configured")
    rows = (len(columns) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), squeeze=False)
    for ax, column in zip(axes.flat, columns):
        values = [frame[column].to_numpy() for frame in (normal, attack)]
        use_log = column in config.log_plot_features
        if use_log:
            values = [np.log1p(v) for v in values]
        bins = np.histogram_bin_edges(np.concatenate(values), bins=40)
        for label, values_for_class, color in zip(
            ["Normal", "Attack"], values, ["tab:blue", "tab:orange"]
        ):
            ax.hist(
                values_for_class, bins=bins, density=True, histtype="step", label=label, color=color
            )
        ax.set_xlabel(f"log1p({column})" if use_log else column)
        ax.set_ylabel("Density")
        ax.legend()
    for ax in list(axes.flat)[len(columns) :]:
        ax.set_visible(False)
    fig.suptitle(f"{config.name}: numerical distributions by class")
    fig.tight_layout()
    return fig


def main():
    """Run from the command line, resolving default paths against the repository."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["nsl_kdd", "unsw_nb15"],
        help="Run only this dataset (default: run both)",
    )
    parser.add_argument("--input", type=Path, help="Override the default training CSV")
    parser.add_argument("--output-root", type=Path, help="Override the repository output folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if args.input is not None and args.dataset is None:
        parser.error("--input requires --dataset to identify the CSV schema")
    root = Path(__file__).resolve().parent.parent
    datasets = {
        "nsl_kdd": (NSL_KDD, "NSL-KDD"),
        "unsw_nb15": (UNSW_NB15, "UNSW-NB15"),
    }
    selected = [args.dataset] if args.dataset is not None else list(datasets)
    for name in selected:
        config, folder = datasets[name]
        print(f"Running preprocessing for {folder}")
        run_preprocessing(
            args.input if args.input is not None else root / "data" / folder / "train.csv",
            config,
            output_root=args.output_root if args.output_root is not None else root / "output",
            seed=args.seed,
            validation_fraction=args.validation_fraction,
        )


if __name__ == "__main__":
    main()
