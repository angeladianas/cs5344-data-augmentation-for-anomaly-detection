"""Shared-pipeline checks using small data independent of released CSVs."""

import numpy as np
import polars as pl
import pytest

from utils.dataset_configs import NSL_KDD, UNSW_NB15, DatasetConfig
from utils.preprocessing import make_development_split, run_preprocessing


def test_configs_cover_different_semantics():
    assert len(NSL_KDD.features) == len(set(NSL_KDD.features)) == 41
    assert len(UNSW_NB15.features) == len(set(UNSW_NB15.features)) == 42
    assert NSL_KDD.semantic_type("serror_rate") == "bounded_rate"
    assert UNSW_NB15.semantic_type("rate") == "continuous"
    assert UNSW_NB15.semantic_type("is_ftp_login") == "discrete_state"


def test_duplicate_groups_and_conflicting_labels():
    frame = pl.DataFrame(
        {
            "feature": [i for i in range(20) for _ in range(3)],
            "label": [int(i >= 10) for i in range(20) for _ in range(3)],
        }
    )
    development, validation, manifest = make_development_split(frame, ["feature"], "label", 0.2, 42)
    assert development.height == 48 and validation.height == 12
    assert development.join(validation, on="feature", how="inner").height == 0
    _, _, repeated = make_development_split(frame, ["feature"], "label", 0.2, 42)
    assert manifest.equals(repeated)
    conflict = pl.concat([frame, pl.DataFrame({"feature": [0], "label": [1]})])
    with pytest.raises(ValueError, match="conflicting labels"):
        make_development_split(conflict, ["feature"], "label", 0.2, 42)


def test_pipeline_supports_a_custom_schema_and_preserves_tokens(tmp_path):
    config = DatasetConfig(
        name="fixture",
        categorical=("category",),
        counts=("count", "constant"),
        continuous=("volume",),
        binary=("binary",),
    )
    frame = pl.DataFrame(
        {
            "count": list(range(100)),
            "constant": [0] * 100,
            "volume": [float(i) / 10 for i in range(70)] + [1e9] * 30,
            "binary": [i % 2 for i in range(100)],
            "category": ["-" if i % 2 else "normal" for i in range(70)] + ["attack_only"] * 30,
            "is_anomaly": [0] * 70 + [1] * 30,
        }
    )
    source = tmp_path / "train.csv"
    frame.write_csv(source)
    result = run_preprocessing(source, config, tmp_path / "output", max_quantiles=20)
    metadata = result["preprocessing_metadata"]
    assert metadata["columns"]["volume"]["empirical_normal_bounds"]["max"] < 1e9
    assert metadata["generator_vocabularies"]["category"] == ["-", "normal"]
    assert "attack_only" in metadata["boundary_vocabularies"]["category"]
    assert result["audit_report"]["status"] == "passed"
    assert result["gmm_X"].shape[0] == 56
    assert result["tabddpm_X_cat"].dtype == np.int64
    assert result["decoded_gmm"].schema == frame.drop("is_anomaly").schema
    assert (tmp_path / "output" / "fixture" / "audit" / "audit_report.json").is_file()


def test_wrong_dataset_schema_fails_before_export(tmp_path):
    source = tmp_path / "train.csv"
    pl.DataFrame({"unexpected": [1, 2], "is_anomaly": [0, 1]}).write_csv(source)
    with pytest.raises(ValueError, match="configured schema"):
        run_preprocessing(source, UNSW_NB15, tmp_path / "output")
    assert not (tmp_path / "output").exists()
