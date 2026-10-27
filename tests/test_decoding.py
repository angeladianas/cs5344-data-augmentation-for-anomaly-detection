"""Decoding contracts independent of the local NSL-KDD files."""

import numpy as np
import polars as pl
import pytest
from sklearn.preprocessing import StandardScaler

from utils import TabularDecoder


@pytest.fixture
def decoder() -> TabularDecoder:
    r"""Build a mock TabularDecoder with continuous, discrete, and constant features."""
    transformer = StandardScaler().fit(np.array([[0.0], [10.0]]))
    preprocessing = {
        "feature_order": ["count", "service", "binary", "constant"],
        "columns": {
            "count": {
                "dtype": "Int64",
                "integer_required": True,
                "domain": {"min": 0},
                "empirical_normal_bounds": {"min": 0, "max": 10},
            },
            "service": {"dtype": "String", "domain": {"allowed_values": ["http", "smtp"]}},
            "binary": {"dtype": "Int64", "domain": {"allowed_values": [0, 1]}},
            "constant": {"dtype": "Int64", "domain": {"min": 0}},
        },
    }
    model = {
        "feature_order": preprocessing["feature_order"],
        "numerical_columns": ["count"],
        "discrete_columns": ["service", "binary"],
        "constant_values": {"constant": 0},
        "discrete_vocabularies": {"service": ["http", "smtp"], "binary": [0, 1]},
    }
    return TabularDecoder(preprocessing, model, {"count": transformer})


def test_restore_constraints_types_and_order(decoder: TabularDecoder) -> None:
    r"""Verify restoration of domain bounds, integer casting, and schema column order."""
    result = decoder.decode_tabddpm(
        np.array([[-2.0], [0.12], [3.0]]), np.array([[0, 1], [1, 0], [0, 0]])
    )
    assert result.to_dict(as_series=False) == {
        "count": [0, 6, 10],
        "service": ["http", "smtp", "http"],
        "binary": [1, 0, 0],
        "constant": [0, 0, 0],
    }
    assert result.schema == {
        "count": pl.Int64,
        "service": pl.String,
        "binary": pl.Int64,
        "constant": pl.Int64,
    }
    unconstrained = decoder.decode_tabddpm(
        np.array([[3.0]]), np.array([[0, 0]]), clip_to_normal_bounds=False
    )
    assert unconstrained["count"][0] == 20


@pytest.mark.parametrize("codes", [[[-1, 0]], [[2, 0]], [[0.5, 0]], [[0, np.nan]]])
def test_invalid_codes_rejected(decoder: TabularDecoder, codes: list[list[float]]) -> None:
    r"""Verify that out-of-vocabulary and non-integer categorical codes raise ValueError."""
    with pytest.raises(ValueError):
        decoder.decode_tabddpm(np.array([[0.0]]), np.array(codes))


def test_invalid_shape_nonfinite_and_row_alignment(decoder: TabularDecoder) -> None:
    r"""Verify shape checks, non-finite value checks, and row alignment exceptions."""
    for values in [np.array([0.0]), np.array([[0.0, 1.0]]), np.array([[np.inf]])]:
        with pytest.raises(ValueError):
            decoder.decode_tabddpm(values, np.array([[0, 0]]))
    with pytest.raises(ValueError, match="row counts"):
        decoder.decode_tabddpm(np.zeros((2, 1)), np.zeros((1, 2)))


def test_gmm_preserves_whole_discrete_tuple(decoder: TabularDecoder) -> None:
    r"""Verify that GMM discrete decoding preserves joint donor tuples and removes provenance."""
    donors = pl.DataFrame({"service": ["smtp", "http"], "binary": [1, 0], "source_row": [8, 2]})
    output = decoder.decode_gmm(np.array([[0.0], [1.0]]), donors)
    assert output.select("service", "binary").equals(donors.select("service", "binary"))
    assert "source_row" not in output.columns
    with pytest.raises(ValueError, match="donor values"):
        decoder.decode_gmm(np.array([[0.0]]), pl.DataFrame({"service": ["unseen"], "binary": [0]}))
    with pytest.raises(ValueError, match="row counts"):
        decoder.decode_gmm(np.zeros((1, 1)), donors)


def test_empty_arrays_restore_schema(decoder: TabularDecoder) -> None:
    r"""Verify that decoding empty arrays preserves column schema and zero row height."""
    output = decoder.decode_tabddpm(np.empty((0, 1)), np.empty((0, 2), dtype=np.int64))
    assert output.shape == (0, 4)
