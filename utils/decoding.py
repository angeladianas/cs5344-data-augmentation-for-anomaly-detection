"""Decode generator arrays using the metadata saved by preprocessing.

Example::

    decoder = TabularDecoder.from_directory("output/nsl_kdd/model_inputs")
    rows = decoder.decode_tabddpm(X_num, X_cat)
    rows = decoder.decode_gmm(X, discrete_donors)

GMM callers supply one raw discrete donor tuple per generated numerical row,
selected using their fitted component assignments. Outputs contain feature
columns only; labels, submission IDs, and boundary filtering belong to callers.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl


class TabularDecoder:
    """Restore scales, categories, constants, feature order, and original dtypes."""

    def __init__(self, preprocessing, model_inputs, transformers):
        self.preprocessing = preprocessing
        self.model_inputs = model_inputs
        self.transformers = transformers
        self.features = preprocessing["feature_order"]
        self.numerical = model_inputs["numerical_columns"]
        self.discrete = model_inputs["discrete_columns"]
        self.constants = model_inputs["constant_values"]
        self.vocabularies = model_inputs["discrete_vocabularies"]
        if self.features != model_inputs["feature_order"]:
            raise ValueError("Preprocessing and model feature orders differ")
        combined = [*self.numerical, *self.discrete, *self.constants]
        if len(combined) != len(set(combined)) or set(combined) != set(self.features):
            raise ValueError("Model columns must partition the original features")
        if set(transformers) != set(self.numerical):
            raise ValueError("Numerical transformer columns do not match model inputs")

    @classmethod
    def from_directory(cls, directory):
        """Load trusted local artifacts, checking linked preprocessing checksums.

        Joblib deserialization executes Python code: use artifacts you generated
        or otherwise trust. Paths are resolved relative to model metadata.
        """
        directory = Path(directory)
        model = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        paths = {
            "preprocessing_metadata": directory / model["preprocessing_metadata"],
            "numeric_transformers": directory / model["numeric_transformers"],
        }
        for key, path in paths.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != model[f"{key}_sha256"]:
                raise ValueError(f"Artifact checksum mismatch: {key}")
        preprocessing = json.loads(paths["preprocessing_metadata"].read_text(encoding="utf-8"))
        return cls(preprocessing, model, joblib.load(paths["numeric_transformers"]))

    @staticmethod
    def _matrix(values, width, name):
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError(f"{name} must be a two-dimensional array with {width} columns")
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite numerical values")
        return values

    def _decode_numerical(self, values, clip_to_normal_bounds):
        decoded = {}
        for i, column in enumerate(self.numerical):
            rule = self.preprocessing["columns"][column]
            if len(values):
                restored = (
                    self.transformers[column]
                    .inverse_transform(values[:, i].astype(np.float64).reshape(-1, 1))
                    .ravel()
                )
            else:
                restored = np.empty(0, dtype=np.float64)
            if not np.isfinite(restored).all():
                raise ValueError(f"Inverse transform produced nonfinite values: {column}")
            domain = rule["domain"]
            lower, upper = domain.get("min", -np.inf), domain.get("max", np.inf)
            if clip_to_normal_bounds:
                bounds = rule["empirical_normal_bounds"]
                lower, upper = max(lower, bounds["min"]), min(upper, bounds["max"])
            if rule["integer_required"]:
                restored = np.rint(restored)
            decoded[column] = np.clip(restored, lower, upper)
        return decoded

    def _finish(self, numeric, discrete, rows):
        columns = {**numeric, **discrete}
        columns.update({column: [value] * rows for column, value in self.constants.items()})
        output = []
        for column in self.features:
            rule = self.preprocessing["columns"][column]
            dtype = getattr(pl, rule["dtype"], None)
            if dtype is None:
                raise ValueError(f"Unsupported original dtype: {rule['dtype']}")
            series = pl.Series(column, columns[column]).cast(dtype, strict=True)
            if series.null_count():
                raise ValueError(f"Decoded column contains nulls: {column}")
            domain = rule["domain"]
            if "allowed_values" in domain and not series.is_in(domain["allowed_values"]).all():
                raise ValueError(f"Decoded values violate discrete support: {column}")
            output.append(series)
        return pl.DataFrame(output)

    def decode_tabddpm(self, X_num, X_cat, *, clip_to_normal_bounds=True):
        """Decode arrays; reject noninteger/out-of-range category codes.

        Binary and small discrete states are decoded through the same saved
        vocabularies as string categories. No unknown code is a valid output.
        """
        numeric = self._matrix(X_num, len(self.numerical), "X_num")
        categorical = self._matrix(X_cat, len(self.discrete), "X_cat")
        if numeric.shape[0] != categorical.shape[0]:
            raise ValueError("Numerical and categorical row counts differ")
        decoded = {}
        for i, column in enumerate(self.discrete):
            codes = categorical[:, i]
            vocabulary = self.vocabularies[column]
            if ((codes != np.floor(codes)) | (codes < 0) | (codes >= len(vocabulary))).any():
                raise ValueError(f"Invalid categorical codes: {column}")
            decoded[column] = [vocabulary[int(code)] for code in codes]
        return self._finish(
            self._decode_numerical(numeric, clip_to_normal_bounds), decoded, len(numeric)
        )

    def decode_gmm(self, X, donors, *, clip_to_normal_bounds=True):
        """Combine GMM numerical samples with aligned raw discrete donor rows.

        Select complete donor tuples conditional on GMM components before
        calling. This method does not fit a sampler or pick donors independently.
        """
        numeric = self._matrix(X, len(self.numerical), "X")
        if donors.height != len(numeric):
            raise ValueError("Numerical and donor row counts differ")
        if not set(self.discrete).issubset(donors.columns):
            raise ValueError("Donor table is missing discrete columns")
        decoded = {}
        for column in self.discrete:
            series = donors[column]
            if series.null_count() or not series.is_in(self.vocabularies[column]).all():
                raise ValueError(f"Invalid discrete donor values: {column}")
            decoded[column] = series.to_list()
        return self._finish(
            self._decode_numerical(numeric, clip_to_normal_bounds), decoded, len(numeric)
        )
