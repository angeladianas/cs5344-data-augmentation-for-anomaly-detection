"""LightGBM-based Anomaly Boundary Rejection Filter (Stage 3).

Prunes synthetic normal candidate records that leak into anomaly decision
regions before downstream detector evaluation.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


def detect_feature_types(
    df: pd.DataFrame | pl.DataFrame, target_col: str = "is_anomaly"
) -> tuple[list[str], list[str], list[str]]:
    r"""Detect feature names, categorical columns, and numerical columns.

    Compatible with both Pandas and Polars DataFrames across all versions.

    Args:
        df: Input DataFrame containing features and optional label/id.
        target_col: Name of the binary target label column to exclude.

    Returns:
        tuple of (feature_cols, cat_cols, num_cols).
    """
    if isinstance(df, pl.DataFrame):
        feature_cols = [c for c in df.columns if c != target_col and c != "id"]
        cat_cols = []
        num_cols = []
        for col in feature_cols:
            dtype = df.schema[col]
            if dtype in (pl.String, pl.Categorical):
                cat_cols.append(col)
            else:
                num_cols.append(col)
        return feature_cols, cat_cols, num_cols

    # Pandas path
    feature_cols = [c for c in df.columns if c != target_col and c != "id"]
    cat_cols = []
    num_cols = []

    for col in feature_cols:
        dtype = df[col].dtype
        if (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
        ):
            cat_cols.append(col)
        else:
            num_cols.append(col)

    return feature_cols, cat_cols, num_cols


class LightGBMBoundaryFilter:
    r"""LightGBM-based Anomaly Boundary Estimator for Synthetic Candidate Filtering.

    Trained on labeled development data (both normal and known attack samples)
    to estimate P(is_anomaly = 1 | x). Prunes synthetic candidates that drift
    into anomalous decision spaces to prevent normal envelope dilation.

    Attributes:
        cat_cols: List of categorical feature names.
        params: LightGBM classifier hyperparameters.
        random_state: Random state for reproducibility.
        model: Fitted LGBMClassifier instance.
        feature_names: List of feature names in training order.
        known_categories: Dict mapping categorical column to list of known categories.
    """

    def __init__(
        self,
        cat_cols: list[str] | None = None,
        params: dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> None:
        self.cat_cols: list[str] = list(cat_cols) if cat_cols else []
        self.random_state: int = random_state
        self.default_params: dict[str, Any] = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "class_weight": "balanced",
            "random_state": self.random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }
        if params:
            self.default_params.update(params)
        self.model: lgb.LGBMClassifier = lgb.LGBMClassifier(**self.default_params)
        self.feature_names: list[str] = []
        self.known_categories: dict[str, list[str]] = {}

    def _to_pandas(self, df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        """Ensure input is converted to a clean Pandas DataFrame (resilient to missing pyarrow)."""
        if isinstance(df, pl.DataFrame):
            try:
                return df.to_pandas()
            except ImportError:
                return pd.DataFrame(df.to_dict(as_series=False))
        return df.copy()

    def _prepare_features(self, df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        """Standardize feature matrix, dropping metadata and setting categorical dtypes."""
        pdf = self._to_pandas(df)
        drop_cols = [c for c in ("id", "is_anomaly", "source_row") if c in pdf.columns]
        if drop_cols:
            pdf = pdf.drop(columns=drop_cols)

        known_cats = getattr(self, "known_categories", {})
        for col in self.cat_cols:
            if col in pdf.columns:
                if col in known_cats:
                    cat_type = pd.CategoricalDtype(categories=known_cats[col], ordered=False)
                    pdf[col] = pdf[col].astype(str).astype(cat_type)
                else:
                    pdf[col] = pdf[col].astype(str).astype("category")
        return pdf

    def fit(
        self,
        train_df: pd.DataFrame | pl.DataFrame | None = None,
        target_col: str = "is_anomaly",
        *,
        val_df: pd.DataFrame | pl.DataFrame | None = None,
    ) -> LightGBMBoundaryFilter:
        r"""Fit the boundary estimator on labeled development training data.

        WARNING: Pass only development training data (never held-out validation data)
        to prevent validation leakage.

        Args:
            train_df: Labeled development DataFrame containing features and target_col.
            target_col: Name of the binary target label column.
            val_df: Deprecated alias for train_df (supported for backwards compatibility).

        Returns:
            self (fitted filter).
        """
        source_df = train_df if train_df is not None else val_df
        if source_df is None:
            raise ValueError("train_df must be provided to fit the boundary filter")

        pdf = self._to_pandas(source_df)
        if target_col not in pdf.columns:
            raise ValueError(f"Target column '{target_col}' not found in training DataFrame")

        y = pdf[target_col].to_numpy()
        unique_labels = set(np.unique(y))
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Expected binary labels {{0, 1}}, got {unique_labels}")

        self.feature_names, detected_cats, _ = detect_feature_types(pdf, target_col)
        if not self.cat_cols:
            self.cat_cols = detected_cats

        # Record known categories for robust inference
        for col in self.cat_cols:
            if col in pdf.columns:
                self.known_categories[col] = sorted(pdf[col].dropna().astype(str).unique().tolist())

        X = self._prepare_features(pdf)
        X = X[self.feature_names]

        self.model.fit(X, y)
        return self

    def predict_anomaly_prob(self, df_candidates: pd.DataFrame | pl.DataFrame) -> np.ndarray:
        r"""Predict anomaly probability P(is_anomaly = 1 | x) for candidate points.

        Args:
            df_candidates: Candidate normal records (Pandas or Polars DataFrame).

        Returns:
            np.ndarray of predicted attack probabilities in [0.0, 1.0].
        """
        if not self.feature_names:
            raise ValueError("Boundary filter has not been fitted yet. Call fit() first.")

        X = self._prepare_features(df_candidates)
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Candidates missing required features: {missing_features}")

        X = X[self.feature_names]
        probs = self.model.predict_proba(X)[:, 1]
        return np.asarray(probs, dtype=np.float64)

    def filter_candidates(
        self,
        df_candidates: pd.DataFrame | pl.DataFrame,
        tau_reject: float = 0.10,
        min_survivors: int | None = None,
    ) -> tuple[pd.DataFrame | pl.DataFrame, float, np.ndarray]:
        r"""Prune candidates whose predicted anomaly probability exceeds tau_reject.

        Includes dynamic fallback: if min_survivors is specified and the threshold
        is too aggressive (yielding fewer than min_survivors), it automatically
        relaxes tau_reject to guarantee the minimum required candidate volume.

        Args:
            df_candidates: Synthetic candidate records (Pandas or Polars DataFrame).
            tau_reject: Rejection threshold. Discard x if P(anomaly | x) > tau_reject.
            min_survivors: Minimum surviving candidate count required (e.g., target N).

        Returns:
            tuple of (filtered_df, survival_rate, anomaly_probabilities).
            The returned filtered_df matches the input DataFrame type (Pandas or Polars).
        """
        is_polars = isinstance(df_candidates, pl.DataFrame)
        n_total = len(df_candidates)
        if n_total == 0:
            return df_candidates, 0.0, np.empty(0, dtype=np.float64)

        probs = self.predict_anomaly_prob(df_candidates)
        keep_mask = probs <= tau_reject

        # Dynamic fallback safeguard against candidate depletion
        if min_survivors is not None and keep_mask.sum() < min_survivors:
            target_count = min(min_survivors, n_total)
            logger.warning(
                "Filter threshold tau_reject=%.3f yielded %d candidates, below required %d. "
                "Dynamically relaxing threshold.",
                tau_reject,
                keep_mask.sum(),
                target_count,
            )
            # Find the target_count-th lowest probability to retain enough candidates
            sorted_indices = np.argsort(probs)
            keep_indices = sorted_indices[:target_count]
            keep_mask = np.zeros(n_total, dtype=bool)
            keep_mask[keep_indices] = True

        survival_rate = float(keep_mask.sum()) / max(n_total, 1)

        if is_polars:
            filtered_df = df_candidates.filter(pl.Series(keep_mask))
        else:
            filtered_df = df_candidates[keep_mask].copy()

        return filtered_df, survival_rate, probs

    def save(self, filepath: str | Path) -> None:
        """Save the trained filter to disk via pickle."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("BoundaryFilter saved to %s", path)

    @staticmethod
    def load(filepath: str | Path) -> LightGBMBoundaryFilter:
        """Load a pre-trained BoundaryFilter from disk."""
        path = Path(filepath)
        with open(path, "rb") as f:
            obj: LightGBMBoundaryFilter = pickle.load(f)
        logger.info("BoundaryFilter loaded from %s", path)
        return obj
