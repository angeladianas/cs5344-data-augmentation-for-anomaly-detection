"""
LightGBM-based Anomaly Boundary Rejection Filter (Task 2.5).

Used in Stage 3 to prune synthetic normal candidates that leak into 
anomaly decision regions before downstream detector evaluation.
"""

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb


def detect_feature_types(df, target_col="is_anomaly"):
    """
    Universally detects categorical and numerical columns across Pandas 1.x, 2.x, and 3.x.
    """
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
    """
    LightGBM-based Anomaly Boundary Estimator for Synthetic Candidate Filtering.

    Attributes:
        cat_cols (list): List of categorical feature names.
        params (dict): LightGBM classifier hyperparameters.
    """

    def __init__(self, cat_cols=None, params=None, random_state=42):
        self.cat_cols = cat_cols or []
        self.random_state = random_state
        self.default_params = {
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
        self.model = lgb.LGBMClassifier(**self.default_params)
        self.feature_names = []
        self.cat_dtypes = {}

    def _prepare_features(self, df):
        X = df.copy()
        if "id" in X.columns:
            X = X.drop(columns=["id"])
        if "is_anomaly" in X.columns:
            X = X.drop(columns=["is_anomaly"])

        for col in self.cat_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).astype("category")
        return X

    def fit(self, val_df, target_col="is_anomaly"):
        """
        Fits the boundary estimator on the labeled validation split.
        """
        self.feature_names, self.cat_cols, _ = detect_feature_types(val_df, target_col)
        X = self._prepare_features(val_df)
        y = val_df[target_col].values

        for col in self.cat_cols:
            self.cat_dtypes[col] = X[col].dtype

        self.model.fit(X, y)
        return self

    def predict_anomaly_prob(self, df_candidates):
        """
        Predicts anomaly probability P(is_anomaly = 1 | x) for candidate points.
        """
        X = self._prepare_features(df_candidates)
        X = X[self.feature_names]
        return self.model.predict_proba(X)[:, 1]

    def filter_candidates(self, df_candidates, tau_reject=0.10):
        """
        Prunes candidates whose predicted anomaly probability exceeds tau_reject.

        Args:
            df_candidates (pd.DataFrame): Synthetic candidate records.
            tau_reject (float): Rejection threshold (default: 0.10).
                                Discard candidate x if P(anomaly | x) > tau_reject.

        Returns:
            filtered_df (pd.DataFrame): Surviving candidate records.
            survival_rate (float): Proportion of candidates that passed the filter.
            anomaly_probabilities (np.ndarray): Predicted anomaly probabilities.
        """
        probs = self.predict_anomaly_prob(df_candidates)
        keep_mask = probs <= tau_reject
        filtered_df = df_candidates[keep_mask].copy()
        survival_rate = float(keep_mask.sum()) / max(len(df_candidates), 1)
        return filtered_df, survival_rate, probs

    def save(self, filepath):
        """Saves the trained filter to a pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"BoundaryFilter saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Loads a pre-trained BoundaryFilter from a pickle file."""
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        print(f"BoundaryFilter loaded from {filepath}")
        return obj
