from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GMM_INPUTS = REPO_ROOT / "output" / "nsl_kdd" / "model_inputs" / "gmm_inputs.csv"


def load_gmm_inputs(path: str | Path = DEFAULT_GMM_INPUTS) -> pd.DataFrame:
    """Load the prepared GMM numerical input CSV used by the augmentation pipeline."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"GMM input file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def gmm_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the numerical feature columns, excluding provenance metadata."""
    exclude = {"source_row"}
    return [column for column in frame.columns if column not in exclude]


def fit_gmm_baseline(
    frame: pd.DataFrame,
    *,
    candidate_components: tuple[int, ...] = (2, 5, 8, 10, 15),
    random_state: int = 42,
):
    """Fit a GMM across candidate component counts and keep the best BIC model."""
    feature_columns = gmm_feature_columns(frame)
    if not feature_columns:
        raise ValueError("No feature columns available to fit the GMM")

    X = frame[feature_columns].to_numpy(dtype=float)
    best_model = None
    best_score = None
    best_info = None

    for n_components in candidate_components:
        if n_components >= X.shape[0]:
            continue
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=3,
            random_state=random_state,
        )
        model.fit(X)
        bic = model.bic(X)
        info = {
            "n_components": n_components,
            "bic": float(bic),
            "aic": float(model.aic(X)),
            "weights": model.weights_.copy(),
            "means": model.means_.copy(),
            "covariances": model.covariances_.copy(),
        }
        if best_score is None or bic < best_score:
            best_model = model
            best_score = bic
            best_info = info

    if best_model is None:
        raise ValueError("No valid GMM component setting was evaluated.")

    return best_model, {"features": feature_columns, **best_info}


def sample_gmm(
    frame: pd.DataFrame,
    *,
    n_samples: int = 1000,
    candidate_components: tuple[int, ...] = (2, 5, 8, 10, 15, 20, 25, 30, 40, 50),
    random_state: int = 42,
) -> tuple[pd.DataFrame, GaussianMixture, dict]:
    """Fit the GMM and generate synthetic normal samples with the same schema."""
    model, info = fit_gmm_baseline(
        frame,
        candidate_components=candidate_components,
        random_state=random_state,
    )
    feature_columns = info["features"]
    np.random.seed(random_state)
    synthetic = model.sample(n_samples=n_samples)[0]
    synthetic_frame = pd.DataFrame(synthetic, columns=feature_columns)
    return synthetic_frame, model, info


if __name__ == "__main__":
    df = load_gmm_inputs()
    print(f"Loaded GMM inputs: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df.head())

    sampled, model, info = sample_gmm(df, n_samples=5000)
    print(f"\nBest GMM: {info['n_components']} components")
    print(f"BIC: {info['bic']:.3f}")
    print(f"AIC: {info['aic']:.3f}")
    print(sampled.head())
