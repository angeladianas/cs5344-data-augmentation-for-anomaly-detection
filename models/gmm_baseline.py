r"""Gaussian Mixture Model (GMM) baseline training and evaluation utilities.

Provides functions to select optimal GMM component counts via Bayesian Information
Criterion (BIC) and generate synthetic continuous feature vectors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GMM_INPUTS = REPO_ROOT / "output" / "unsw_nb15" / "model_inputs" / "gmm_inputs.csv"
DEFAULT_CANDIDATE_COMPONENTS = (2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250)


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
    candidate_components: tuple[int, ...] = DEFAULT_CANDIDATE_COMPONENTS,
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
    criteria = []

    for n_components in sorted(set(candidate_components)):
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
        aic = model.aic(X)
        criteria.append(
            {
                "n_components": n_components,
                "bic": float(bic),
                "aic": float(aic),
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
            }
        )
        if not model.converged_:
            continue
        info = {
            "n_components": n_components,
            "bic": float(bic),
            "aic": float(aic),
            "converged": bool(model.converged_),
            "n_iter": int(model.n_iter_),
            "weights": model.weights_.copy(),
            "means": model.means_.copy(),
            "covariances": model.covariances_.copy(),
        }
        if best_score is None or bic < best_score:
            best_model = model
            best_score = bic
            best_info = info

    if best_model is None:
        raise ValueError("No candidate GMM converged; inspect the input and fit settings.")

    return best_model, {"features": feature_columns, "criteria": criteria, **best_info}


def plot_gmm_information_criteria(criteria: list[dict[str, int | float]]) -> None:
    """Plot the fitted models' BIC and AIC scores by component count."""
    component_counts = [result["n_components"] for result in criteria]
    bic_scores = [result["bic"] for result in criteria]
    aic_scores = [result["aic"] for result in criteria]

    _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(component_counts, bic_scores, marker="o", label="BIC")
    ax.plot(component_counts, aic_scores, marker="o", label="AIC")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Information criterion score")
    ax.set_title("GMM model selection")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def sample_gmm(
    frame: pd.DataFrame,
    *,
    n_samples: int = 1000,
    candidate_components: tuple[int, ...] = DEFAULT_CANDIDATE_COMPONENTS,
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
    print(f"Converged: {info['converged']} in {info['n_iter']} iterations")
    print(f"BIC: {info['bic']:.3f}")
    print(f"AIC: {info['aic']:.3f}")
    print("\nCandidate fits:")
    for result in info["criteria"]:
        print(
            f"K={result['n_components']:>3}: "
            f"converged={result['converged']}, "
            f"iterations={result['n_iter']}, "
            f"BIC={result['bic']:.3f}"
        )
    plot_gmm_information_criteria(info["criteria"])
    print(sampled.head())
