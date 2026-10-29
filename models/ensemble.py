r"""Dual-Generator TabDDPM + Grouped GMM Ensemble and Boundary Rejection Engine.

Combines synthetic candidates from:
1. Deep TabDDPM (continuous-discrete score-based diffusion model)
2. Protocol-Grouped GMM V2 (stratified Gaussian Mixture with conditional donor sampling)
Filters candidates with a supervised LightGBM boundary critic, enforces physical protocol
invariants, and exports competition submission files.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Thread safety settings
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import polars as pl
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.gmm_baseline import fit_gmm_baseline
from models.gmm_candidates import (
    allocate_group_sample_counts,
    generation_group_masks,
    sample_component_conditional_donors,
)
from models.tabddpm import (
    GaussianMultinomialDiffusion,
    TabDDPMModel,
    TabDDPMSampler,
    resolve_device,
)
from utils.boundary_filter import LightGBMBoundaryFilter
from utils.dataset_configs import NSL_KDD, UNSW_NB15
from utils.decoding import TabularDecoder
from utils.domain_rules import enforce_domain_invariants, infer_source_decimal_precision

device = resolve_device("auto")


def load_tabddpm_sampler(
    dataset_name: str, compute_device: torch.device = device
) -> TabDDPMSampler:
    r"""Load the trained TabDDPM checkpoint and return a configured TabDDPMSampler."""
    ckpt_path = REPO_ROOT / f"results/models/{dataset_name}/tabddpm_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing TabDDPM checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=compute_device, weights_only=False)
    model = TabDDPMModel(
        num_numerical=ckpt["num_numerical"],
        cardinalities=ckpt["cardinalities"],
        hidden_dim=512,
        num_blocks=3,
        dropout=0.10,
    ).to(compute_device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diffusion = GaussianMultinomialDiffusion(
        num_timesteps=1000, schedule="cosine", device=compute_device
    )
    return TabDDPMSampler(
        model=model,
        diffusion=diffusion,
        cardinalities=ckpt["cardinalities"],
        num_numerical=ckpt["num_numerical"],
        device=compute_device,
    )


def sample_tabddpm_candidates(
    dataset_name: str,
    n_samples: int,
    decoder: TabularDecoder,
    batch_size: int = 2048,
    seed: int = 42,
) -> pl.DataFrame:
    r"""Generate synthetic candidate rows directly from the trained TabDDPM model."""
    if n_samples <= 0:
        return pl.DataFrame()

    print(
        f"Sampling {n_samples:,} candidates directly from trained TabDDPM model "
        f"({dataset_name.upper()}, device={device})..."
    )
    t0 = time.time()
    sampler = load_tabddpm_sampler(dataset_name, compute_device=device)
    df = sampler.sample_to_dataframe(
        num_samples=n_samples,
        decoder=decoder,
        batch_size=batch_size,
        seed=seed,
    )
    print(f"TabDDPM generation complete: {df.height:,} rows in {time.time() - t0:.1f}s.")
    return df


def fit_and_sample_gmm(
    dataset_name: str,
    decoder: TabularDecoder,
    n_samples: int,
    gmm_version: str = "v2",
    seed: int = 42,
) -> pl.DataFrame:
    r"""Fit GMM and sample candidates using component-conditional donor sampling."""
    if n_samples <= 0:
        return pl.DataFrame()

    input_dir = REPO_ROOT / f"output/{dataset_name}/model_inputs"
    numerical_df = pd.read_csv(input_dir / "gmm_inputs.csv")
    donors = pl.read_parquet(input_dir / "gmm_donors.parquet")

    protocol_column = "protocol_type" if dataset_name == "nsl_kdd" else "proto"
    group_masks = generation_group_masks(
        donors[protocol_column].to_pandas(),
        dataset=dataset_name,
        generation_version=gmm_version,
    )
    group_sample_counts = allocate_group_sample_counts(group_masks, n_samples)

    generated_frames = []
    print(f"Fitting Grouped GMM ({gmm_version.upper()}) and sampling {n_samples:,} candidates...")
    t0 = time.time()

    for group_name, group_mask in group_masks.items():
        count = group_sample_counts.get(group_name, 0)
        if count == 0:
            continue

        group_numerical = numerical_df[group_mask]
        best_model, info = fit_gmm_baseline(
            group_numerical,
            candidate_components=(2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50),
            random_state=seed,
        )

        feature_cols = info["features"]
        X = group_numerical[feature_cols].to_numpy(dtype=float)
        responsibilities = best_model.predict_proba(X)

        sampled_X, sampled_components = best_model.sample(count)
        group_donors = sample_component_conditional_donors(
            donors=donors.filter(pl.Series(group_mask)),
            responsibilities=responsibilities,
            sampled_components=sampled_components,
            random_state=seed,
        )

        group_decoded = decoder.decode_gmm(sampled_X, group_donors)
        generated_frames.append(group_decoded)

    combined_gmm = pl.concat(generated_frames, how="vertical")
    print(
        f"GMM {gmm_version.upper()} generation complete: {combined_gmm.height:,} rows in {time.time() - t0:.1f}s."
    )
    return combined_gmm


def export_competition_submission(
    dataset_name: str,
    target_n: int | None = None,
    alpha: float = 0.65,
    gmm_version: str = "v2",
    tau_reject: float | None = None,
    seed: int = 42,
) -> Path:
    r"""Export competition submission via 65:35 TabDDPM + Grouped GMM V2 + Boundary Filter."""
    dataset_name = dataset_name.lower().replace("-", "_")
    assert dataset_name in ["nsl_kdd", "unsw_nb15"], f"Unsupported dataset: {dataset_name}"

    if target_n is None:
        target_n = 40000 if dataset_name == "nsl_kdd" else 70000

    if tau_reject is None:
        tau_reject = 0.04 if dataset_name == "nsl_kdd" else 0.05

    n_tab_target = round(alpha * target_n)
    n_gmm_target = target_n - n_tab_target

    tab_headroom = 1.6 if dataset_name == "nsl_kdd" else 2.3
    gmm_headroom = 1.3 if dataset_name == "nsl_kdd" else 1.6
    n_tab_pool = round(n_tab_target * tab_headroom)
    n_gmm_pool = round(n_gmm_target * gmm_headroom)

    print(f"\n{'=' * 75}")
    print(f"GENERATING SUBMISSION FROM MODELS: {dataset_name.upper()}")
    print(
        f"Target Rows: {target_n:,} -> {int(alpha * 100)}% TabDDPM ({n_tab_target:,}) + "
        f"{int((1 - alpha) * 100)}% Grouped GMM {gmm_version.upper()} ({n_gmm_target:,})"
    )
    print(
        f"Candidate Pools: TabDDPM={n_tab_pool:,}, GMM={n_gmm_pool:,} | Boundary Filter: tau <= {tau_reject}"
    )
    print(f"{'=' * 75}\n")

    input_dir = REPO_ROOT / f"output/{dataset_name}/model_inputs"
    decoder = TabularDecoder.from_directory(input_dir)
    config = NSL_KDD if dataset_name == "nsl_kdd" else UNSW_NB15
    train_csv = (
        REPO_ROOT / f"data/{'NSL-KDD' if dataset_name == 'nsl_kdd' else 'UNSW-NB15'}/train.csv"
    )

    # 1. Load boundary filter
    filter_path = REPO_ROOT / f"results/models/boundary_filter_{dataset_name}.pkl"
    boundary_filter = LightGBMBoundaryFilter.load(filter_path)
    if hasattr(boundary_filter, "model"):
        boundary_filter.model.set_params(n_jobs=1)

    # 2. Sample raw candidates directly from both models
    df_tab = sample_tabddpm_candidates(
        dataset_name, n_samples=n_tab_pool, decoder=decoder, seed=seed
    )
    df_gmm = fit_and_sample_gmm(
        dataset_name, decoder=decoder, n_samples=n_gmm_pool, gmm_version=gmm_version, seed=seed
    )

    # 3. Apply LightGBM boundary filter
    print(f"\nApplying LightGBM Negative Boundary Filter (tau <= {tau_reject})...")
    tab_probs = boundary_filter.predict_anomaly_prob(df_tab.to_pandas())
    gmm_probs = boundary_filter.predict_anomaly_prob(df_gmm.to_pandas())

    tab_mask = tab_probs <= tau_reject
    gmm_mask = gmm_probs <= tau_reject

    tab_surv = df_tab.filter(pl.Series("surv", tab_mask))
    gmm_surv = df_gmm.filter(pl.Series("surv", gmm_mask))

    print(
        f"TabDDPM survivors (tau <= {tau_reject}): {tab_surv.height:,} / {df_tab.height:,} "
        f"({tab_mask.mean() * 100:.1f}%)"
    )
    print(
        f"GMM V2  survivors (tau <= {tau_reject}): {gmm_surv.height:,} / {df_gmm.height:,} "
        f"({gmm_mask.mean() * 100:.1f}%)"
    )

    tab_surv_p = tab_probs[tab_mask]
    gmm_surv_p = gmm_probs[gmm_mask]

    selected_tab = (
        tab_surv[np.argsort(tab_surv_p)[:n_tab_target]]
        if tab_surv.height >= n_tab_target
        else df_tab[np.argsort(tab_probs)[:n_tab_target]]
    )
    selected_gmm = (
        gmm_surv[np.argsort(gmm_surv_p)[:n_gmm_target]]
        if gmm_surv.height >= n_gmm_target
        else df_gmm[np.argsort(gmm_probs)[:n_gmm_target]]
    )

    combined = pl.concat([selected_tab, selected_gmm], how="vertical")
    print(
        f"Assembled exact 65:35 ensemble ({selected_tab.height:,} TabDDPM + "
        f"{selected_gmm.height:,} GMM V2 = {combined.height:,} rows)."
    )

    # 4. Enforce domain invariants and decimal precision
    cat_cols = list(config.categorical)
    numeric_cols = [c for c in decoder.features if c not in cat_cols]
    precision = infer_source_decimal_precision(train_csv, decoder.features, numeric_cols)
    final_synthetic = enforce_domain_invariants(combined, dataset_name, precision)

    # 5. Format with 'id' column as the very first column
    submission = final_synthetic.with_columns(
        pl.arange(0, target_n, dtype=pl.Int64).alias("id")
    ).select(["id", *decoder.features])

    # Guardrail checks
    assert submission.height == target_n, f"Expected {target_n} rows, got {submission.height}"
    assert "is_anomaly" not in submission.columns, "Found is_anomaly in submission!"
    assert submission.columns[0] == "id", "First column is not 'id'!"
    assert submission.null_count().sum_horizontal().sum() == 0, (
        "Null values detected in submission!"
    )

    # Save directly to submissions folder
    root_dir = REPO_ROOT / "submissions"
    root_dir.mkdir(parents=True, exist_ok=True)
    out_csv_root = root_dir / f"submission_{dataset_name}.csv"

    submission.write_csv(out_csv_root)
    print("\n[PASSED] Saved and verified submission:")
    print(f"         Path: {out_csv_root}")
    print(f"         Rows: {submission.height:,} | Cols: {len(submission.columns)} | Nulls: 0\n")
    return out_csv_root
