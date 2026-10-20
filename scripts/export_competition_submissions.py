"""Export competition submissions using the Unified 65:35 Dual-Generator Ensemble with LightGBM Boundary Filtering.

Generates synthetic normal candidates directly from the two best generative models:
1. Deep TabDDPM (65%): Sampled via reverse diffusion from `results/models/{dataset_name}/tabddpm_best.pt`.
2. Protocol-Grouped GMM V2 (35%): Fitted per protocol group (`tcp`, `udp`, `other`) with component-conditional donor decoding.
3. LightGBM Boundary Filter: Rejects candidates with anomaly probability > tau_reject (`results/models/boundary_filter_{dataset_name}.pkl`).
4. Domain Invariants & Precision Alignment: Enforces physical packet rules and training decimal precision.
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

from utils.boundary_filter import LightGBMBoundaryFilter
from utils.dataset_configs import NSL_KDD, UNSW_NB15
from utils.decoding import TabularDecoder
from models.tabddpm import (
    TabDDPMModel,
    GaussianMultinomialDiffusion,
    TabDDPMSampler,
    resolve_device,
)
from models.gmm_baseline import fit_gmm_baseline
from models.gmm_candidates import (
    generation_group_masks,
    allocate_group_sample_counts,
    sample_component_conditional_donors,
    infer_source_decimal_precision,
)

device = resolve_device("auto")


def enforce_domain_invariants(
    df: pl.DataFrame,
    dataset_name: str,
    decimal_precision: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Clean up synthetic rows so they obey real-world network rules."""
    result = df.clone()

    # 1. Round to observed decimal precision
    if decimal_precision:
        round_exprs = [
            pl.col(c).round(places)
            for c, places in decimal_precision.items()
            if c in result.columns
        ]
        if round_exprs:
            result = result.with_columns(round_exprs)

    # 2. Dataset-specific physical invariants
    if dataset_name == "unsw_nb15":
        if "proto" in result.columns:
            non_tcp = pl.col("proto").str.to_lowercase() != "tcp"
            tcp_fields = [
                f for f in ["swin", "stcpb", "dtcpb", "dwin"] if f in result.columns
            ]
            if tcp_fields:
                result = result.with_columns(
                    [
                        pl.when(non_tcp).then(pl.lit(0)).otherwise(pl.col(f)).alias(f)
                        for f in tcp_fields
                    ]
                )

        if all(c in result.columns for c in ["synack", "ackdat", "tcprtt"]):
            synack = result["synack"].to_numpy().astype(float)
            ackdat = result["ackdat"].to_numpy().astype(float)
            total = synack + ackdat
            over_bound = total > 3.5
            scale = np.ones_like(total)
            np.divide(3.5, total, out=scale, where=over_bound)
            synack = np.clip(synack * scale, 0, 3.5)
            ackdat = np.clip(ackdat * scale, 0, 3.5)
            result = result.with_columns(
                [
                    pl.Series("synack", synack).cast(result.schema["synack"]),
                    pl.Series("ackdat", ackdat).cast(result.schema["ackdat"]),
                    pl.Series("tcprtt", synack + ackdat).cast(result.schema["tcprtt"]),
                ]
            )

        if all(c in result.columns for c in ["sbytes", "spkts", "smean"]):
            sbytes = result["sbytes"].to_numpy().astype(float)
            spkts = result["spkts"].to_numpy().astype(float)
            zero_pkts = spkts == 0
            sbytes[zero_pkts] = 0
            smean = np.zeros_like(sbytes)
            np.divide(sbytes, spkts, out=smean, where=~zero_pkts)
            result = result.with_columns(
                [
                    pl.Series("sbytes", sbytes).cast(result.schema["sbytes"]),
                    pl.Series("smean", np.floor(smean + 0.5).astype(np.int64)),
                ]
            )

        if all(c in result.columns for c in ["dbytes", "dpkts", "dmean"]):
            dbytes = result["dbytes"].to_numpy().astype(float)
            dpkts = result["dpkts"].to_numpy().astype(float)
            zero_dpkts = dpkts == 0
            dbytes[zero_dpkts] = 0
            dmean = np.zeros_like(dbytes)
            np.divide(dbytes, dpkts, out=dmean, where=~zero_dpkts)
            result = result.with_columns(
                [
                    pl.Series("dbytes", dbytes).cast(result.schema["dbytes"]),
                    pl.Series("dmean", np.floor(dmean + 0.5).astype(np.int64)),
                ]
            )
    return result


def load_tabddpm_sampler(dataset_name: str, compute_device: torch.device = device) -> TabDDPMSampler:
    """Load the trained TabDDPM checkpoint and return a configured TabDDPMSampler."""
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
    """Generate synthetic candidate rows directly from the trained TabDDPM model."""
    if n_samples <= 0:
        return pl.DataFrame()

    print(f"Sampling {n_samples:,} candidates directly from trained TabDDPM model ({dataset_name.upper()}, device={device})...")
    t0 = time.time()
    sampler = load_tabddpm_sampler(dataset_name, compute_device=device)
    df = sampler.sample_to_dataframe(
        num_samples=n_samples,
        decoder=decoder,
        batch_size=batch_size,
        seed=seed,
    )
    print(f"TabDDPM generated {df.height:,} rows in {time.time() - t0:.1f}s.")
    return df


def fit_and_sample_gmm(
    dataset_name: str,
    decoder: TabularDecoder,
    n_samples: int,
    gmm_version: str = "v2",
    seed: int = 42,
) -> pl.DataFrame:
    """Fit Protocol-Grouped GMM (v2) directly on model inputs and sample candidates."""
    if n_samples <= 0:
        return pl.DataFrame()

    print(f"Fitting and sampling {n_samples:,} candidates directly from Protocol-Grouped GMM {gmm_version.upper()} ({dataset_name.upper()})...")
    t0 = time.time()
    input_dir = REPO_ROOT / f"output/{dataset_name}/model_inputs"
    numerical_df = pd.read_csv(input_dir / "gmm_inputs.csv")
    donors = pl.read_parquet(input_dir / "gmm_donors.parquet")

    protocol_column = "protocol_type" if dataset_name == "nsl_kdd" else "proto"
    group_masks = generation_group_masks(
        dataset_name,
        gmm_version,
        donors[protocol_column].to_list(),
    )
    group_sample_counts = allocate_group_sample_counts(group_masks, n_samples)

    generated_frames = []
    for group_name, group_mask in group_masks.items():
        count = group_sample_counts[group_name]
        if count == 0:
            continue

        group_numerical = numerical_df[group_mask]
        best_model, info = fit_gmm_baseline(
            group_numerical,
            candidate_components=(5, 10, 15, 20),
            random_state=seed,
        )

        feature_cols = info["features"]
        X = group_numerical[feature_cols].to_numpy(dtype=float)
        responsibilities = best_model.predict_proba(X)

        sampled_X, sampled_components = best_model.sample(count)

        group_donors = sample_component_conditional_donors(
            sampled_components,
            responsibilities,
            donors.filter(pl.Series(group_mask)),
            random_state=seed,
        )
        group_decoded = decoder.decode_gmm(sampled_X, group_donors)
        generated_frames.append(group_decoded)

    df = pl.concat(generated_frames, how="vertical")
    print(f"Protocol-Grouped GMM {gmm_version.upper()} generated {df.height:,} rows in {time.time() - t0:.1f}s.")
    return df


# Compatibility wrappers that generate directly from the models
def load_tabddpm_candidates(
    dataset_name: str,
    n_samples: int | None = None,
    decoder: TabularDecoder | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate TabDDPM candidates directly from the trained TabDDPM model."""
    if decoder is None:
        decoder = TabularDecoder.from_directory(REPO_ROOT / f"output/{dataset_name}/model_inputs")
    target = n_samples if n_samples is not None else (50000 if dataset_name == "nsl_kdd" else 100000)
    return sample_tabddpm_candidates(dataset_name, n_samples=target, decoder=decoder, seed=seed)


def load_gmm_candidates(
    dataset_name: str,
    decoder: TabularDecoder | None = None,
    n_samples: int | None = None,
    gmm_version: str = "v2",
    seed: int = 42,
) -> pl.DataFrame:
    """Generate GMM candidates directly from the Protocol-Grouped GMM V2 model."""
    if decoder is None:
        decoder = TabularDecoder.from_directory(REPO_ROOT / f"output/{dataset_name}/model_inputs")
    target = n_samples if n_samples is not None else (25000 if dataset_name == "nsl_kdd" else 40000)
    return fit_and_sample_gmm(dataset_name, decoder=decoder, n_samples=target, gmm_version=gmm_version, seed=seed)


def export_competition_submission(
    dataset_name: str,
    target_n: int,
    alpha: float = 0.65,
    gmm_version: str = "v2",
    tau_reject: float | None = None,
    seed: int = 42,
) -> Path:
    """Generate and export the exact N-row competition submission CSV using the standardized 65:35 + GMM V2 recipe."""
    if tau_reject is None:
        tau_reject = 0.04 if dataset_name == "nsl_kdd" else 0.05

    n_tab_target = int(round(alpha * target_n))
    n_gmm_target = target_n - n_tab_target

    # Candidate pool headroom so both generators have ample survivors under tau_reject
    tab_headroom = 1.6 if dataset_name == "nsl_kdd" else 2.3
    gmm_headroom = 1.3 if dataset_name == "nsl_kdd" else 1.6
    n_tab_pool = int(round(n_tab_target * tab_headroom))
    n_gmm_pool = int(round(n_gmm_target * gmm_headroom))

    print(f"\n{'=' * 75}")
    print(f"GENERATING SUBMISSION FROM MODELS: {dataset_name.upper()}")
    print(f"Target Rows: {target_n:,} -> {int(alpha*100)}% TabDDPM ({n_tab_target:,}) + {int((1-alpha)*100)}% Grouped GMM {gmm_version.upper()} ({n_gmm_target:,})")
    print(f"Candidate Pools: TabDDPM={n_tab_pool:,}, GMM={n_gmm_pool:,} | Boundary Filter: tau <= {tau_reject}")
    print(f"{'=' * 75}\n")

    input_dir = REPO_ROOT / f"output/{dataset_name}/model_inputs"
    decoder = TabularDecoder.from_directory(input_dir)
    config = NSL_KDD if dataset_name == "nsl_kdd" else UNSW_NB15
    train_csv = REPO_ROOT / f"data/{'NSL-KDD' if dataset_name == 'nsl_kdd' else 'UNSW-NB15'}/train.csv"

    # 1. Load trained LightGBM boundary filter first
    filter_path = REPO_ROOT / f"results/models/boundary_filter_{dataset_name}.pkl"
    boundary_filter = LightGBMBoundaryFilter.load(filter_path)
    if hasattr(boundary_filter, "model"):
        boundary_filter.model.set_params(n_jobs=1)

    # 2. Sample raw candidates directly from both models (TabDDPM + Grouped GMM V2)
    df_tab = sample_tabddpm_candidates(dataset_name, n_samples=n_tab_pool, decoder=decoder, seed=seed)
    df_gmm = fit_and_sample_gmm(dataset_name, decoder=decoder, n_samples=n_gmm_pool, gmm_version=gmm_version, seed=seed)

    # 3. Score with LightGBM boundary filter and select exact 65:35 target split
    tab_probs = boundary_filter.predict_anomaly_prob(df_tab.to_pandas())
    gmm_probs = boundary_filter.predict_anomaly_prob(df_gmm.to_pandas())

    tab_mask = tab_probs <= tau_reject
    gmm_mask = gmm_probs <= tau_reject
    tab_surv = df_tab.filter(pl.Series("s", tab_mask))
    gmm_surv = df_gmm.filter(pl.Series("s", gmm_mask))
    tab_surv_p = tab_probs[tab_mask]
    gmm_surv_p = gmm_probs[gmm_mask]

    print(f"TabDDPM survivors (tau <= {tau_reject}): {tab_surv.height:,} / {df_tab.height:,} ({tab_mask.mean()*100:.1f}%)")
    print(f"GMM V2  survivors (tau <= {tau_reject}): {gmm_surv.height:,} / {df_gmm.height:,} ({gmm_mask.mean()*100:.1f}%)")

    if dataset_name == "nsl_kdd":
        # Diversity-preserving sampling across surviving normal modes
        selected_tab = (
            tab_surv.sample(n=n_tab_target, seed=seed)
            if tab_surv.height >= n_tab_target
            else df_tab[np.argsort(tab_probs)[:n_tab_target]]
        )
        selected_gmm = (
            gmm_surv.sample(n=n_gmm_target, seed=seed)
            if gmm_surv.height >= n_gmm_target
            else df_gmm[np.argsort(gmm_probs)[:n_gmm_target]]
        )
    else:
        # Cleanest-rank selection within each generator pool
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
    print(f"Assembled exact 65:35 ensemble ({selected_tab.height:,} TabDDPM + {selected_gmm.height:,} GMM V2 = {combined.height:,} rows).")

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
    assert submission.null_count().sum_horizontal().sum() == 0, "Null values detected in submission!"

    # Save directly to submissions folder
    root_dir = REPO_ROOT / "submissions"
    root_dir.mkdir(parents=True, exist_ok=True)
    out_csv_root = root_dir / f"submission_{dataset_name}.csv"

    submission.write_csv(out_csv_root)
    print(f"\n[PASSED] Saved and verified submission:")
    print(f"         Path: {out_csv_root}")
    print(f"         Rows: {submission.height:,} | Cols: {len(submission.columns)} | Nulls: 0\n")
    return out_csv_root


if __name__ == "__main__":
    print("Exporting competition submissions directly from trained models (65:35 TabDDPM + Grouped GMM V2)...")
    export_competition_submission("nsl_kdd", target_n=40000, alpha=0.65, tau_reject=0.04)
    export_competition_submission("unsw_nb15", target_n=70000, alpha=0.65, tau_reject=0.05)
    print("All submissions exported and verified successfully!")
