"""Build clean, objective, chronologically structured step_5_ensemble.ipynb."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_cell(cell_type: str, source_text: str):
    lines = [line + "\n" for line in source_text.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


cells = []

# Cell 1: Overview
cells.append(
    make_cell(
        "markdown",
        """# Step 5: Dual-Generator Ensemble and Boundary Filtering

## Objective
Generate synthetic normal datasets for NSL-KDD and UNSW-NB15 by combining TabDDPM (diffusion) and Protocol-Grouped GMM V2 (mixture model). Candidate samples are filtered using a supervised LightGBM boundary critic and cleaned with domain rules to maximize downstream detection utility ($S_{\\text{val}}$).

## Pipeline Architecture
1. **Candidate Generation**: Sample continuous and categorical features from TabDDPM and Protocol-Grouped GMM V2.
2. **Boundary Filtering**: Discard candidates with anomaly probability above $\\tau$ to prevent synthetic points from leaking into attack regions.
3. **Domain Cleaning**: Round numeric columns to training precision and enforce protocol-specific invariants.
4. **Ratio Evaluation**: Test mixture ratios on validation splits to measure downstream ECOD and Isolation Forest utility.
5. **Submission Export**: Generate the final $N$-row submission datasets (40,000 for NSL-KDD, 70,000 for UNSW-NB15) using the unified 65:35 recipe.""",
    )
)

# Cell 2: Setup
cells.append(
    make_cell(
        "code",
        """import os
import sys
from pathlib import Path

# Thread safety settings
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch


def find_repo_root():
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "utils").exists():
            return parent
    return cwd

REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print(f"Project root: {REPO_ROOT}")

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
from utils.evaluator import evaluate_synthetic_utility

device = resolve_device("auto")
print(f"Compute device: {device}")""",
    )
)

# Cell 3: Generative Engines Markdown
cells.append(
    make_cell(
        "markdown",
        """## 1. Generative Engines: TabDDPM and Protocol-Grouped GMM V2

The ensemble pairs two complementary models:
- **TabDDPM**: A score-based diffusion model trained on normal traffic. It models complex feature interactions and heavy-tailed numeric distributions, strengthening downstream ECOD tail estimation.
- **Protocol-Grouped GMM V2**: Fits separate Gaussian mixture models partitioned by network protocol (`tcp` vs. `non-tcp` for NSL-KDD; `tcp`, `udp`, and `other` for UNSW-NB15). Protocol grouping prevents continuous parameters from mixing across traffic types and preserves mode coverage for Isolation Forest.""",
    )
)

# Cell 4: Generative Engines Code
cells.append(
    make_cell(
        "code",
        """def load_tabddpm_sampler(dataset_name: str, compute_device: torch.device):
    \"\"\"Load the trained TabDDPM checkpoint and return a configured sampler.\"\"\"
    ckpt_path = REPO_ROOT / f"results/models/{dataset_name}/tabddpm_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing weights: {ckpt_path}")

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

    diffusion = GaussianMultinomialDiffusion(num_timesteps=1000, schedule="cosine", device=compute_device)
    return TabDDPMSampler(
        model=model,
        diffusion=diffusion,
        cardinalities=ckpt["cardinalities"],
        num_numerical=ckpt["num_numerical"],
        device=compute_device,
    )

def sample_tabddpm(sampler: TabDDPMSampler, decoder: TabularDecoder, n_samples: int, seed: int = 42) -> pl.DataFrame:
    \"\"\"Sample rows directly from TabDDPM.\"\"\"
    if n_samples <= 0:
        return pl.DataFrame()
    return sampler.sample_to_dataframe(num_samples=n_samples, decoder=decoder, batch_size=2048, seed=seed)

def fit_and_sample_gmm(
    dataset_name: str,
    decoder: TabularDecoder,
    n_samples: int,
    gmm_version: str = "v2",
    seed: int = 42,
) -> pl.DataFrame:
    \"\"\"Fit protocol-grouped GMM on model inputs and sample decoded rows.\"\"\"
    if n_samples <= 0:
        return pl.DataFrame()

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

    return pl.concat(generated_frames, how="vertical")

print("Generators initialized.")""",
    )
)

# Cell 5: Boundary Filtering & Invariants Markdown
cells.append(
    make_cell(
        "markdown",
        """## 2. Boundary Filtering and Domain Rules

Synthetic points must not overlap with attack distributions or violate physical network protocol identities:
- **LightGBM Boundary Filter**: A supervised classifier trained on labeled training data (`is_anomaly == 0` vs. `is_anomaly == 1`). Candidates with predicted anomaly probability above $\\tau_{\\text{reject}}$ are discarded.
- **Decimal Precision**: Numeric columns are rounded to match the maximum observed decimal places in `train.csv`.
- **UNSW-NB15 Protocol Invariants**:
  - Non-TCP records have window and sequence counters set to zero (`swin`, `stcpb`, `dtcpb`, `dwin`).
  - Round trip time identity: `tcprtt = synack + ackdat`, capped at 3.5 seconds.
  - Byte-packet consistency: Packet means are recalculated as `round(bytes / pkts)`.""",
    )
)

# Cell 6: Boundary Filtering & Invariants Code
cells.append(
    make_cell(
        "code",
        """def infer_source_decimal_precision(source_path: Path, feature_cols: list[str], numeric_cols: list[str]) -> dict[str, int]:
    \"\"\"Find decimal precision used in the original training CSV.\"\"\"
    precision = {col: 0 for col in numeric_cols}
    import csv
    with open(source_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in numeric_cols:
                val = row[col].strip()
                if '.' in val:
                    places = len(val.split('.')[1])
                    precision[col] = max(precision[col], places)
    return precision

def enforce_domain_invariants(df: pl.DataFrame, dataset_name: str, decimal_precision: dict[str, int] | None = None) -> pl.DataFrame:
    \"\"\"Enforce protocol invariants and decimal precision on synthetic rows.\"\"\"
    result = df.clone()

    if decimal_precision:
        round_exprs = [pl.col(c).round(places) for c, places in decimal_precision.items() if c in result.columns]
        if round_exprs:
            result = result.with_columns(round_exprs)

    if dataset_name == "unsw_nb15":
        if "proto" in result.columns:
            non_tcp = pl.col("proto").str.to_lowercase() != "tcp"
            tcp_fields = [f for f in ["swin", "stcpb", "dtcpb", "dwin"] if f in result.columns]
            if tcp_fields:
                result = result.with_columns([
                    pl.when(non_tcp).then(pl.lit(0)).otherwise(pl.col(f)).alias(f)
                    for f in tcp_fields
                ])

        if all(c in result.columns for c in ["synack", "ackdat", "tcprtt"]):
            synack = result["synack"].to_numpy().astype(float)
            ackdat = result["ackdat"].to_numpy().astype(float)
            total = synack + ackdat
            over_bound = total > 3.5
            scale = np.ones_like(total)
            np.divide(3.5, total, out=scale, where=over_bound)
            synack = np.clip(synack * scale, 0, 3.5)
            ackdat = np.clip(ackdat * scale, 0, 3.5)
            result = result.with_columns([
                pl.Series("synack", synack).cast(result.schema["synack"]),
                pl.Series("ackdat", ackdat).cast(result.schema["ackdat"]),
                pl.Series("tcprtt", synack + ackdat).cast(result.schema["tcprtt"]),
            ])

        if all(c in result.columns for c in ["sbytes", "spkts", "smean"]):
            sbytes = result["sbytes"].to_numpy().astype(float)
            spkts = result["spkts"].to_numpy().astype(float)
            zero_pkts = spkts == 0
            sbytes[zero_pkts] = 0
            smean = np.zeros_like(sbytes)
            np.divide(sbytes, spkts, out=smean, where=~zero_pkts)
            result = result.with_columns([
                pl.Series("sbytes", sbytes).cast(result.schema["sbytes"]),
                pl.Series("smean", np.floor(smean + 0.5).astype(np.int64)),
            ])

        if all(c in result.columns for c in ["dbytes", "dpkts", "dmean"]):
            dbytes = result["dbytes"].to_numpy().astype(float)
            dpkts = result["dpkts"].to_numpy().astype(float)
            zero_dpkts = dpkts == 0
            dbytes[zero_dpkts] = 0
            dmean = np.zeros_like(dbytes)
            np.divide(dbytes, dpkts, out=dmean, where=~zero_dpkts)
            result = result.with_columns([
                pl.Series("dbytes", dbytes).cast(result.schema["dbytes"]),
                pl.Series("dmean", np.floor(dmean + 0.5).astype(np.int64)),
            ])

    return result

print("Filtering and invariant functions loaded.")""",
    )
)

# Cell 7: Ratio Evaluation Markdown
cells.append(
    make_cell(
        "markdown",
        """## 3. Mixture Ratio Evaluation

The evaluation function samples candidate pools from both generators, applies boundary filtering and domain rules, and evaluates downstream detection performance on unseen validation attacks ($S_{\\text{val}} = (\\text{AUPRC}_{\\text{ECOD}} + \\text{AUPRC}_{\\text{IForest}}) / 2$).""",
    )
)

# Cell 8: Ratio Evaluation Code
cells.append(
    make_cell(
        "code",
        """def run_ratio_sweep(
    dataset_name: str,
    total_candidates: int = 10000,
    ratios: list[tuple[float, float, str]] | None = None,
    gmm_version: str = "v2",
    tau_reject: float = 0.04,
    seed: int = 42,
) -> pd.DataFrame:
    \"\"\"Run ratio sweep and report downstream detection utility.\"\"\"
    if ratios is None:
        ratios = [
            (1.0, 0.0, "100% TabDDPM Alone"),
            (0.7, 0.3, "70% TabDDPM : 30% GMM"),
            (0.65, 0.35, "65% TabDDPM : 35% GMM (Selected Recipe)"),
            (0.5, 0.5, "50% TabDDPM : 50% GMM"),
            (0.3, 0.7, "30% TabDDPM : 70% GMM"),
            (0.0, 1.0, "100% GMM Alone"),
        ]

    input_dir = REPO_ROOT / f"output/{dataset_name}/model_inputs"
    val_path = REPO_ROOT / f"output/{dataset_name}/splits/validation.parquet"
    train_csv = REPO_ROOT / f"data/{'NSL-KDD' if dataset_name == 'nsl_kdd' else 'UNSW-NB15'}/train.csv"

    decoder = TabularDecoder.from_directory(input_dir)
    df_val = pl.read_parquet(val_path)

    config = NSL_KDD if dataset_name == "nsl_kdd" else UNSW_NB15
    cat_cols = list(config.categorical)
    numeric_cols = [c for c in decoder.features if c not in cat_cols]
    precision = infer_source_decimal_precision(train_csv, decoder.features, numeric_cols)

    filter_path = REPO_ROOT / f"results/models/boundary_filter_{dataset_name}.pkl"
    boundary_filter = LightGBMBoundaryFilter.load(filter_path)
    if hasattr(boundary_filter, "model"):
        boundary_filter.model.set_params(n_jobs=1)

    sampler = load_tabddpm_sampler(dataset_name, device)
    tabddpm_pool = sample_tabddpm(sampler, decoder, total_candidates, seed=seed)
    gmm_pool = fit_and_sample_gmm(dataset_name, decoder, total_candidates, gmm_version=gmm_version, seed=seed)

    sweep_results = []
    for alpha, beta, label in ratios:
        n_tab = round(alpha * total_candidates)
        n_gmm = total_candidates - n_tab

        parts = []
        if n_tab > 0:
            parts.append(tabddpm_pool.slice(0, n_tab))
        if n_gmm > 0:
            parts.append(gmm_pool.slice(0, n_gmm))

        combined = pl.concat(parts, how="vertical")
        probs = boundary_filter.predict_anomaly_prob(combined.to_pandas())
        survivors_mask = probs <= tau_reject
        filtered = combined.filter(pl.Series("survived", survivors_mask))
        cleaned = enforce_domain_invariants(filtered, dataset_name, precision)

        eval_metrics = evaluate_synthetic_utility(
            synthetic_df=cleaned,
            validation_df=df_val,
            categorical_cols=cat_cols,
            n_jobs=1,
        )
        s_val = float(eval_metrics["S_val"])
        auprc_ecod = float(eval_metrics["AUPRC_ECOD"])
        auprc_iforest = float(eval_metrics["AUPRC_IForest_mean"])

        sweep_results.append({
            "Dataset": dataset_name.upper(),
            "Configuration": label,
            "TabDDPM %": f"{int(alpha*100)}%",
            "GMM %": f"{int(beta*100)}%",
            "Survivors": f"{cleaned.height:,}",
            "S_val": round(s_val, 5),
            "AUPRC (ECOD)": round(auprc_ecod, 5),
            "AUPRC (IForest)": round(auprc_iforest, 5),
        })

    return pd.DataFrame(sweep_results)

print("Evaluation runner configured.")""",
    )
)

# Cell 9: Experiment 1 NSL-KDD Markdown
cells.append(
    make_cell(
        "markdown",
        """## 4. Empirical Evaluation: NSL-KDD

Testing mixture ratios on NSL-KDD to observe ECOD versus Isolation Forest trade-offs.""",
    )
)

# Cell 10: Experiment 1 NSL-KDD Code
cells.append(
    make_cell(
        "code",
        """nsl_results = run_ratio_sweep(
    dataset_name="nsl_kdd",
    total_candidates=10000,
    tau_reject=0.04,
    seed=42,
)
display(nsl_results)""",
    )
)

# Cell 11: Experiment 2 UNSW-NB15 Markdown
cells.append(
    make_cell(
        "markdown",
        """## 5. Empirical Evaluation: UNSW-NB15

Running the identical evaluation sweep on UNSW-NB15.""",
    )
)

# Cell 12: Experiment 2 UNSW-NB15 Code
cells.append(
    make_cell(
        "code",
        """unsw_results = run_ratio_sweep(
    dataset_name="unsw_nb15",
    total_candidates=10000,
    tau_reject=0.05,
    seed=42,
)
display(unsw_results)""",
    )
)

# Cell 13: Comparison Visualization Markdown
cells.append(
    make_cell(
        "markdown",
        """## 6. Ratio Comparison

Visualizing composite $S_{\\text{val}}$, ECOD, and Isolation Forest scores across mixture ratios.""",
    )
)

# Cell 14: Comparison Visualization Code
cells.append(
    make_cell(
        "code",
        """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (res_df, title) in enumerate([(nsl_results, "NSL-KDD"), (unsw_results, "UNSW-NB15")]):
    ax = axes[idx]
    labels = [c.split("(")[0].strip() for c in res_df["Configuration"]]
    x = np.arange(len(labels))
    width = 0.25

    ax.bar(x - width, res_df["S_val"], width, label="Composite S_val", color="#1f77b4")
    ax.bar(x, res_df["AUPRC (ECOD)"], width, label="ECOD", color="#2ca02c", alpha=0.8)
    ax.bar(x + width, res_df["AUPRC (IForest)"], width, label="IForest", color="#ff7f0e", alpha=0.8)

    ax.set_title(f"{title}: Performance Across Mixture Ratios", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Validation AUPRC", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

plt.tight_layout()
plt.show()""",
    )
)

# Cell 15: Submission Export Markdown
cells.append(
    make_cell(
        "markdown",
        """## 7. Competition Submission Export

Exporting the competition submissions using the standardized 65% TabDDPM : 35% Protocol-Grouped GMM V2 recipe:
- **NSL-KDD**: Exactly 40,000 rows (26,000 TabDDPM + 14,000 GMM V2, $\\tau \\le 0.04$).
- **UNSW-NB15**: Exactly 70,000 rows (45,500 TabDDPM + 24,500 GMM V2, $\\tau \\le 0.05$).
- Output files are written to `submissions/submission_{dataset}.csv`.""",
    )
)

# Cell 16: Submission Export Code
cells.append(
    make_cell(
        "code",
        """from scripts.export_competition_submissions import export_competition_submission

export_competition_submission("nsl_kdd", target_n=40000, alpha=0.65, tau_reject=0.04)
export_competition_submission("unsw_nb15", target_n=70000, alpha=0.65, tau_reject=0.05)""",
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

(REPO_ROOT / "tmp_notebooks/step_5_ensemble.ipynb").write_text(
    json.dumps(nb, indent=1), encoding="utf-8"
)
print("Updated tmp_notebooks/step_5_ensemble.ipynb successfully.")
