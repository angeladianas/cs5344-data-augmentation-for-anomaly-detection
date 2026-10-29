"""Train TabDDPM on intrusion detection datasets and evaluate downstream utility.

Provides a unified CLI runner to:
1. Load preprocessed continuous & discrete arrays (tabddpm_inputs.npz).
2. Fit the hybrid TabDDPM denoiser on Apple Silicon (MPS), CUDA, or CPU.
3. Save the model checkpoint to results/models/.
4. Sample synthetic candidates and decode back to original tabular schema.
5. Score downstream anomaly detection utility S_val via Kaggle-equivalent evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import polars as pl

from models.tabddpm import (
    GaussianMultinomialDiffusion,
    TabDDPMModel,
    TabDDPMSampler,
    TabDDPMTrainer,
    resolve_device,
)
from utils.dataset_configs import NSL_KDD, UNSW_NB15
from utils.decoding import TabularDecoder
from utils.evaluator import evaluate_synthetic_utility


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate TabDDPM.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="nsl_kdd",
        choices=["nsl_kdd", "unsw_nb15"],
        help="Target dataset name ('nsl_kdd' or 'unsw_nb15').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size (default: 256).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension of residual MLP backbone (default: 512).",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=3,
        help="Number of residual MLP blocks (default: 3).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Diffusion timesteps T (default: 1000).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16000,
        help="Number of candidate rows to generate (default: 16000).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for AdamW (default: 2e-4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('mps', 'cuda', 'cpu', or 'auto').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def train_and_evaluate(
    dataset_name: str,
    epochs: int = 100,
    batch_size: int = 256,
    hidden_dim: int = 512,
    num_blocks: int = 3,
    timesteps: int = 1000,
    num_samples: int = 16000,
    lr: float = 2e-4,
    device_str: str = "auto",
    seed: int = 42,
) -> dict[str, object]:
    """Train TabDDPM, generate candidate rows, and evaluate downstream benchmark."""
    device = resolve_device(device_str)
    print(f"\n{'=' * 70}")
    print(f"TabDDPM Pipeline: Dataset={dataset_name} | Device={device} | Epochs={epochs}")
    print(f"{'=' * 70}\n")

    # 1. Load preprocessed model inputs
    input_path = f"output/{dataset_name}/model_inputs/tabddpm_inputs.npz"
    if not os.path.exists(input_path):
        print(f"Preprocessed inputs not found at {input_path}. Running preprocessing pipeline...")
        from utils.preprocessing import run_preprocessing

        config, folder = (
            (NSL_KDD, "NSL-KDD") if dataset_name == "nsl_kdd" else (UNSW_NB15, "UNSW-NB15")
        )
        csv_path = f"data/{folder}/train.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Raw training data not found at {csv_path}")
        run_preprocessing(csv_path, config, output_root="output", seed=seed)

    npz_data = np.load(input_path)
    x_num = npz_data["X_num"]
    x_cat = npz_data["X_cat"]
    cardinalities = npz_data["categorical_cardinalities"].tolist()

    d_num = x_num.shape[1]
    print(
        f"Loaded {len(x_num)} training records: {d_num} continuous, {len(cardinalities)} categorical features."
    )
    print(f"Categorical cardinalities: {cardinalities}")

    # 2. Build model and diffusion engines
    diffusion = GaussianMultinomialDiffusion(
        num_timesteps=timesteps, schedule="cosine", device=device
    )
    model = TabDDPMModel(
        num_numerical=d_num,
        cardinalities=cardinalities,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=0.10,
    ).to(device)

    # 3. Train TabDDPM
    checkpoint_dir = f"results/models/{dataset_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/tabddpm_best.pt"

    trainer = TabDDPMTrainer(
        model=model,
        diffusion=diffusion,
        cardinalities=cardinalities,
        learning_rate=lr,
        batch_size=batch_size,
        epochs=epochs,
        patience=25,
        device=device,
    )

    t0 = time.time()
    history = trainer.fit(
        x_num,
        x_cat,
        val_ratio=0.10,
        seed=seed,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )
    train_time = time.time() - t0
    print(
        f"\nTraining completed in {train_time:.1f}s. Best model checkpoint saved to {checkpoint_path}"
    )

    results_dir = f"results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/tabddpm_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 4. Reverse sample candidate pool
    print(f"\nSampling {num_samples} candidate rows using trained reverse diffusion...")
    t_sample = time.time()
    sampler = TabDDPMSampler(
        model=model,
        diffusion=diffusion,
        cardinalities=cardinalities,
        num_numerical=d_num,
        device=device,
    )

    decoder = TabularDecoder.from_directory(f"output/{dataset_name}/model_inputs")
    df_synthetic = sampler.sample_to_dataframe(
        num_samples=num_samples, decoder=decoder, batch_size=2048, seed=seed
    )
    sample_time = time.time() - t_sample
    print(
        f"Sampling & decoding completed in {sample_time:.1f}s. Synthetic shape: {df_synthetic.shape}"
    )

    # Save synthetic candidate pool
    syn_dir = f"output/{dataset_name}/synthetic"
    os.makedirs(syn_dir, exist_ok=True)
    syn_path = f"{syn_dir}/tabddpm_candidates.parquet"
    df_synthetic.write_parquet(syn_path)
    print(f"Saved candidates to {syn_path}")

    # 5. Evaluate downstream utility against validation set
    val_path = f"output/{dataset_name}/splits/validation.parquet"
    print(f"\nEvaluating synthetic downstream utility against {val_path}...")
    df_val = pl.read_parquet(val_path)

    config = NSL_KDD if dataset_name == "nsl_kdd" else UNSW_NB15
    categorical_cols = list(config.categorical)

    eval_results = evaluate_synthetic_utility(
        synthetic_df=df_synthetic,
        validation_df=df_val,
        categorical_cols=categorical_cols,
    )

    s_val = float(eval_results["S_val"])
    auprc_ecod = float(eval_results["AUPRC_ECOD"])
    auprc_iforest = float(eval_results["AUPRC_IForest_mean"])
    eval_duration = float(eval_results["duration_seconds"])

    print(f"\n{'=' * 70}")
    print(f"RESULTS FOR {dataset_name.upper()} (TabDDPM Alone):")
    print(f"  Composite Score S_val: {s_val:.5f}")
    print(f"  AUPRC (ECOD):          {auprc_ecod:.5f}")
    print(f"  AUPRC (IForest):       {auprc_iforest:.5f}")
    print(f"  Evaluation Time:       {eval_duration:.1f}s")
    print(f"{'=' * 70}\n")

    summary: dict[str, object] = {
        "dataset": dataset_name,
        "epochs": epochs,
        "train_time_seconds": round(train_time, 2),
        "sample_time_seconds": round(sample_time, 2),
        "synthetic_rows": num_samples,
        "S_val": s_val,
        "AUPRC_ECOD": auprc_ecod,
        "AUPRC_IForest_mean": auprc_iforest,
        "duration_seconds": eval_duration,
    }

    results_dir = f"results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/tabddpm_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    r"""Parse command line arguments and execute TabDDPM training, sampling, and evaluation."""
    args = parse_args()
    train_and_evaluate(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        timesteps=args.timesteps,
        num_samples=args.num_samples,
        lr=args.lr,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
