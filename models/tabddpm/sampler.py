"""Reverse diffusion sampling engine for TabDDPM.

Generates synthetic tabular candidates by iteratively denoising pure Gaussian
noise and uniform categorical states from timestep T-1 down to 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from models.tabddpm.diffusion import GaussianMultinomialDiffusion
from models.tabddpm.modules import TabDDPMModel

if TYPE_CHECKING:
    import polars as pl

    from utils.decoding import TabularDecoder


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    r"""Resolve target PyTorch compute device automatically or explicitly.

    Args:
        device: Device specification ('cuda', 'mps', 'cpu', 'auto', or None).

    Returns:
        torch.device instance.
    """
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        cleaned = device.strip().lower()
        if cleaned in {"auto", ""}:
            pass
        elif cleaned in {"gpu", "cud"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(cleaned)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TabDDPMSampler:
    r"""Reverse diffusion sampling coordinator for tabular generation.

    Coordinates reverse Gaussian sampling for continuous numerical columns and
    Bayesian posterior categorical transitions for discrete categories.
    """

    def __init__(
        self,
        model: TabDDPMModel,
        diffusion: GaussianMultinomialDiffusion,
        cardinalities: Sequence[int],
        num_numerical: int,
        device: torch.device | str | None = None,
    ) -> None:
        r"""Initialize reverse sampler.

        Args:
            model: Trained TabDDPM neural denoiser.
            diffusion: GaussianMultinomialDiffusion engine with precomputed schedules.
            cardinalities: Sequence of discrete state counts for each categorical feature.
            num_numerical: Number of continuous features.
            device: Target device (defaults to auto-detection: cuda -> mps -> cpu).
        """
        self.device: torch.device = resolve_device(device)
        self.model: TabDDPMModel = model.to(self.device)
        self.diffusion: GaussianMultinomialDiffusion = diffusion.to(self.device)
        self.cardinalities: list[int] = list(cardinalities)
        self.num_numerical: int = num_numerical

    def sample(
        self,
        num_samples: int,
        batch_size: int = 1024,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Generate synthetic continuous and categorical tabular arrays.

        Args:
            num_samples: Total number of synthetic candidate rows to generate.
            batch_size: Batch size for batched reverse diffusion passes.
            seed: Optional random seed for reproducible sampling.

        Returns:
            Tuple of:
            - sampled_num: float32 ndarray of shape (num_samples, num_numerical).
            - sampled_cat: int64 ndarray of shape (num_samples, len(cardinalities)).
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.model.eval()

        collected_num: list[np.ndarray] = []
        collected_cat: list[np.ndarray] = []
        remaining = num_samples

        while remaining > 0:
            curr_b = min(remaining, batch_size)

            # 1. Initialize continuous features from pure standard normal prior N(0, I)
            if self.num_numerical > 0:
                x_t_num: torch.Tensor | None = torch.randn(
                    curr_b, self.num_numerical, device=self.device, dtype=torch.float32
                )
            else:
                x_t_num = None

            # 2. Initialize categorical features from uniform discrete distribution
            if len(self.cardinalities) > 0:
                cat_cols = [
                    torch.randint(0, k, (curr_b,), device=self.device, dtype=torch.long)
                    for k in self.cardinalities
                ]
                x_t_cat: torch.Tensor | None = torch.stack(cat_cols, dim=1)
            else:
                x_t_cat = None

            # 3. Iterative reverse diffusion loop from t = T-1 down to t = 0
            with torch.no_grad():
                for t in reversed(range(self.diffusion.num_timesteps)):
                    t_tensor = torch.full((curr_b,), t, dtype=torch.long, device=self.device)
                    noise_pred, cat_logits = self.model(x_t_num, x_t_cat, t_tensor)

                    if self.num_numerical > 0 and x_t_num is not None:
                        assert noise_pred is not None
                        x_t_num = self.diffusion.p_sample_numerical_step(
                            x_t_num, t_tensor, noise_pred
                        )

                    if len(self.cardinalities) > 0 and x_t_cat is not None:
                        x_t_cat = self.diffusion.p_sample_categorical_step(
                            x_t_cat, t_tensor, cat_logits, self.cardinalities
                        )

            # 4. Collect sampled batch
            if x_t_num is not None:
                collected_num.append(x_t_num.detach().cpu().numpy().astype(np.float32))
            if x_t_cat is not None:
                collected_cat.append(x_t_cat.detach().cpu().numpy().astype(np.int64))

            remaining -= curr_b

        final_num = (
            np.concatenate(collected_num, axis=0)
            if collected_num
            else np.empty((num_samples, 0), dtype=np.float32)
        )
        final_cat = (
            np.concatenate(collected_cat, axis=0)
            if collected_cat
            else np.empty((num_samples, 0), dtype=np.int64)
        )

        return final_num, final_cat

    def sample_to_dataframe(
        self,
        num_samples: int,
        decoder: TabularDecoder,
        batch_size: int = 1024,
        seed: int | None = None,
    ) -> pl.DataFrame:
        r"""Generate synthetic candidates and decode directly into tabular Polars DataFrame.

        Args:
            num_samples: Total number of rows to sample.
            decoder: TabularDecoder instance loaded with dataset artifacts.
            batch_size: Batch size for reverse diffusion.
            seed: Random seed for sampling.

        Returns:
            Polars DataFrame with valid feature types, strings, and clip bounds.
        """
        sampled_num, sampled_cat = self.sample(
            num_samples=num_samples, batch_size=batch_size, seed=seed
        )
        return decoder.decode_tabddpm(sampled_num, sampled_cat)
