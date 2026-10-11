"""Gaussian and Multinomial Hybrid Diffusion Schedules and Transitions.

Implements hybrid tabular diffusion (Kotelnikov et al., ICML 2023):
- Gaussian forward/reverse diffusion for continuous numerical features.
- Multinomial transition forward/reverse diffusion for discrete categorical features.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F


def make_beta_schedule(
    schedule: Literal["cosine", "linear"],
    num_timesteps: int = 1000,
    linear_start: float = 1e-4,
    linear_end: float = 0.02,
) -> torch.Tensor:
    r"""Compute the variance schedule beta_1, ..., beta_T.

    Args:
        schedule: Noise schedule type ('cosine' or 'linear').
        num_timesteps: Total diffusion steps T (default: 1000).
        linear_start: Initial beta value for linear schedule.
        linear_end: Final beta value for linear schedule.

    Returns:
        1D torch.Tensor of shape (num_timesteps,) with dtype float32.
    """
    if schedule == "linear":
        betas = torch.linspace(linear_start, linear_end, num_timesteps, dtype=torch.float32)
    elif schedule == "cosine":
        # Cosine schedule from Nichol & Dhariwal (2021)
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, min=1e-5, max=0.999)
    else:
        raise ValueError(f"Unsupported schedule: {schedule}")
    return betas


class GaussianMultinomialDiffusion:
    r"""Mathematical engine for hybrid Gaussian and Multinomial diffusion.

    Pre-computes closed-form coefficients for forward sampling q(x_t | x_0)
    and reverse posterior transitions p_theta(x_{t-1} | x_t).
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: Literal["cosine", "linear"] = "cosine",
        device: torch.device | None = None,
    ) -> None:
        self.num_timesteps: int = num_timesteps
        self.device: torch.device = device or torch.device("cpu")

        betas = make_beta_schedule(schedule, num_timesteps).to(self.device)
        self.betas: torch.Tensor = betas
        self.alphas: torch.Tensor = 1.0 - betas
        self.alphas_cumprod: torch.Tensor = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev: torch.Tensor = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Continuous Gaussian diffusion parameters
        self.sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod: torch.Tensor = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas: torch.Tensor = torch.sqrt(1.0 / self.alphas)

        # Posterior variance for q(x_{t-1} | x_t, x_0)
        self.posterior_variance: torch.Tensor = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped: torch.Tensor = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )

    def to(self, device: torch.device) -> GaussianMultinomialDiffusion:
        """Move diffusion tensors to specified device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        return self

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
        """Extract coefficients at timesteps t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # -------------------------------------------------------------------------
    # Continuous Features (Gaussian Diffusion)
    # -------------------------------------------------------------------------

    def q_sample_numerical(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Sample continuous noisy state q(x_t | x_0) at timestep t.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def p_sample_numerical_step(
        self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor
    ) -> torch.Tensor:
        r"""Single reverse diffusion step p_theta(x_{t-1} | x_t) for continuous features.

        mu = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * noise_pred)
        x_{t-1} = mu + sigma_t * z
        """
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        beta_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        model_mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_one_minus_alpha * noise_pred)

        # Variance is non-zero only for t > 0
        nonzero_mask = (t != 0).float().reshape(t.shape[0], *((1,) * (len(x_t.shape) - 1)))
        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + nonzero_mask * torch.sqrt(posterior_var) * noise

    # -------------------------------------------------------------------------
    # Categorical Features (Multinomial Transition Diffusion)
    # -------------------------------------------------------------------------

    def q_sample_categorical(
        self, c_0: torch.Tensor, t: torch.Tensor, cardinalities: list[int]
    ) -> torch.Tensor:
        r"""Sample noisy categorical state q(c_t | c_0) at timestep t.

        For each categorical feature with K states:
        q(c_t | c_0) = alpha_bar_t * one_hot(c_0) + (1 - alpha_bar_t) / K
        """
        batch_size = c_0.shape[0]
        alpha_bar = self._extract(self.alphas_cumprod, t, (batch_size, 1))

        sampled_cats = []
        for i, k in enumerate(cardinalities):
            val_0 = c_0[:, i]  # (B,) int64
            one_hot = F.one_hot(val_0, num_classes=k).float()  # (B, K)
            # Transition probability vector: mixture of clean label and uniform distribution
            prob_t = alpha_bar * one_hot + (1.0 - alpha_bar) / float(k)
            # Sample from categorical distribution
            cat_t = torch.multinomial(prob_t, num_samples=1).squeeze(-1)  # (B,)
            sampled_cats.append(cat_t)

        return torch.stack(sampled_cats, dim=1)  # (B, D_cat)

    def p_sample_categorical_step(
        self,
        c_t: torch.Tensor,
        t: torch.Tensor,
        pred_logits_list: list[torch.Tensor],
        cardinalities: list[int],
    ) -> torch.Tensor:
        r"""Single reverse diffusion step p_theta(c_{t-1} | c_t) for categorical features.

        Computes posterior q(c_{t-1} | c_t, pred_c_0) by Bayes rule.
        """
        batch_size = c_t.shape[0]
        alpha_t = self._extract(self.alphas, t, (batch_size, 1))
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev, t, (batch_size, 1))

        sampled_prev = []
        for i, k in enumerate(cardinalities):
            logits = pred_logits_list[i]  # (B, K)
            p_0_pred = F.softmax(logits, dim=-1)  # (B, K)

            curr_c = c_t[:, i]
            curr_one_hot = F.one_hot(curr_c, num_classes=k).float()

            p_step = alpha_t * curr_one_hot + (1.0 - alpha_t) / float(k)
            p_accum = alpha_bar_prev * p_0_pred + (1.0 - alpha_bar_prev) / float(k)

            posterior_prob = p_step * p_accum
            posterior_prob = posterior_prob / (posterior_prob.sum(dim=-1, keepdim=True) + 1e-12)

            c_prev_sample = torch.multinomial(posterior_prob, num_samples=1).squeeze(-1)
            c_prev_argmax = torch.argmax(p_0_pred, dim=-1)
            is_zero = (t == 0).reshape(-1)
            c_prev = torch.where(is_zero, c_prev_argmax, c_prev_sample)
            sampled_prev.append(c_prev)

        return torch.stack(sampled_prev, dim=1)
