"""Unit and integration test suite for TabDDPM generative engine.

Tests hybrid continuous Gaussian and discrete Multinomial diffusion mechanics,
MLP denoiser architecture, trainer convergence, reverse sampling, and
end-to-end decoding with TabularDecoder.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from models.tabddpm import (
    GaussianMultinomialDiffusion,
    TabDDPMModel,
    TabDDPMSampler,
    TabDDPMTrainer,
    make_beta_schedule,
    resolve_device,
)
from utils.decoding import TabularDecoder


class TestDiffusionSchedules:
    """Tests for beta variance schedules and mathematical transition tensors."""

    def test_linear_schedule_bounds(self) -> None:
        r"""Verify linear beta schedule boundary values, shape, and monotonicity."""
        betas = make_beta_schedule("linear", num_timesteps=100, linear_start=1e-4, linear_end=0.02)
        assert betas.shape == (100,)
        assert betas.dtype == torch.float32
        assert torch.all(betas > 0)
        assert torch.all(betas < 1)
        assert torch.isclose(betas[0], torch.tensor(1e-4))
        assert torch.isclose(betas[-1], torch.tensor(0.02))

    def test_cosine_schedule_bounds(self) -> None:
        r"""Verify cosine beta schedule bounds according to Nichol & Dhariwal formulation."""
        betas = make_beta_schedule("cosine", num_timesteps=100)
        assert betas.shape == (100,)
        assert torch.all(betas >= 1e-5)
        assert torch.all(betas <= 0.999)

    def test_invalid_schedule_raises(self) -> None:
        r"""Verify that unsupported schedule names raise a ValueError."""
        with pytest.raises(ValueError, match="Unsupported schedule"):
            make_beta_schedule("exponential", num_timesteps=100)  # type: ignore[arg-type]


class TestGaussianMultinomialDiffusion:
    """Tests for Gaussian and Multinomial forward/reverse transitions."""

    @pytest.fixture
    def diffusion(self) -> GaussianMultinomialDiffusion:
        r"""Instantiate a fast 50-step diffusion engine for unit tests."""
        return GaussianMultinomialDiffusion(num_timesteps=50, schedule="linear")

    def test_q_sample_numerical(self, diffusion: GaussianMultinomialDiffusion) -> None:
        r"""Verify forward Gaussian q-sampling produces correct tensor shapes and noise scale."""
        x_0 = torch.zeros(16, 5)
        t = torch.full((16,), 49, dtype=torch.long)
        x_t, noise = diffusion.q_sample_numerical(x_0, t)

        assert x_t.shape == (16, 5)
        assert noise.shape == (16, 5)
        # At t=T-1 with linear schedule, variance is large
        assert x_t.std() > 0.5

    def test_p_sample_numerical_step(self, diffusion: GaussianMultinomialDiffusion) -> None:
        r"""Verify single reverse Gaussian denoising step produces finite tensors."""
        x_t = torch.randn(16, 5)
        t = torch.full((16,), 10, dtype=torch.long)
        noise_pred = torch.randn(16, 5)
        x_prev = diffusion.p_sample_numerical_step(x_t, t, noise_pred)

        assert x_prev.shape == (16, 5)
        assert not torch.isnan(x_prev).any()

    def test_q_sample_categorical(self, diffusion: GaussianMultinomialDiffusion) -> None:
        r"""Verify forward categorical transition maintains valid integer class indices."""
        cardinalities = [3, 5, 10]
        c_0 = torch.tensor([[0, 2, 8], [1, 4, 0], [2, 1, 5]], dtype=torch.long)
        t = torch.tensor([5, 10, 20], dtype=torch.long)

        c_t = diffusion.q_sample_categorical(c_0, t, cardinalities)
        assert c_t.shape == (3, 3)
        for j, k in enumerate(cardinalities):
            assert torch.all(c_t[:, j] >= 0)
            assert torch.all(c_t[:, j] < k)

    def test_p_sample_categorical_step(self, diffusion: GaussianMultinomialDiffusion) -> None:
        r"""Verify reverse categorical step generates valid categorical indices."""
        cardinalities = [3, 4]
        c_t = torch.tensor([[1, 2], [0, 3]], dtype=torch.long)
        t = torch.tensor([1, 1], dtype=torch.long)
        pred_logits = [torch.randn(2, 3), torch.randn(2, 4)]

        c_prev = diffusion.p_sample_categorical_step(c_t, t, pred_logits, cardinalities)
        assert c_prev.shape == (2, 2)
        assert torch.all(c_prev[:, 0] >= 0) and torch.all(c_prev[:, 0] < 3)
        assert torch.all(c_prev[:, 1] >= 0) and torch.all(c_prev[:, 1] < 4)


class TestTabDDPMModel:
    """Tests for MLP Denoiser architecture and forward passes."""

    def test_hybrid_forward_shapes(self) -> None:
        r"""Verify hybrid architecture output shapes for continuous noise and categorical logits."""
        cardinalities = [2, 5, 8]
        model = TabDDPMModel(
            num_numerical=10,
            cardinalities=cardinalities,
            embedding_dim=16,
            hidden_dim=64,
            num_blocks=2,
        )

        b = 8
        x_num = torch.randn(b, 10)
        x_cat = torch.tensor([[0, 2, 7]] * b, dtype=torch.long)
        t = torch.randint(0, 100, (b,))

        noise_pred, cat_logits = model(x_num, x_cat, t)

        assert noise_pred is not None
        assert noise_pred.shape == (b, 10)
        assert len(cat_logits) == 3
        assert cat_logits[0].shape == (b, 2)
        assert cat_logits[1].shape == (b, 5)
        assert cat_logits[2].shape == (b, 8)

    def test_pure_numerical_forward(self) -> None:
        r"""Verify forward pass on continuous-only tabular architectures."""
        model = TabDDPMModel(
            num_numerical=6,
            cardinalities=[],
            hidden_dim=32,
            num_blocks=1,
        )
        x_num = torch.randn(4, 6)
        t = torch.zeros(4, dtype=torch.long)
        noise_pred, cat_logits = model(x_num, None, t)

        assert noise_pred is not None
        assert noise_pred.shape == (4, 6)
        assert len(cat_logits) == 0

    def test_pure_categorical_forward(self) -> None:
        r"""Verify forward pass on categorical-only tabular architectures."""
        model = TabDDPMModel(
            num_numerical=0,
            cardinalities=[4, 6],
            hidden_dim=32,
            num_blocks=1,
        )
        x_cat = torch.tensor([[1, 3], [2, 0]], dtype=torch.long)
        t = torch.zeros(2, dtype=torch.long)
        noise_pred, cat_logits = model(None, x_cat, t)

        assert noise_pred is None
        assert len(cat_logits) == 2
        assert cat_logits[0].shape == (2, 4)
        assert cat_logits[1].shape == (2, 6)

    def test_empty_architecture_raises(self) -> None:
        r"""Verify that constructing a model with 0 features raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one"):
            TabDDPMModel(num_numerical=0, cardinalities=[])


class TestTabDDPMSamplerAndTrainer:
    """Tests for trainer convergence and batched reverse sampling."""

    def test_trainer_and_sampler_cycle(self) -> None:
        r"""Verify full training convergence cycle and batched reverse sampling on synthetic data."""
        # Generate toy dataset
        np.random.seed(42)
        torch.manual_seed(42)

        n = 120
        d_num = 4
        cardinalities = [3, 4]

        x_num = np.random.randn(n, d_num).astype(np.float32)
        x_cat = np.column_stack(
            [
                np.random.randint(0, 3, size=n),
                np.random.randint(0, 4, size=n),
            ]
        ).astype(np.int64)

        diffusion = GaussianMultinomialDiffusion(num_timesteps=10, schedule="linear")
        model = TabDDPMModel(
            num_numerical=d_num,
            cardinalities=cardinalities,
            hidden_dim=32,
            num_blocks=1,
        )

        trainer = TabDDPMTrainer(
            model=model,
            diffusion=diffusion,
            cardinalities=cardinalities,
            learning_rate=1e-3,
            batch_size=32,
            epochs=5,
            patience=5,
            device="cpu",
        )

        history = trainer.fit(x_num, x_cat, val_ratio=0.2, verbose=False)
        assert len(history["train_loss"]) == 5
        assert all(loss > 0 for loss in history["train_loss"])

        # Test reverse sampler
        sampler = TabDDPMSampler(
            model=model,
            diffusion=diffusion,
            cardinalities=cardinalities,
            num_numerical=d_num,
            device="cpu",
        )

        sampled_num, sampled_cat = sampler.sample(num_samples=25, batch_size=10, seed=123)

        assert sampled_num.shape == (25, d_num)
        assert sampled_cat.shape == (25, 2)
        assert np.all(sampled_cat[:, 0] >= 0) and np.all(sampled_cat[:, 0] < 3)
        assert np.all(sampled_cat[:, 1] >= 0) and np.all(sampled_cat[:, 1] < 4)

    def test_sampler_invalid_arguments(self) -> None:
        r"""Verify that non-positive sample count or batch size raises ValueError."""
        diffusion = GaussianMultinomialDiffusion(num_timesteps=5)
        model = TabDDPMModel(num_numerical=2, cardinalities=[2], hidden_dim=16)
        sampler = TabDDPMSampler(model, diffusion, [2], 2, device="cpu")

        with pytest.raises(ValueError, match="num_samples must be positive"):
            sampler.sample(num_samples=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            sampler.sample(num_samples=10, batch_size=-1)


class TestTabDDPMDecoderIntegration:
    """Tests integration between TabDDPMSampler and TabularDecoder on real preprocessed data."""

    def test_nsl_kdd_end_to_end_decoding(self) -> None:
        r"""Verify end-to-end TabDDPM sampling and restoration to raw unscaled NSL-KDD DataFrame."""
        data = np.load("output/nsl_kdd/model_inputs/tabddpm_inputs.npz")
        cardinalities = data["categorical_cardinalities"].tolist()
        d_num = data["X_num"].shape[1]

        diffusion = GaussianMultinomialDiffusion(num_timesteps=5, schedule="linear")
        model = TabDDPMModel(
            num_numerical=d_num,
            cardinalities=cardinalities,
            hidden_dim=32,
            num_blocks=1,
        )

        sampler = TabDDPMSampler(
            model=model,
            diffusion=diffusion,
            cardinalities=cardinalities,
            num_numerical=d_num,
            device=resolve_device("auto"),
        )

        decoder = TabularDecoder.from_directory("output/nsl_kdd/model_inputs")
        df_synthetic = sampler.sample_to_dataframe(
            num_samples=15, decoder=decoder, batch_size=10, seed=42
        )

        assert isinstance(df_synthetic, pl.DataFrame)
        assert df_synthetic.shape == (15, 41)
        assert "is_anomaly" not in df_synthetic.columns
        assert "duration" in df_synthetic.columns
        assert "protocol_type" in df_synthetic.columns
        assert df_synthetic["protocol_type"].dtype == pl.String
        assert not df_synthetic.is_empty()
