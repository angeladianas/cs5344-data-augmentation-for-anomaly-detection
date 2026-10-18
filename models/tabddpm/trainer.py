"""Training engine and loss computation for TabDDPM.

Trains the residual denoiser network on hybrid tabular data using Gaussian
Mean Squared Error for continuous features and Multinomial Cross-Entropy
for discrete categorical features.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.tabddpm.diffusion import GaussianMultinomialDiffusion
from models.tabddpm.modules import TabDDPMModel
from models.tabddpm.sampler import resolve_device


class TabDDPMTrainer:
    r"""Training coordinator and loss optimizer for TabDDPM.

    Optimizes the combined hybrid loss:
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{num}} + \lambda_{\text{cat}} \mathcal{L}_{\text{cat}}$$
    """

    def __init__(
        self,
        model: TabDDPMModel,
        diffusion: GaussianMultinomialDiffusion,
        cardinalities: Sequence[int],
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 200,
        lambda_cat: float = 1.0,
        clip_grad_norm: float = 1.0,
        patience: int = 30,
        device: torch.device | str | None = None,
    ) -> None:
        r"""Initialize trainer.

        Args:
            model: TabDDPM neural denoiser instance.
            diffusion: GaussianMultinomialDiffusion scheduling engine.
            cardinalities: Discrete cardinalities for categorical features.
            learning_rate: Initial AdamW learning rate (default: 2e-4).
            weight_decay: AdamW weight decay regularization (default: 1e-4).
            batch_size: Mini-batch size.
            epochs: Maximum training epochs.
            lambda_cat: Weight multiplier for categorical cross-entropy loss.
            clip_grad_norm: Maximum gradient norm for clipping.
            patience: Early stopping patience epochs based on validation loss.
            device: Target device (cuda, mps, cpu, or auto).
        """
        self.device: torch.device = resolve_device(device)
        self.model: TabDDPMModel = model.to(self.device)
        self.diffusion: GaussianMultinomialDiffusion = diffusion.to(self.device)
        self.cardinalities: list[int] = list(cardinalities)

        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.lambda_cat: float = lambda_cat
        self.clip_grad_norm: float = clip_grad_norm
        self.patience: int = patience

    def _compute_loss(
        self,
        x_num_batch: torch.Tensor | None,
        x_cat_batch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Compute hybrid diffusion loss for a mini-batch.

        Returns:
            Tuple of (total_loss, loss_num, loss_cat).
        """
        batch_size = (
            x_num_batch.shape[0] if x_num_batch is not None else x_cat_batch.shape[0]  # type: ignore[union-attr]
        )

        # Sample uniform discrete timesteps t in {0, ..., T-1}
        t = torch.randint(
            0, self.diffusion.num_timesteps, (batch_size,), device=self.device, dtype=torch.long
        )

        # 1. Continuous forward diffusion and loss
        if x_num_batch is not None and self.model.num_numerical > 0:
            x_t_num, target_noise = self.diffusion.q_sample_numerical(x_num_batch, t)
        else:
            x_t_num = None
            target_noise = None

        # 2. Categorical forward diffusion and loss
        if x_cat_batch is not None and len(self.cardinalities) > 0:
            x_t_cat = self.diffusion.q_sample_categorical(x_cat_batch, t, self.cardinalities)
        else:
            x_t_cat = None

        # 3. Model forward pass
        noise_pred, cat_logits = self.model(x_t_num, x_t_cat, t)

        # 4. Numerical loss: MSE between true and predicted noise
        if target_noise is not None and noise_pred is not None:
            loss_num = F.mse_loss(noise_pred, target_noise)
        else:
            loss_num = torch.tensor(0.0, device=self.device)

        # 5. Categorical loss: Mean Cross-Entropy over all discrete features
        if x_cat_batch is not None and len(self.cardinalities) > 0:
            cat_losses = [
                F.cross_entropy(cat_logits[j], x_cat_batch[:, j])
                for j in range(len(self.cardinalities))
            ]
            loss_cat = torch.stack(cat_losses).mean()
        else:
            loss_cat = torch.tensor(0.0, device=self.device)

        total_loss = loss_num + self.lambda_cat * loss_cat
        return total_loss, loss_num, loss_cat

    def fit(
        self,
        x_num: np.ndarray | torch.Tensor | None,
        x_cat: np.ndarray | torch.Tensor | None,
        val_ratio: float = 0.1,
        seed: int = 42,
        checkpoint_path: str | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        r"""Train TabDDPM denoiser on tabular dataset.

        Args:
            x_num: Continuous feature array of shape (N, D_num) or None.
            x_cat: Discrete categorical array of shape (N, D_cat) or None.
            val_ratio: Proportion of data held out for validation early stopping.
            seed: Random seed for dataset splitting.
            checkpoint_path: Optional path to save best model checkpoint.
            verbose: Whether to print epoch progress.

        Returns:
            Dictionary containing loss histories: 'train_loss', 'val_loss', 'loss_num', 'loss_cat'.
        """
        # Prepare PyTorch tensors
        tensors: list[torch.Tensor] = []
        has_num = x_num is not None and (
            x_num.shape[1] > 0 if hasattr(x_num, "shape") else len(x_num) > 0
        )
        has_cat = x_cat is not None and (
            x_cat.shape[1] > 0 if hasattr(x_cat, "shape") else len(x_cat) > 0
        )

        if not has_num and not has_cat:
            raise ValueError("fit() requires at least one continuous or categorical feature array.")

        n_samples = (
            len(x_num) if has_num and x_num is not None else len(x_cat)  # type: ignore[arg-type]
        )

        if has_num and x_num is not None:
            t_num = torch.as_tensor(x_num, dtype=torch.float32)
            tensors.append(t_num)
        if has_cat and x_cat is not None:
            t_cat = torch.as_tensor(x_cat, dtype=torch.long)
            tensors.append(t_cat)

        dataset = TensorDataset(*tensors)

        # Train / validation split
        generator = torch.Generator().manual_seed(seed)
        if 0.0 < val_ratio < 1.0:
            val_size = int(n_samples * val_ratio)
            train_size = n_samples - val_size
            train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
        else:
            train_set = dataset
            val_set = None

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=len(train_set) > self.batch_size,
        )
        val_loader = (
            DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
            if val_set is not None
            else None
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6
        )

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "loss_num": [],
            "loss_cat": [],
        }

        for epoch in range(1, self.epochs + 1):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            epoch_loss_num = 0.0
            epoch_loss_cat = 0.0
            num_batches = 0

            for batch in train_loader:
                idx = 0
                b_num = batch[idx].to(self.device) if has_num else None
                if has_num:
                    idx += 1
                b_cat = batch[idx].to(self.device) if has_cat else None

                optimizer.zero_grad()
                loss, l_num, l_cat = self._compute_loss(b_num, b_cat)
                loss.backward()

                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.clip_grad_norm
                    )

                optimizer.step()

                epoch_loss += loss.item()
                epoch_loss_num += l_num.item()
                epoch_loss_cat += l_cat.item()
                num_batches += 1

            train_loss = epoch_loss / max(num_batches, 1)
            train_l_num = epoch_loss_num / max(num_batches, 1)
            train_l_cat = epoch_loss_cat / max(num_batches, 1)

            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_epoch_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        idx = 0
                        b_num = batch[idx].to(self.device) if has_num else None
                        if has_num:
                            idx += 1
                        b_cat = batch[idx].to(self.device) if has_cat else None

                        v_loss, _, _ = self._compute_loss(b_num, b_cat)
                        val_epoch_loss += v_loss.item()
                        val_batches += 1
                val_loss = val_epoch_loss / max(val_batches, 1)
            else:
                val_loss = train_loss

            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["loss_num"].append(train_l_num)
            history["loss_cat"].append(train_l_cat)

            # Early stopping check
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                if checkpoint_path is not None:
                    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "val_loss": val_loss,
                            "cardinalities": self.cardinalities,
                            "num_numerical": self.model.num_numerical,
                        },
                        checkpoint_path,
                    )
            else:
                patience_counter += 1

            if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == self.epochs):
                print(
                    f"Epoch {epoch:03d}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} (num: {train_l_num:.4f}, cat: {train_l_cat:.4f}) | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Patience: {patience_counter}/{self.patience}"
                )

            if patience_counter >= self.patience:
                if verbose:
                    print(
                        f"Early stopping triggered at epoch {epoch}. Restoring best weights with val_loss={best_val_loss:.4f}."
                    )
                break

        # Restore best weights
        self.model.load_state_dict(best_weights)
        return history
