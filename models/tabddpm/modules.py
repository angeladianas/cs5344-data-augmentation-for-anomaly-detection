"""Denoiser neural network architecture for TabDDPM.

Implements the residual multilayer perceptron (MLP) denoiser conditioned on
diffusion timesteps via sinusoidal embeddings, with continuous feature projection
and categorical embedding layers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    r"""Sinusoidal positional encoding for diffusion timesteps.

    Given a 1D tensor of integer timesteps $t \in \{0, \dots, T-1\}$,
    computes frequency encodings:
    $$\text{PE}(t, 2i) = \sin(t / 10000^{2i/d}), \quad \text{PE}(t, 2i+1) = \cos(t / 10000^{2i/d})$$
    """

    def __init__(self, dim: int) -> None:
        r"""Initialize sinusoidal time embedding module.

        Args:
            dim: Output embedding dimension.
        """
        super().__init__()
        self.dim: int = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        r"""Compute sinusoidal embeddings for timesteps.

        Args:
            timesteps: 1D tensor of shape (B,) or (B, 1) containing integer timesteps.

        Returns:
            Tensor of shape (B, dim) with float32 sinusoidal embeddings.
        """
        timesteps = timesteps.reshape(-1)
        half_dim = self.dim // 2
        exponent = (
            -math.log(10000.0)
            * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
            / half_dim
        )
        freqs = torch.exp(exponent)
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.0)
        return emb


class ResidualBlock(nn.Module):
    r"""Residual MLP block conditioned on diffusion timestep embedding.

    Follows the standard diffusion block layout:
    $$h \leftarrow \text{LayerNorm}(x)$$
    $$h \leftarrow \text{Linear}_1(h) + W_t e(t)$$
    $$h \leftarrow \text{SiLU}(h)$$
    $$h \leftarrow \text{Dropout}(h)$$
    $$h \leftarrow \text{Linear}_2(h)$$
    $$y = x + \text{Dropout}(h)$$
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.10) -> None:
        r"""Initialize residual block.

        Args:
            hidden_dim: Dimensionality of latent representations.
            dropout: Dropout probability.
        """
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.linear1: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.act: nn.SiLU = nn.SiLU()
        self.linear2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        r"""Forward pass through residual block.

        Args:
            x: Input feature representation of shape (B, hidden_dim).
            time_emb: Project time embedding of shape (B, hidden_dim).

        Returns:
            Output feature representation of shape (B, hidden_dim).
        """
        residual = x
        h = self.norm(x)
        h = self.linear1(h) + self.time_proj(time_emb)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.dropout(h)
        return residual + h


class TabDDPMModel(nn.Module):
    r"""Deep MLP Denoiser for hybrid continuous-discrete tabular data.

    Accommodates mixed continuous (Gaussian) and categorical (Multinomial)
    tabular features. Continuous features are linearly projected, categorical
    features are embedded via discrete lookup tables, and timesteps are
    injected via sinusoidal positional encodings into residual MLP blocks.
    """

    def __init__(
        self,
        num_numerical: int,
        cardinalities: Sequence[int],
        embedding_dim: int = 16,
        hidden_dim: int = 512,
        num_blocks: int = 3,
        dropout: float = 0.10,
        time_dim: int = 128,
    ) -> None:
        r"""Initialize TabDDPMModel.

        Args:
            num_numerical: Number of continuous features (D_num >= 0).
            cardinalities: Sequence of discrete state counts for each categorical feature.
            embedding_dim: Embedding dimension for each categorical feature.
            hidden_dim: Latent representation width of MLP blocks.
            num_blocks: Number of residual blocks in backbone.
            dropout: Dropout probability.
            time_dim: Dimension of raw sinusoidal time encoding.
        """
        super().__init__()
        if num_numerical <= 0 and len(cardinalities) == 0:
            raise ValueError("TabDDPMModel requires at least one numerical or categorical feature.")

        self.num_numerical: int = num_numerical
        self.cardinalities: list[int] = list(cardinalities)
        self.num_categorical: int = len(self.cardinalities)
        self.hidden_dim: int = hidden_dim
        self.embedding_dim: int = embedding_dim

        # Timestep conditioning MLP: Sinusoidal -> Linear -> SiLU -> Linear (hidden_dim)
        self.time_mlp: nn.Sequential = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Categorical embedding layers
        if self.num_categorical > 0:
            self.cat_embeddings: nn.ModuleList = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=k, embedding_dim=embedding_dim)
                    for k in self.cardinalities
                ]
            )
            cat_total_dim = self.num_categorical * embedding_dim
        else:
            self.cat_embeddings = nn.ModuleList()
            cat_total_dim = 0

        # Input projection layer
        in_dim = self.num_numerical + cat_total_dim
        self.input_proj: nn.Linear = nn.Linear(in_dim, hidden_dim)
        self.input_act: nn.SiLU = nn.SiLU()

        # Residual MLP backbone
        self.blocks: nn.ModuleList = nn.ModuleList(
            [ResidualBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )

        # Final LayerNorm before output prediction heads
        self.final_norm: nn.LayerNorm = nn.LayerNorm(hidden_dim)

        # Output prediction heads
        if self.num_numerical > 0:
            self.head_numerical: nn.Linear | None = nn.Linear(hidden_dim, self.num_numerical)
        else:
            self.head_numerical = None

        if self.num_categorical > 0:
            self.head_categorical: nn.ModuleList = nn.ModuleList(
                [nn.Linear(hidden_dim, k) for k in self.cardinalities]
            )
        else:
            self.head_categorical = nn.ModuleList()

    def forward(
        self,
        x_num: torch.Tensor | None,
        x_cat: torch.Tensor | None,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
        r"""Compute predicted continuous noise and discrete category logits.

        Args:
            x_num: Continuous noisy features (B, D_num) or None.
            x_cat: Discrete noisy categorical indices (B, D_cat) with dtype int64 or None.
            timesteps: Timestep tensor of shape (B,) or (B, 1).

        Returns:
            Tuple of:
            - noise_pred: Predicted Gaussian noise (B, D_num) or None.
            - cat_logits: List of predicted logits (B, K_j) for each categorical feature.
        """
        # 1. Compute timestep conditioning embedding
        time_emb = self.time_mlp(timesteps)  # (B, hidden_dim)

        # 2. Gather input features
        features: list[torch.Tensor] = []
        if self.num_numerical > 0:
            if x_num is None:
                raise ValueError("Model requires x_num input but received None.")
            features.append(x_num.float())

        if self.num_categorical > 0:
            if x_cat is None:
                raise ValueError("Model requires x_cat input but received None.")
            cat_indices = x_cat.long()
            cat_reps: list[torch.Tensor] = []
            for j, emb_layer in enumerate(self.cat_embeddings):
                # Clamp indices defensively against adversarial out-of-bound inputs
                c_j = torch.clamp(cat_indices[:, j], min=0, max=self.cardinalities[j] - 1)
                cat_reps.append(emb_layer(c_j))
            features.append(torch.cat(cat_reps, dim=-1))

        x_in = torch.cat(features, dim=-1)
        h = self.input_act(self.input_proj(x_in))

        # 3. Pass through residual backbone
        for block in self.blocks:
            h = block(h, time_emb)

        h = self.final_norm(h)

        # 4. Predict outputs
        noise_pred = self.head_numerical(h) if self.head_numerical is not None else None
        cat_logits = [head(h) for head in self.head_categorical]

        return noise_pred, cat_logits
