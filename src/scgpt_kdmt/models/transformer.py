"""Lightweight Transformer encoder for scRNA-seq data."""

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """A compact Transformer encoder for gene expression input.

    Projects scalar gene expression values through a learnable linear embedding
    and processes them via a stack of Transformer encoder layers. Gene-index
    embeddings are used to encode individual gene identity.

    Args:
        input_dim: Number of input genes.
        hidden_dim: Hidden dimension of the Transformer.
        num_heads: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Gene expression matrix of shape (batch_size, input_dim).

        Returns:
            Cell-level embedding of shape (batch_size, hidden_dim).
        """
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.encoder(x)
        return x.squeeze(1)