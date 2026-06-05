"""Masked gene prediction head."""

import torch.nn as nn


class MaskedGenePredictor(nn.Module):
    """Predict original expression values from masked gene embeddings.

    Args:
        input_dim: Number of input genes.
        hidden_dim: Embedding dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.predictor = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.predictor(x)