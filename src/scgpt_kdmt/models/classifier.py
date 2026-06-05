"""Cell type classification head."""

import torch.nn as nn


class Classifier(nn.Module):
    """MLP classification head.

    Maps cell embeddings to class logits via a two-layer MLP.

    Args:
        input_dim: Embedding dimension.
        num_classes: Number of cell types.
        hidden_dim: Hidden layer dimension (default: 256).
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)