"""Lightweight Transformer student model for cell type classification.

The student model combines a Transformer encoder, a classification head,
a reconstruction decoder, and a masked gene prediction head.

Model size: ~2.92M parameters (with input_dim=2000, hidden_dim=128,
num_layers=3, num_heads=4, num_classes=31).
"""

import torch.nn as nn

from .transformer import TransformerEncoder
from .classifier import Classifier
from .masked_prediction import MaskedGenePredictor


class StudentModel(nn.Module):
    """Multi-task student model for knowledge distillation.

    Produces four outputs from gene expression input:
    - embedding: Cell-level representation (mean-pooled across genes)
    - logits: Cell type class probabilities
    - recon: Reconstructed gene expression (global reconstruction)
    - masked_pred: Predicted expression for masked genes

    Args:
        input_dim: Number of input genes (e.g., 2000 HVGs).
        hidden_dim: Hidden dimension of the Transformer encoder.
        num_heads: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        num_classes: Number of cell type classes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        num_classes: int = 31,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        self.classifier = Classifier(
            input_dim=hidden_dim,
            num_classes=num_classes,
        )

        self.decoder = nn.Linear(hidden_dim, input_dim)

        self.mask_predictor = MaskedGenePredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Expression matrix of shape (batch_size, input_dim).

        Returns:
            dict with keys: embedding, logits, recon, masked_pred.
        """
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        recon = self.decoder(embedding)
        masked_pred = self.mask_predictor(embedding)

        return {
            "embedding": embedding,
            "logits": logits,
            "recon": recon,
            "masked_pred": masked_pred,
        }