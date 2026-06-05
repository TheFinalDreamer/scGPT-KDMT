"""Masked gene prediction data utility."""

import torch


def mask_input(x: torch.Tensor, mask_ratio: float = 0.15, mask_token: float = 0.0):
    """Randomly mask a fraction of genes with a mask token.

    Args:
        x: Input tensor of shape (batch_size, n_genes).
        mask_ratio: Fraction of genes to mask (default: 0.15).
        mask_token: Value to use for masked positions (default: 0.0).

    Returns:
        Tuple of (masked_x, mask) where mask is a boolean tensor.
    """
    mask = torch.rand_like(x) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = mask_token
    return x_masked, mask