"""Reproducibility utilities."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set all random seeds for deterministic training.

    Args:
        seed: Integer seed for Python, NumPy, and PyTorch RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False