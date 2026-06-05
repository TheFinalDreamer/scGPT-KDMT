"""PyTorch Dataset and DataLoader for scRNA-seq expression matrices."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SingleCellDataset(Dataset):
    """Dataset wrapping expression matrix and optional labels.

    Args:
        X: Expression matrix of shape (n_cells, n_genes).
        y: Optional integer labels of shape (n_cells,).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return {"x": self.X[idx]}
        return {"x": self.X[idx], "y": self.y[idx]}


def build_dataloader(
    dataset: SingleCellDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for a SingleCellDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )