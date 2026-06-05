"""Data splitting utilities."""

import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(
    X: np.ndarray, y: np.ndarray, val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42
):
    """Split data into train/validation/test sets with stratification.

    Args:
        X: Expression matrix.
        y: Integer labels.
        val_size: Validation set proportion.
        test_size: Test set proportion.
        random_state: Random seed.

    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test)).
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_frac,
        stratify=y_trainval,
        random_state=random_state,
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)