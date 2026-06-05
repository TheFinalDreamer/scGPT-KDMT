"""Evaluation metrics for cell type classification and embedding quality."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Compute standard classification metrics from logits.

    Args:
        logits: Predicted logits of shape (n_samples, n_classes).
        labels: Ground-truth labels of shape (n_samples,).

    Returns:
        Dictionary with accuracy, macro_f1, macro_precision, macro_recall.
    """
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }


def compute_confusion_matrix(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute confusion matrix from logits."""
    preds = np.argmax(logits, axis=1)
    return confusion_matrix(labels, preds)


def clustering_metrics(
    embeddings: np.ndarray, labels: np.ndarray, n_clusters: int = None
) -> dict:
    """Compute clustering quality metrics (ARI, NMI).

    Uses KMeans clustering on the embeddings and compares with ground-truth labels.
    """
    from sklearn.cluster import KMeans

    if n_clusters is None:
        n_clusters = len(np.unique(labels))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    pred = kmeans.fit_predict(embeddings)

    return {
        "ARI": float(adjusted_rand_score(labels, pred)),
        "NMI": float(normalized_mutual_info_score(labels, pred)),
    }


def evaluate(model, loader, device) -> dict:
    """Evaluate a model on a dataloader.

    Args:
        model: PyTorch model.
        loader: DataLoader yielding {"x": ..., "y": ...}.
        device: torch device.

    Returns:
        Classification metrics dictionary.
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            outputs = model(x)
            all_logits.append(outputs["logits"].cpu().numpy())
            all_labels.append(y.cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    return classification_metrics(logits, labels)