#!/usr/bin/env python
"""Evaluate a trained model on a test set.

Usage:
    python scripts/evaluate.py \\
        --config configs/student_kdmt.yaml \\
        --checkpoint /path/to/checkpoint.pth \\
        --data /path/to/test_data
"""

import argparse
import json
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scgpt_kdmt.utils.config import load_config
from src.scgpt_kdmt.utils.seed import set_seed
from src.scgpt_kdmt.data.dataset import SingleCellDataset, build_dataloader
from src.scgpt_kdmt.models.student import StudentModel
from src.scgpt_kdmt.evaluation.metrics import (
    evaluate,
    compute_confusion_matrix,
    clustering_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint .pth file.")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to directory with test.npz.")
    parser.add_argument("--output", type=str, default="outputs/results",
                        help="Output directory for evaluation results.")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save per-cell predictions to CSV.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    test_data = np.load(os.path.join(args.data, "test.npz"))
    test_loader = build_dataloader(
        SingleCellDataset(test_data["X"], test_data["y"]),
        batch_size=config.get("batch_size", 64),
        shuffle=False,
    )

    model = StudentModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    metrics = evaluate(model, test_loader, device)
    print("Classification metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()