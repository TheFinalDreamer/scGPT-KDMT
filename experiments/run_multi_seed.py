#!/usr/bin/env python
"""Run multi-seed experiment: trains the KDMT student across multiple random seeds.

Reports mean and standard deviation of classification metrics across seeds.

Usage:
    python experiments/run_multi_seed.py --config configs/multi_seed.yaml
"""

import argparse
import json
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scgpt_kdmt.utils.seed import set_seed
from src.scgpt_kdmt.utils.config import load_config
from src.scgpt_kdmt.data.dataset import SingleCellDataset, build_dataloader
from src.scgpt_kdmt.models.student import StudentModel
from src.scgpt_kdmt.training.trainer import train_one_epoch
from src.scgpt_kdmt.evaluation.metrics import evaluate


def run_one_seed(seed, config, device, train_loader, val_loader):
    """Train and evaluate the student model for a single random seed."""
    set_seed(seed)

    student = StudentModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
    ).to(device)

    teacher = StudentModel(
        input_dim=config["teacher_input_dim"],
        hidden_dim=config["teacher_hidden_dim"],
        num_heads=config["teacher_num_heads"],
        num_layers=config["teacher_num_layers"],
        num_classes=config["num_classes"],
    ).to(device)
    teacher.load_state_dict(torch.load(config["teacher_checkpoint"], map_location=device))
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    best_f1 = -1.0
    for epoch in range(config.get("num_epochs", 50)):
        train_one_epoch(
            model=student, loader=train_loader, optimizer=optimizer,
            device=device, teacher_model=teacher,
            use_kd_feat=config.get("use_kd_feat", True),
            use_kd_logit=config.get("use_kd_logit", True),
            use_recon=config.get("use_recon", True),
            use_mask=config.get("use_mask", True),
            use_contrast=config.get("use_contrast", True),
            lambda_feat=config.get("lambda_feat", 0.5),
            lambda_logit=config.get("lambda_logit", 0.5),
            lambda_recon=config.get("lambda_recon", 0.5),
            lambda_mask=config.get("lambda_mask", 0.5),
            lambda_contrast=config.get("lambda_contrast", 0.5),
            kd_temperature=config.get("kd_temperature", 2.0),
        )
        metrics = evaluate(student, val_loader, device)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]

    return best_f1


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed experiment.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_path:
        config["data_path"] = args.data_path
    if args.output_dir:
        config["output_dir"] = args.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["output_dir"], exist_ok=True)

    train_data = np.load(os.path.join(config["data_path"], "train.npz"))
    val_data = np.load(os.path.join(config["data_path"], "val.npz"))

    train_loader = build_dataloader(
        SingleCellDataset(train_data["X"], train_data["y"]),
        batch_size=config.get("batch_size", 64),
    )
    val_loader = build_dataloader(
        SingleCellDataset(val_data["X"], val_data["y"]),
        batch_size=config.get("batch_size", 64),
        shuffle=False,
    )

    seeds = config.get("seeds", [42, 2024, 2025])
    f1_per_seed = {}

    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        f1 = run_one_seed(seed, config, device, train_loader, val_loader)
        f1_per_seed[str(seed)] = f1
        print(f"[Seed {seed}] Best val Macro-F1: {f1:.4f}")

    f1s = list(f1_per_seed.values())
    results = {
        "experiment": "multi_seed",
        "seeds": seeds,
        "f1_per_seed": f1_per_seed,
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s)),
        "min_macro_f1": float(np.min(f1s)),
        "max_macro_f1": float(np.max(f1s)),
    }
    with open(os.path.join(config["output_dir"], "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nMulti-seed summary: mean={results['mean_macro_f1']:.4f} "
          f"std={results['std_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
