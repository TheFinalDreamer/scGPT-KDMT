#!/usr/bin/env python
"""Train-only HVG experiment.

Selects highly variable genes using only the training set (not the full
dataset) to ensure no information leakage from validation/test sets into
gene selection. The experiment is otherwise identical to the main KDMT setup.

Usage:
    python experiments/run_train_only_hvg.py --config configs/train_only_hvg.yaml
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


def main():
    parser = argparse.ArgumentParser(
        description="Train-only HVG experiment.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to train-only HVG preprocessed data.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_path:
        config["data_path"] = args.data_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.teacher_checkpoint:
        config["teacher_checkpoint"] = args.teacher_checkpoint

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["output_dir"], exist_ok=True)

    print("[INFO] Train-only HVG experiment")
    print("[INFO] NOTE: Data must be preprocessed with train-only HVG selection.")
    print("[INFO] Run: python scripts/preprocess_data.py with a separate workflow")
    print("[INFO] that selects HVGs based on the training split only.")

    train_path = os.path.join(config["data_path"], "train.npz")
    val_path = os.path.join(config["data_path"], "val.npz")
    if not os.path.exists(train_path):
        print(f"[ERROR] Training data not found: {train_path}")
        print("Please preprocess data with train-only HVG selection first.")
        sys.exit(1)

    train_data = np.load(train_path)
    val_data = np.load(val_path)

    train_loader = build_dataloader(
        SingleCellDataset(train_data["X"], train_data["y"]),
        batch_size=config.get("batch_size", 64),
    )
    val_loader = build_dataloader(
        SingleCellDataset(val_data["X"], val_data["y"]),
        batch_size=config.get("batch_size", 64),
        shuffle=False,
    )

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
    best_epoch = -1

    for epoch in range(config.get("num_epochs", 50)):
        loss = train_one_epoch(
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
        print(f"[Epoch {epoch+1}] loss={loss:.4f} val_macro_f1={metrics['macro_f1']:.4f}")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch + 1
            torch.save(
                student.state_dict(),
                os.path.join(config["output_dir"], "train_only_hvg_best.pth"),
            )

    results = {
        "experiment": "train_only_hvg",
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
    }
    with open(os.path.join(config["output_dir"], "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Train-only HVG — Best val Macro-F1: {best_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
