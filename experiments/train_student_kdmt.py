#!/usr/bin/env python
"""Train the scGPT-KDMT student model with multi-task knowledge distillation.

Usage:
    python experiments/train_student_kdmt.py --config configs/student_kdmt.yaml
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
    parser = argparse.ArgumentParser(description="Train scGPT-KDMT student with multi-task KD.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--data_path", type=str, default=None, help="Override data path.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--teacher_checkpoint", type=str, default=None,
                        help="Override teacher checkpoint path.")
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

    student = StudentModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
    ).to(device)

    teacher = None
    if config.get("use_kd", True):
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
    train_losses = []
    val_metrics_history = []

    for epoch in range(config.get("num_epochs", 50)):
        loss = train_one_epoch(
            model=student, loader=train_loader, optimizer=optimizer, device=device,
            teacher_model=teacher,
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
        train_losses.append(loss)
        val_metrics_history.append(metrics)

        print(f"[Epoch {epoch+1}] loss={loss:.4f} val_macro_f1={metrics['macro_f1']:.4f}")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch + 1
            torch.save(
                student.state_dict(),
                os.path.join(config["output_dir"], "student_best.pth"),
            )

    results = {
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "train_losses": train_losses,
        "val_metrics_history": val_metrics_history,
    }
    with open(os.path.join(config["output_dir"], "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Best val Macro-F1: {best_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()