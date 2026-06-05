#!/usr/bin/env python
"""Fine-tune the teacher model for cell type classification.

Usage:
    python experiments/train_teacher_scgpt.py --config configs/teacher_scgpt.yaml
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
from src.scgpt_kdmt.evaluation.metrics import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune teacher for cell type classification.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--data_path", type=str, default=None, help="Override data path.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_path:
        config["data_path"] = args.data_path
    if args.output_dir:
        config["output_dir"] = args.output_dir

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

    from src.scgpt_kdmt.models.student import StudentModel
    teacher = StudentModel(
        input_dim=config["input_dim"],
        hidden_dim=config["teacher_hidden_dim"],
        num_heads=config["teacher_num_heads"],
        num_layers=config["teacher_num_layers"],
        num_classes=config["num_classes"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        teacher.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    ce_loss = torch.nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_epoch = -1

    for epoch in range(config.get("num_epochs", 10)):
        teacher.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()
            outputs = teacher(x)
            loss = ce_loss(outputs["logits"], y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(teacher, val_loader, device)
        print(f"[Epoch {epoch+1}] loss={total_loss/len(train_loader):.4f} "
              f"val_macro_f1={metrics['macro_f1']:.4f}")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch + 1
            torch.save(
                teacher.state_dict(),
                os.path.join(config["output_dir"], "teacher_best.pth"),
            )

    print(f"Best val Macro-F1: {best_f1:.4f} at epoch {best_epoch}")

    with open(os.path.join(config["output_dir"], "teacher_metrics.json"), "w") as f:
        json.dump({"best_macro_f1": best_f1, "best_epoch": best_epoch}, f, indent=2)


if __name__ == "__main__":
    main()