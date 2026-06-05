#!/usr/bin/env python
"""Gene-number fair comparison experiment.

Compares model performance at different gene counts (500, 1000, 2000, 4000, 8000)
to ensure fair comparison between teacher and student models at matched input
dimensions. Each gene count runs a separate preprocessing+training pipeline.

Usage:
    python experiments/run_gene_number_fair_comparison.py \\
        --config configs/gene_number_fair_comparison.yaml
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


def run_one_gene_count(n_genes, config, device, train_loader, val_loader):
    """Train and evaluate the KDMT student at a specific gene count."""
    set_seed(config.get("seed", 42))

    student = StudentModel(
        input_dim=n_genes,
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
    ).to(device)

    teacher = StudentModel(
        input_dim=n_genes,
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
    ).to(device)
    teacher.load_state_dict(torch.load(config["teacher_checkpoint"],
                           map_location=device), strict=False)
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
    parser = argparse.ArgumentParser(
        description="Gene-number fair comparison experiment.")
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

    print("[INFO] Gene-number fair comparison experiment")
    print("[INFO] NOTE: Each gene count requires preprocessed data at that")
    print("[INFO] dimension. Place preprocessed files as:")
    print("[INFO]   {data_path}/g{n_genes}/train.npz, val.npz, test.npz")

    gene_counts = config.get("gene_counts", [500, 1000, 2000, 4000, 8000])
    results = {}

    for n_genes in gene_counts:
        sub_data_path = os.path.join(config["data_path"], f"g{n_genes}")
        train_file = os.path.join(sub_data_path, "train.npz")
        val_file = os.path.join(sub_data_path, "val.npz")

        if not os.path.exists(train_file):
            print(f"[WARN] Data for {n_genes} genes not found at {sub_data_path}. Skipping.")
            results[str(n_genes)] = {"status": "data_not_found"}
            continue

        train_data = np.load(train_file)
        val_data = np.load(val_file)
        actual_genes = train_data["X"].shape[1]
        print(f"\n{'='*60}\nGene count: {n_genes} (actual dim: {actual_genes})\n{'='*60}")

        train_loader = build_dataloader(
            SingleCellDataset(train_data["X"], train_data["y"]),
            batch_size=config.get("batch_size", 64),
        )
        val_loader = build_dataloader(
            SingleCellDataset(val_data["X"], val_data["y"]),
            batch_size=config.get("batch_size", 64),
            shuffle=False,
        )

        f1 = run_one_gene_count(actual_genes, config, device, train_loader, val_loader)
        results[str(n_genes)] = {"input_dim": actual_genes, "best_macro_f1": f1}
        print(f"[{n_genes} genes] Best val Macro-F1: {f1:.4f}")

    with open(os.path.join(config["output_dir"], "metrics.json"), "w") as f:
        json.dump({"experiment": "gene_number_fair_comparison",
                   "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
