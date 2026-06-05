#!/usr/bin/env python
"""Run ablation experiments by toggling individual loss components.

Usage:
    python experiments/run_ablation.py --config configs/ablation.yaml
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


ABLATION_CONFIGS = {
    "Full_MultiTask": {
        "use_kd_feat": True, "use_kd_logit": True,
        "use_recon": True, "use_mask": True, "use_contrast": True,
    },
    "w/o_Reconstruction": {
        "use_kd_feat": True, "use_kd_logit": True,
        "use_recon": False, "use_mask": True, "use_contrast": True,
    },
    "w/o_Contrastive": {
        "use_kd_feat": True, "use_kd_logit": True,
        "use_recon": True, "use_mask": True, "use_contrast": False,
    },
    "KD_Only": {
        "use_kd_feat": True, "use_kd_logit": True,
        "use_recon": False, "use_mask": False, "use_contrast": False,
    },
    "w/o_FeatureKD": {
        "use_kd_feat": False, "use_kd_logit": True,
        "use_recon": True, "use_mask": True, "use_contrast": True,
    },
    "w/o_MaskPred": {
        "use_kd_feat": True, "use_kd_logit": True,
        "use_recon": True, "use_mask": False, "use_contrast": True,
    },
    "w/o_LogitsKD": {
        "use_kd_feat": True, "use_kd_logit": False,
        "use_recon": True, "use_mask": True, "use_contrast": True,
    },
    "w/o_KD_Both": {
        "use_kd_feat": False, "use_kd_logit": False,
        "use_recon": True, "use_mask": True, "use_contrast": True,
    },
    "Student_CE_Only": {
        "use_kd_feat": False, "use_kd_logit": False,
        "use_recon": False, "use_mask": False, "use_contrast": False,
    },
}


def run_one_ablation(name, switches, config, device, train_loader, val_loader):
    set_seed(config.get("seed", 42))
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
        teacher.load_state_dict(
            torch.load(config["teacher_checkpoint"], map_location=device))
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
            kd_temperature=config.get("kd_temperature", 2.0),
            **switches,
        )
        metrics = evaluate(student, val_loader, device)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]

    return {"name": name, "best_macro_f1": best_f1, **switches}


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--ablations", type=str, nargs="*",
                        help="Specific ablation names to run (default: all).")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_path:
        config["data_path"] = args.data_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.teacher_checkpoint:
        config["teacher_checkpoint"] = args.teacher_checkpoint

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

    names = args.ablations if args.ablations else list(ABLATION_CONFIGS.keys())
    results = []
    for name in names:
        print(f"\n{'='*60}\nRunning ablation: {name}\n{'='*60}")
        switches = ABLATION_CONFIGS[name]
        result = run_one_ablation(name, switches, config, device, train_loader, val_loader)
        results.append(result)
        print(f"[{name}] Best val Macro-F1: {result['best_macro_f1']:.4f}")

    out_path = os.path.join(config["output_dir"], "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {out_path}")


if __name__ == "__main__":
    main()