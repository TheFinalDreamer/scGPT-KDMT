#!/usr/bin/env python
"""Generate figures from experiment results (training curves, bar charts, etc.).

Usage:
    python scripts/generate_figures.py \\
        --config configs/figure_generation.yaml \\
        --results_dir outputs/ \\
        --output_dir figures/
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.scgpt_kdmt.utils.config import load_config


def plot_training_curves(metrics_history, output_path):
    """Plot training loss and validation metric curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(metrics_history) + 1)
    val_f1 = [m.get("macro_f1", 0) for m in metrics_history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_f1, "b-", linewidth=2, label="Val Macro-F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Validation Macro-F1 over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Training curve saved to {output_path}")


def plot_ablation_bar(ablation_results, output_path):
    """Plot ablation study results as a bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.get("name", "?") for r in ablation_results]
    f1s = [r.get("best_macro_f1", 0) for r in ablation_results]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), f1s, color="steelblue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Best Macro-F1")
    ax.set_title("Ablation Study: Macro-F1 by Configuration")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Ablation bar chart saved to {output_path}")


def plot_multi_seed_box(seed_results, output_path):
    """Plot multi-seed experiment results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(seed_results, list) and len(seed_results) > 0:
        f1s = []
        labels = []
        for entry in seed_results:
            if isinstance(entry, dict):
                labels.append(entry.get("experiment", "?"))
                f1_vals = entry.get("f1_per_seed", [])
                if f1_vals:
                    f1s.append(f1_vals)

        if f1s:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.boxplot(f1s, labels=labels)
            ax.set_ylabel("Macro-F1")
            ax.set_title("Multi-Seed Stability")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
            print(f"[INFO] Multi-seed box plot saved to {output_path}")
            return
    print("[WARN] No multi-seed data available for plotting.")


def main():
    parser = argparse.ArgumentParser(description="Generate figures from results.")
    parser.add_argument("--config", type=str, default="configs/figure_generation.yaml",
                        help="Path to figure generation config file.")
    parser.add_argument("--results_dir", type=str, default="outputs",
                        help="Root directory containing experiment outputs.")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Output directory for generated figures.")
    args = parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
    output_dir = args.output_dir or config.get("output_dir", "figures")
    results_dir = args.results_dir or config.get("results_dir", "outputs")

    os.makedirs(output_dir, exist_ok=True)

    # Look for metrics files and generate figures
    metrics_path = os.path.join(results_dir, "kdmt_student", "metrics.json")
    if os.path.exists(metrics_path):
        metrics = json.load(open(metrics_path))
        if "val_metrics_history" in metrics:
            plot_training_curves(
                metrics["val_metrics_history"],
                os.path.join(output_dir, "training_curve.png"),
            )

    ablation_path = os.path.join(results_dir, "ablation", "ablation_results.json")
    if os.path.exists(ablation_path):
        ablation = json.load(open(ablation_path))
        plot_ablation_bar(
            ablation,
            os.path.join(output_dir, "ablation_bar.png"),
        )

    summary_path = os.path.join(results_dir, "summaries", "results_summary.json")
    if os.path.exists(summary_path):
        summary = json.load(open(summary_path))
        plot_multi_seed_box(
            summary,
            os.path.join(output_dir, "multi_seed_box.png"),
        )

    print(f"[INFO] Figure generation complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
