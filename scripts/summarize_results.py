#!/usr/bin/env python
"""Summarize experiment results across multiple runs and seeds.

Reads metrics.json files from each experiment output directory and produces
a consolidated summary table (JSON and CSV).

Usage:
    python scripts/summarize_results.py \\
        --results_dir outputs/ \\
        --output outputs/summaries/results_summary
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def find_metrics_files(results_dir):
    """Find all metrics.json files under results_dir."""
    metrics_files = []
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f == "metrics.json":
                metrics_files.append(os.path.join(root, f))
    return sorted(metrics_files)


def load_metrics(path):
    """Load a metrics.json file."""
    with open(path, "r") as f:
        return json.load(f)


def summarize(metrics_files):
    """Consolidate metrics from multiple experiment directories."""
    summary = []
    for mf in metrics_files:
        exp_name = os.path.basename(os.path.dirname(mf))
        try:
            data = load_metrics(mf)
            entry = {"experiment": exp_name, "file": mf}
            if "best_macro_f1" in data:
                entry["best_macro_f1"] = data["best_macro_f1"]
                entry["best_epoch"] = data.get("best_epoch", None)
            elif "macro_f1" in data:
                entry["macro_f1"] = data["macro_f1"]
                entry["accuracy"] = data.get("accuracy", None)
            elif isinstance(data, list):
                entry["n_configs"] = len(data)
                f1s = [d.get("best_macro_f1", 0) for d in data if isinstance(d, dict)]
                if f1s:
                    entry["best_macro_f1"] = max(f1s)
                    entry["mean_macro_f1"] = float(np.mean(f1s))
                    entry["std_macro_f1"] = float(np.std(f1s))
            summary.append(entry)
        except Exception as e:
            summary.append({"experiment": os.path.basename(os.path.dirname(mf)),
                            "error": str(e)})
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results.")
    parser.add_argument("--results_dir", type=str, default="outputs",
                        help="Root directory containing experiment output directories.")
    parser.add_argument("--output", type=str, default="outputs/summaries/results_summary",
                        help="Output path prefix (without extension).")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    metrics_files = find_metrics_files(args.results_dir)
    if not metrics_files:
        print(f"[WARN] No metrics.json files found under {args.results_dir}")
        print("This is expected if experiments have not been run yet.")
        with open(args.output + ".json", "w") as f:
            json.dump({"status": "no_results", "message":
                       "No experiment results found. Run experiments first."}, f, indent=2)
        return

    print(f"[INFO] Found {len(metrics_files)} metrics file(s)")
    summary = summarize(metrics_files)

    with open(args.output + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary saved to {args.output}.json")

    for entry in summary:
        print(f"  {entry.get('experiment', 'unknown')}: {json.dumps({k: v for k, v in entry.items() if k not in ('experiment', 'file')})}")


if __name__ == "__main__":
    main()
