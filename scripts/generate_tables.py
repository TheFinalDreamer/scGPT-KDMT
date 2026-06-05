#!/usr/bin/env python
"""Generate summary tables (LaTeX and CSV) from experiment results.

Usage:
    python scripts/generate_tables.py \\
        --summary outputs/summaries/results_summary.json \\
        --output outputs/tables
"""

import argparse
import json
import os
import sys


def load_summary(path):
    """Load a summary JSON file."""
    if not os.path.exists(path):
        print(f"[ERROR] Summary file not found: {path}")
        print("Run scripts/summarize_results.py first to generate the summary.")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def generate_latex_main_table(summary):
    """Generate main results table in LaTeX format."""
    rows = []
    for entry in summary:
        name = entry.get("experiment", "unknown")
        f1 = entry.get("best_macro_f1") or entry.get("macro_f1")
        acc = entry.get("accuracy")
        if f1 is not None:
            row = f"{name} & {f1:.4f}"
            if acc is not None:
                row += f" & {acc:.4f}"
            row += " \\\\"
            rows.append(row)

    if not rows:
        return None

    header = "\\begin{tabular}{lcc}\n\\toprule\nExperiment & Macro-F1 & Accuracy \\\\\n\\midrule"
    footer = "\\bottomrule\n\\end{tabular}"
    return header + "\n" + "\n".join(rows) + "\n" + footer


def generate_ablation_latex_table(summary):
    """Generate ablation study table in LaTeX format."""
    for entry in summary:
        if isinstance(entry, dict) and "n_configs" in entry:
            return (
                "\\begin{tabular}{lc}\n\\toprule\n"
                "Configuration & Macro-F1 \\\\\n\\midrule\n"
                f"Best & {entry.get('best_macro_f1', 0):.4f} \\\\\n"
                f"Mean & {entry.get('mean_macro_f1', 0):.4f} \\\\\n"
                f"Std & {entry.get('std_macro_f1', 0):.4f} \\\\\n"
                "\\bottomrule\n\\end{tabular}"
            )
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate summary tables.")
    parser.add_argument("--summary", type=str, required=True,
                        help="Path to results summary JSON file.")
    parser.add_argument("--output", type=str, default="outputs/tables",
                        help="Output directory for generated tables.")
    args = parser.parse_args()

    summary = load_summary(args.summary)
    os.makedirs(args.output, exist_ok=True)

    # Save as CSV
    import csv
    csv_path = os.path.join(args.output, "results_summary.csv")
    if summary and isinstance(summary, list) and len(summary) > 0:
        keys = ["experiment"] + [k for k in summary[0].keys()
                                  if k not in ("experiment", "file")]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary)
        print(f"[INFO] CSV table saved to {csv_path}")

    # Generate LaTeX tables
    latex_main = generate_latex_main_table(summary)
    if latex_main:
        latex_path = os.path.join(args.output, "main_results.tex")
        with open(latex_path, "w") as f:
            f.write(latex_main)
        print(f"[INFO] LaTeX table saved to {latex_path}")

    if not summary or (isinstance(summary, list) and len(summary) == 0):
        print("[WARN] No results to tabulate. Run experiments first, then summarize_results.py.")


if __name__ == "__main__":
    main()
