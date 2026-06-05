#!/usr/bin/env python
"""Preprocess scRNA-seq data from h5ad to train/val/test npz files.

Usage:
    python scripts/preprocess_data.py \\
        --config configs/preprocess_pbmc10x.yaml \\
        --input /path/to/input.h5ad \\
        --output outputs/processed/pbmc10x
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scgpt_kdmt.utils.config import load_config
from src.scgpt_kdmt.data.split import split_dataset
from src.scgpt_kdmt.utils.io import save_pickle

import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.preprocessing import LabelEncoder


# =========================
# Utility functions
# =========================
def _to_dense_float32(X):
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    return X


def _safe_replace_nonfinite(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32)


def _safe_standardize(X):
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    std[std == 0] = 1.0
    X = (X - mean) / std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32)


def _find_batch_key(adata, preferred_keys=None):
    if preferred_keys is None:
        preferred_keys = [
            "batch", "sample", "orig.ident", "donor",
            "patient", "sample_id", "batch_id",
        ]
    for key in preferred_keys:
        if key in adata.obs.columns:
            n_unique = adata.obs[key].astype(str).nunique()
            if n_unique > 1:
                return key
    return None


def _ensure_raw_counts(adata):
    X = adata.X
    if sparse.issparse(X):
        sample = X[:1000, :1000].toarray()
    else:
        sample = np.asarray(X[:1000, :1000])
    sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
    x_min = float(sample.min())
    x_max = float(sample.max())
    is_integer_like = np.allclose(sample, np.round(sample), atol=1e-6)
    is_count_like = (x_min >= 0) and is_integer_like and (x_max >= 5)
    return is_count_like, x_min, x_max


def _run_qc(adata, min_genes, max_genes, min_cells, min_counts, max_counts, max_mt_pct):
    print("[INFO] Running QC ...")
    var_names_upper = adata.var_names.astype(str).str.upper()
    adata.var["mt"] = var_names_upper.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    print(f"[INFO] Before QC: cells={adata.n_obs}, genes={adata.n_vars}")
    sc.pp.filter_genes(adata, min_cells=min_cells)
    cell_mask = (
        (adata.obs["n_genes_by_counts"] >= min_genes) &
        (adata.obs["n_genes_by_counts"] <= max_genes) &
        (adata.obs["total_counts"] >= min_counts) &
        (adata.obs["total_counts"] <= max_counts) &
        (adata.obs["pct_counts_mt"] < max_mt_pct)
    )
    adata = adata[cell_mask].copy()
    print(f"[INFO] After QC: cells={adata.n_obs}, genes={adata.n_vars}")
    return adata


def _run_scrublet(adata, batch_key, expected_doublet_rate, sim_doublet_ratio, n_prin_comps, filter_doublets):
    print("[INFO] Running doublet detection with Scrublet ...")
    try:
        import scrublet as scr
    except ImportError:
        print("[WARN] Scrublet not installed. Skipping doublet detection.")
        return adata

    adata = adata.copy()
    adata.obs["doublet_score"] = 0.0
    adata.obs["predicted_doublet"] = False

    def _run_on_subset(sub_adata):
        X = sub_adata.X
        if sparse.issparse(X):
            X_input = X.tocsc()
        else:
            X_input = np.asarray(X)
        max_valid_pcs = max(2, min(sub_adata.n_obs - 1, sub_adata.n_vars - 1, n_prin_comps))
        scrub = scr.Scrublet(X_input, expected_doublet_rate=expected_doublet_rate,
                             sim_doublet_ratio=sim_doublet_ratio)
        scores, preds = scrub.scrub_doublets(
            min_counts=2, min_cells=3, min_gene_variability_pctl=85, n_prin_comps=max_valid_pcs)
        return scores.astype(float), preds.astype(bool)

    if batch_key is not None and batch_key in adata.obs.columns:
        print(f"[INFO] Doublet detection by batch: {batch_key}")
        for b in adata.obs[batch_key].astype(str).unique():
            idx = np.where(adata.obs[batch_key].astype(str).values == b)[0]
            sub_adata = adata[idx].copy()
            if sub_adata.n_obs < 50:
                continue
            try:
                scores, preds = _run_on_subset(sub_adata)
                adata.obs.loc[sub_adata.obs_names, "doublet_score"] = scores
                adata.obs.loc[sub_adata.obs_names, "predicted_doublet"] = preds
                print(f"[INFO] Batch={b} | cells={sub_adata.n_obs} | doublets={int(preds.sum())}")
            except Exception as e:
                print(f"[WARN] Scrublet failed on batch '{b}': {e}")
    else:
        if adata.n_obs >= 50:
            scores, preds = _run_on_subset(adata)
            adata.obs["doublet_score"] = scores
            adata.obs["predicted_doublet"] = preds
            print(f"[INFO] Global doublet detection | cells={adata.n_obs} | doublets={int(preds.sum())}")

    before_n = int(adata.n_obs)
    if filter_doublets and "predicted_doublet" in adata.obs.columns:
        adata = adata[~adata.obs["predicted_doublet"]].copy()
    after_n = int(adata.n_obs)
    print(f"[INFO] Doublet filtering done: before={before_n}, after={after_n}")
    return adata


def preprocess_h5ad(
    input_path, output_dir, label_key="celltype.l2",
    min_genes=300, max_genes=6000, min_cells=3, min_counts=1000, max_counts=20000, max_mt_pct=10.0,
    run_scrublet=True, filter_doublets=True, expected_doublet_rate=0.06,
    sim_doublet_ratio=2.0, scrublet_n_prin_comps=30,
    n_top_genes=2000, batch_key=None, run_harmony=True, random_state=42,
):
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Loading h5ad ...")
    adata = sc.read_h5ad(input_path)
    print(f"[INFO] Raw adata shape: {adata.shape}")

    if label_key not in adata.obs.columns:
        raise ValueError(f"label_key '{label_key}' not found in adata.obs")

    detected_batch_key = batch_key or _find_batch_key(adata)
    print(f"[INFO] Batch key: {detected_batch_key}")

    # QC
    adata = _run_qc(adata, min_genes, max_genes, min_cells, min_counts, max_counts, max_mt_pct)
    if adata.n_obs == 0:
        raise ValueError("No cells remaining after QC.")

    # Doublet detection
    is_count_like, _, _ = _ensure_raw_counts(adata)
    if run_scrublet and is_count_like:
        try:
            adata = _run_scrublet(adata, detected_batch_key, expected_doublet_rate,
                                  sim_doublet_ratio, scrublet_n_prin_comps, filter_doublets)
        except Exception as e:
            print(f"[WARN] Scrublet failed: {e}")
    elif run_scrublet and not is_count_like:
        print("[WARN] Matrix does not look like raw counts, skipping Scrublet.")

    if adata.n_obs == 0:
        raise ValueError("No cells remaining after doublet filtering.")

    # HVG selection
    print("[INFO] Selecting HVGs ...")
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3",
                                    batch_key=detected_batch_key if detected_batch_key in (adata.obs.columns if detected_batch_key else []) else None)
    except Exception:
        print("[WARN] seurat_v3 HVG failed, falling back to cell_ranger.")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="cell_ranger")

    n_hvg = int(adata.var["highly_variable"].sum())
    print(f"[INFO] Number of HVGs: {n_hvg}")
    adata = adata[:, adata.var["highly_variable"]].copy()

    np.save(os.path.join(output_dir, "hvg_gene_names.npy"), adata.var_names.to_numpy())

    # Normalize + log1p
    is_count_like, _, _ = _ensure_raw_counts(adata)
    if is_count_like:
        print("[INFO] Applying normalize_total + log1p ...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Convert to features
    print("[INFO] Converting expression matrix to model features ...")
    X = _to_dense_float32(adata.X)
    X = _safe_replace_nonfinite(X)
    X = _safe_standardize(X)
    print(f"[INFO] Final expression feature shape: {X.shape}")

    # Encode labels
    print("[INFO] Encoding labels ...")
    labels = adata.obs[label_key].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(labels).astype(np.int64)
    print(f"[INFO] Num classes: {len(le.classes_)}")
    save_pickle(le, os.path.join(output_dir, "label_encoder.pkl"))

    # Split
    print("[INFO] Splitting dataset ...")
    train, val, test = split_dataset(X, y, random_state=random_state)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train, val, test

    # Save
    print("[INFO] Saving processed files ...")
    np.savez(os.path.join(output_dir, "train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(output_dir, "val.npz"), X=X_val, y=y_val)
    np.savez(os.path.join(output_dir, "test.npz"), X=X_test, y=y_test)

    config_dict = {
        "input_path": input_path, "output_dir": output_dir, "label_key": label_key,
        "min_genes": min_genes, "max_genes": max_genes, "min_counts": min_counts,
        "max_counts": max_counts, "max_mt_pct": max_mt_pct,
        "n_top_genes": n_top_genes, "batch_key": detected_batch_key,
        "random_state": random_state, "num_classes": int(len(le.classes_)),
        "num_features": int(X.shape[1]), "num_cells": int(adata.n_obs),
    }
    with open(os.path.join(output_dir, "preprocess_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print("[INFO] Preprocessing complete.")
    print(f"[INFO] Train: X={X_train.shape}, y={y_train.shape}")
    print(f"[INFO] Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"[INFO] Test:  X={X_test.shape}, y={y_test.shape}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess scRNA-seq h5ad to npz files.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--input", type=str, default=None, help="Override input h5ad path.")
    parser.add_argument("--output", type=str, default=None, help="Override output directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = args.input or config["input_path"]
    output_dir = args.output or config["output_dir"]

    preprocess_h5ad(
        input_path=input_path,
        output_dir=output_dir,
        label_key=config.get("label_key", "celltype.l2"),
        min_genes=config.get("min_genes", 300),
        max_genes=config.get("max_genes", 6000),
        min_cells=config.get("min_cells", 3),
        min_counts=config.get("min_counts", 1000),
        max_counts=config.get("max_counts", 20000),
        max_mt_pct=config.get("max_mt_pct", 10.0),
        run_scrublet=config.get("run_scrublet", True),
        filter_doublets=config.get("filter_doublets", True),
        expected_doublet_rate=config.get("expected_doublet_rate", 0.06),
        n_top_genes=config.get("n_top_genes", 2000),
        batch_key=config.get("batch_key", None),
        run_harmony=config.get("run_harmony", True),
        random_state=config.get("random_state", 42),
    )
    print(f"Preprocessing complete. Output: {output_dir}")


if __name__ == "__main__":
    main()