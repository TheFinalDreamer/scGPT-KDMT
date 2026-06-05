# scGPT-KDMT: scGPT-guided Multi-task Knowledge Distillation for Efficient Single-Cell Cell Type Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository accompanies the following Bioinformatics submission:

> **scGPT-guided Multi-task Knowledge Distillation Framework for Efficient Single-Cell Cell Type Classification**

**Authors**: Wendong Du, Tengyue Mao, Wei Xiong, Lvyi Chen, Cong Liu

## Overview

scGPT-KDMT is a knowledge distillation framework that transfers biological representations from a fine-tuned scGPT teacher (51.3M parameters) into a lightweight Transformer student (2.92M parameters, 17.6x compression) for efficient single-cell RNA-seq cell type classification. The student is trained with a multi-task objective combining supervised classification, feature-level distillation, logit-level distillation, expression reconstruction, masked gene prediction, and supervised contrastive learning.

## Method Summary

- **Teacher model**: Fine-tuned scGPT-based Transformer for cell type classification
- **Student model**: Lightweight Transformer with gene-index embedding, classification head, reconstruction decoder, and mask prediction head
- **Knowledge distillation**: Feature-level (MSE) and logit-level (KL divergence) distillation from teacher to student
- **Multi-task learning**: Joint optimization of classification, distillation, expression reconstruction, and masked gene prediction objectives
- **Cell type classification**: 31-class fine-grained immune cell type annotation on PBMC benchmark dataset

## Repository Structure

```
scGPT-KDMT/
├── README.md
├── LICENSE                               # MIT License
├── MANIFEST.md                           # Repository content manifest
├── CITATION.cff                          # Citation metadata
├── requirements.txt                      # pip dependencies
├── environment.yml                       # Conda environment (Python 3.10)
├── .gitignore                            # Git ignore rules
├── configs/                              # YAML configuration files
│   ├── student_kdmt.yaml                 # Main KDMT student training config
│   ├── teacher_scgpt.yaml                # Teacher fine-tuning config
│   ├── frozen_kd.yaml                    # Frozen scGPT KD config
│   ├── finetuned_kd.yaml                 # Fine-tuned scGPT KD config
│   ├── student_baseline.yaml             # Student CE-only baseline
│   ├── vanilla_transformer.yaml          # Vanilla Transformer baseline
│   ├── ablation.yaml                     # Ablation experiment config
│   ├── multi_seed.yaml                   # Multi-seed experiment config
│   ├── train_only_hvg.yaml               # Train-only HVG experiment config
│   ├── gene_number_fair_comparison.yaml  # Gene-number fair comparison config
│   ├── preprocess_pbmc10x.yaml           # Data preprocessing config
│   └── figure_generation.yaml            # Figure generation config
├── src/scgpt_kdmt/                       # Core library
│   ├── data/                             # Dataset, dataloader, masking, split
│   ├── models/                           # Student model, Transformer, classifier, mask predictor
│   ├── training/                         # Multi-task training loop
│   ├── losses/                           # KD loss, contrastive loss
│   ├── evaluation/                       # Classification and clustering metrics
│   ├── visualization/                    # Plotting utilities
│   └── utils/                            # Seed, I/O, config loading
├── experiments/                          # Experiment entry points
│   ├── train_teacher_scgpt.py            # Teacher fine-tuning
│   ├── train_student_kdmt.py             # Student KDMT training (main)
│   ├── run_ablation.py                   # Ablation experiments (9 configs)
│   ├── run_student_baseline.py           # Student CE-only baseline
│   ├── run_vanilla_transformer.py        # Vanilla Transformer baseline
│   ├── run_multi_seed.py                 # Multi-seed evaluation
│   ├── run_train_only_hvg.py             # Train-only HVG experiment
│   └── run_gene_number_fair_comparison.py # Gene-number fair comparison
├── scripts/                              # Utility scripts
│   ├── preprocess_data.py                # h5ad → npz preprocessing pipeline
│   ├── check_environment.py              # Dependency checker
│   ├── evaluate.py                       # Model evaluation on test set
│   ├── summarize_results.py              # Consolidate results across experiments
│   ├── generate_tables.py                # Generate summary tables (CSV/LaTeX)
│   └── generate_figures.py               # Generate figures from results
├── tests/                                # Unit tests
│   ├── test_imports.py                   # Module import test
│   ├── test_model_forward.py             # Model forward pass test
│   └── test_config_loading.py            # Config loading test
├── examples/                             # Minimal usage examples
│   ├── README.md
│   └── minimal_config.yaml               # Quick test configuration (small model, 3 epochs)
├── data/
│   └── README.md                         # Data sources and preparation guide
├── results/
│   └── README.md                         # Output structure description
├── docs/                                 # Supplementary documentation
│   ├── DATA_AVAILABILITY.md              # Data availability statement
│   ├── REPRODUCIBILITY.md                # Detailed reproduction guide
│   └── MODEL_OVERVIEW.md                 # Model architecture overview
└── figures/
    └── README.md                         # Generated figures location
```

## Installation

### Using conda (recommended)

```bash
conda env create -f environment.yml
conda activate scgpt-kdmt
```

### Using pip

```bash
pip install -r requirements.txt
```

### Verify

```bash
python scripts/check_environment.py
```

## Data

This study uses the publicly available PBMC 10X benchmark dataset from 10x Genomics / Seurat reference resources. Large data files (h5ad, npz) are not included in this GitHub repository.

**Data preparation:**

```bash
# 1. Obtain the PBMC 10X dataset in h5ad format (not provided by this repo)
#    Source: 10x Genomics / Seurat v4 R package (pbmc_multimodal_2023.rds)

# 2. Preprocess
python scripts/preprocess_data.py \
  --config configs/preprocess_pbmc10x.yaml \
  --input data/raw/pbmc_seurat_v4.h5ad \
  --output outputs/processed/pbmc10x
```

See [data/README.md](data/README.md) for detailed data sources and the complete preprocessing workflow.

## Quick Start

After installing dependencies and preparing data:

```bash
# Verify environment
python scripts/check_environment.py

# Quick test with minimal model (3 epochs, small dims)
python experiments/train_student_kdmt.py \
  --config examples/minimal_config.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/quick_test
```

## Reproducing Experiments

### 1. Fine-tune the Teacher Model

```bash
python experiments/train_teacher_scgpt.py \
  --config configs/teacher_scgpt.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/teacher
```

- Architecture: 12 Transformer layers, 512-dim, 8 heads (~51.3M params)
- Epochs: 10; Optimizer: AdamW, lr=1e-4

### 2. Train scGPT-KDMT Student (Fine-tuned KD)

```bash
python experiments/train_student_kdmt.py \
  --config configs/finetuned_kd.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/kdmt_student \
  --teacher_checkpoint outputs/teacher/teacher_best.pth
```

- Architecture: 3 Transformer layers, 128-dim, 4 heads (2.92M params)
- Epochs: 50; KD temperature: 2.0; Six joint objectives

### 3. Frozen scGPT KD

```bash
python experiments/train_student_kdmt.py \
  --config configs/frozen_kd.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/frozen_kd \
  --teacher_checkpoint checkpoints/scgpt_pretrained.pth
```

### 4. Student Baseline (CE only, no KD)

```bash
python experiments/run_student_baseline.py \
  --config configs/student_baseline.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/student_baseline
```

### 5. Vanilla Transformer Baseline

```bash
python experiments/run_vanilla_transformer.py \
  --config configs/vanilla_transformer.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/vanilla_transformer
```

### 6. Ablation Experiments

```bash
python experiments/run_ablation.py \
  --config configs/ablation.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/ablation
```

Nine configurations systematically remove one or more loss components:
`Full_MultiTask`, `w/o_Reconstruction`, `w/o_Contrastive`, `KD_Only`, `w/o_FeatureKD`, `w/o_MaskPred`, `w/o_LogitsKD`, `w/o_KD_Both`, `Student_CE_Only`.

### 7. Multi-Seed Experiments

```bash
python experiments/run_multi_seed.py \
  --config configs/multi_seed.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/multi_seed
```

Seeds: {42, 2024, 2025}. Reports mean and standard deviation of Macro-F1 across seeds.

### 8. Train-Only HVG Experiment

```bash
# First, preprocess with train-only HVG selection
python scripts/preprocess_data.py \
  --config configs/preprocess_pbmc10x.yaml \
  --input data/raw/pbmc_seurat_v4.h5ad \
  --output outputs/processed/pbmc10x_train_only_hvg

python experiments/run_train_only_hvg.py \
  --config configs/train_only_hvg.yaml \
  --data_path outputs/processed/pbmc10x_train_only_hvg \
  --output_dir outputs/train_only_hvg \
  --teacher_checkpoint outputs/teacher/teacher_best.pth
```

### 9. Gene-Number Fair Comparison

```bash
python experiments/run_gene_number_fair_comparison.py \
  --config configs/gene_number_fair_comparison.yaml \
  --data_path outputs/processed \
  --output_dir outputs/gene_number_fair
```

Evaluates model performance at multiple gene counts (500, 1000, 2000, 4000, 8000).

### 10. Evaluation

```bash
python scripts/evaluate.py \
  --config configs/student_kdmt.yaml \
  --checkpoint outputs/kdmt_student/student_best.pth \
  --data outputs/processed/pbmc10x \
  --output outputs/results
```

### 11. Result Aggregation and Visualization

```bash
# Summarize results across all experiments
python scripts/summarize_results.py \
  --results_dir outputs/ \
  --output outputs/summaries/results_summary

# Generate tables (CSV and LaTeX)
python scripts/generate_tables.py \
  --summary outputs/summaries/results_summary.json \
  --output outputs/tables

# Generate figures
python scripts/generate_figures.py \
  --config configs/figure_generation.yaml \
  --results_dir outputs/ \
  --output_dir figures/
```

## Outputs

Training and evaluation produce the following outputs:

- `metrics.json` — Classification metrics (accuracy, macro-F1, precision, recall)
- `*_best.pth` — Model checkpoints
- `ablation_results.json` — Per-configuration ablation metrics
- `results_summary.json` — Consolidated multi-experiment summary
- `*.csv` / `*.tex` — Summary tables
- `*.png` — Generated figures (training curves, bar charts, box plots)
- `preprocess_config.json` — Preprocessing parameters used

## Data Availability

The PBMC benchmark dataset used in this study is publicly available from the 10x Genomics and Seurat reference resources, as described in the manuscript. Source code, preprocessing scripts, model implementation, configuration files, and reproducibility instructions are publicly available in this repository.

Large-scale single-cell datasets and pretrained model weights are not redistributed in this repository due to file size and licensing restrictions. See docs/DATA_AVAILABILITY.md for data sources and preparation instructions.

## Citation

If you use this code, please cite the corresponding manuscript:

Du W, Mao T, Xiong W, Chen L, Liu C. scGPT-guided Multi-task Knowledge Distillation Framework for Efficient Single-Cell Cell Type Classification. Bioinformatics, submitted.

Citation metadata is provided in CITATION.cff.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
