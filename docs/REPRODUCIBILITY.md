# Reproducibility Guide

## Environment

- **Python**: 3.10
- **PyTorch**: 2.0+ (CUDA 11.8 / 12.1 recommended for GPU training)
- **OS**: Linux / Windows / macOS
- **GPU**: Single NVIDIA GPU recommended for full experiments (tested on RTX 3060, RTX 4090). CPU is sufficient for inference and quick tests.

## Setup

### Using conda (recommended)

```bash
conda env create -f environment.yml
conda activate scgpt-kdmt
```

### Using pip

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python scripts/check_environment.py
```

## Key Package Versions

The experiments reported in the manuscript used the following approximate versions:
- scanpy >= 1.9
- scikit-learn >= 1.0
- numpy >= 1.21, < 2.0
- pytorch >= 2.0

Minor version differences should not affect results given fixed random seeds.

## Data Preparation

### 1. Obtain the PBMC 10X Dataset

The PBMC 10X dataset (161,764 cells, 31 cell types at Seurat celltype.l2 resolution) is publicly available from the Seurat v4 R package. It can be exported to h5ad format using standard single-cell tooling.

### 2. Preprocess

```bash
python scripts/preprocess_data.py \
  --config configs/preprocess_pbmc10x.yaml \
  --input data/raw/pbmc_seurat_v4.h5ad \
  --output outputs/processed/pbmc10x
```

Preprocessing steps:
1. QC filtering (min_genes=300, max_genes=6000, min_counts=1000, max_counts=20000, max_mt_pct=10%)
2. Doublet detection with Scrublet
3. HVG selection (top 2000, Seurat v3)
4. Library-size normalization (10,000) + log1p
5. Z-score standardization per gene
6. Stratified 70/15/15 train/val/test split

## Expected Input Format

```
outputs/processed/pbmc10x/
├── train.npz          # X_train (N x 2000 float32), y_train (N int64)
├── val.npz            # X_val, y_val
├── test.npz           # X_test, y_test
├── label_encoder.pkl  # sklearn LabelEncoder
├── hvg_gene_names.npy # HVG gene names
└── preprocess_config.json
```

## Experiment Configurations

All hyperparameters are specified in YAML configuration files under `configs/`:

| Config | Purpose |
|--------|---------|
| `teacher_scgpt.yaml` | Teacher model fine-tuning |
| `student_kdmt.yaml` | Main student KDMT training |
| `frozen_kd.yaml` | Frozen scGPT KD (teacher not fine-tuned on target data) |
| `finetuned_kd.yaml` | Fine-tuned scGPT KD (teacher fine-tuned on target data) |
| `student_baseline.yaml` | Student model with CE loss only (no KD) |
| `vanilla_transformer.yaml` | Vanilla Transformer baseline (CE only) |
| `ablation.yaml` | Ablation experiment (9 configurations) |
| `multi_seed.yaml` | Multi-seed evaluation (3 seeds) |
| `train_only_hvg.yaml` | Train-only HVG selection experiment |
| `gene_number_fair_comparison.yaml` | Gene-number fair comparison |
| `preprocess_pbmc10x.yaml` | Data preprocessing |
| `figure_generation.yaml` | Result visualization |
| `examples/minimal_config.yaml` | Quick test / sanity check |

## Main Experiment Commands

### 1. Fine-tune Teacher

```bash
python experiments/train_teacher_scgpt.py \
  --config configs/teacher_scgpt.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/teacher
```

- Architecture: 12 Transformer layers, 512-dim, 8 heads (~51.3M params)
- Epochs: 10
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Best model selected by validation Macro-F1

### 2. Train scGPT-KDMT Student (Fine-tuned KD)

```bash
python experiments/train_student_kdmt.py \
  --config configs/finetuned_kd.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/kdmt_student \
  --teacher_checkpoint outputs/teacher/teacher_best.pth
```

- Architecture: 3 Transformer layers, 128-dim, 4 heads (2.92M params)
- Epochs: 50
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- KD temperature: tau=2.0
- Loss weights: all lambda=0.5

### 3. Frozen scGPT KD

```bash
python experiments/train_student_kdmt.py \
  --config configs/frozen_kd.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/frozen_kd \
  --teacher_checkpoint checkpoints/scgpt_pretrained.pth
```

Uses a frozen scGPT teacher (not fine-tuned on the target dataset) for feature-level knowledge distillation.

### 4. Student Baseline

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

Nine ablation configurations systematically remove one or more loss components.

### 7. Multi-Seed Experiments

```bash
python experiments/run_multi_seed.py \
  --config configs/multi_seed.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/multi_seed
```

Seeds used in the manuscript: {42, 2024, 2025}. Reports mean ± std of Macro-F1.

### 8. Train-Only HVG Experiment

```bash
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

Note: The train-only HVG selection requires a modified preprocessing workflow that restricts HVG selection to the training split. See the script documentation for details.

### 9. Gene-Number Fair Comparison

```bash
python experiments/run_gene_number_fair_comparison.py \
  --config configs/gene_number_fair_comparison.yaml \
  --data_path outputs/processed \
  --output_dir outputs/gene_number_fair
```

Requires preprocessed data at each gene count. Place files as `{data_path}/g{N}/train.npz, val.npz, test.npz`.

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
# Summarize results across experiments
python scripts/summarize_results.py \
  --results_dir outputs/ \
  --output outputs/summaries/results_summary

# Generate summary tables (CSV and LaTeX)
python scripts/generate_tables.py \
  --summary outputs/summaries/results_summary.json \
  --output outputs/tables

# Generate figures
python scripts/generate_figures.py \
  --config configs/figure_generation.yaml \
  --results_dir outputs/ \
  --output_dir figures/
```

## Expected Outputs

After training, output directories contain:

```
outputs/kdmt_student/
├── student_best.pth      # Best checkpoint (val F1 selection)
└── metrics.json          # Training losses and validation metrics

outputs/ablation/
└── ablation_results.json # Per-configuration metrics

outputs/multi_seed/
└── metrics.json          # Mean ± std across seeds

outputs/results/
└── metrics.json          # Test set evaluation metrics

outputs/summaries/
└── results_summary.json  # Consolidated multi-experiment results

outputs/tables/
├── results_summary.csv   # CSV summary table
└── main_results.tex      # LaTeX table

figures/
├── training_curve.png    # Validation F1 over epochs
├── ablation_bar.png      # Ablation study bar chart
└── multi_seed_box.png    # Multi-seed box plot
```

## Evaluation Metrics

- **Classification**: Accuracy, Macro-F1, macro-precision, macro-recall
- **Clustering**: ARI, NMI (KMeans on embeddings)
- **Efficiency**: Parameter count, inference throughput

## Hardware Notes

Approximate runtimes (single GPU):
- Teacher fine-tuning: ~27 minutes (10 epochs)
- Student KDMT training: ~37 minutes (50 epochs)
- Inference: ~0.71 seconds for 24,265 cells

Large-scale experiments require a CUDA-enabled GPU environment and access to public scGPT pretrained resources. The provided scripts and configuration files are intended to reproduce the reported workflow. Minor runtime variations may occur depending on GPU model, CUDA version, and system configuration.

## Troubleshooting

### NumPy / Scanpy compatibility
- The environment pins `numpy>=1.21,<2.0` to avoid known incompatibilities between NumPy 2.0+ and older scanpy versions.

### Scrublet not installed
- Optional dependency. If not installed, doublet detection is skipped.

### Harmony not installed
- Optional dependency. If not installed, batch correction is skipped. Not needed for single-batch PBMC data.

### CUDA out of memory
- Reduce `batch_size` in the config YAML.
- Use `examples/minimal_config.yaml` for a smaller model configuration.

### Import errors
- Ensure `src/` is on the Python path. The experiment scripts handle this automatically via `sys.path.insert`.

### Data not found
- Make sure the PBMC 10X h5ad file is obtained from public sources before running preprocessing.
- The repository does not redistribute data files. See [data/README.md](../data/README.md).
