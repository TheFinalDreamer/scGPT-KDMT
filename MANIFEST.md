# Repository Manifest

## What Is Included

| Category | Contents |
|----------|----------|
| **Source code** | Full model implementation: Student model, Transformer encoder, classifier, masked gene predictor, knowledge distillation losses, contrastive loss, training loop, evaluation metrics |
| **Data pipeline** | Preprocessing script (QC, HVG selection, normalization, splitting) |
| **Experiment scripts** | Teacher fine-tuning, student KDMT training (frozen and fine-tuned KD), student baseline, vanilla transformer baseline, ablation experiments (9 configs), multi-seed evaluation, train-only HVG, gene-number fair comparison |
| **Configuration files** | 12 YAML configs for all experiments with documented hyperparameters |
| **Utility scripts** | Environment checker, evaluation, result summarization, table generation, figure generation |
| **Tests** | Import test, model forward pass test, config loading test |
| **Documentation** | README, Model Overview, Reproducibility Guide, Data Availability Statement, Manifest |
| **Citation** | CITATION.cff with author metadata |
| **Environment** | requirements.txt, environment.yml (conda) |
| **License** | MIT License |

## What Is NOT Included

| Category | Reason |
|----------|--------|
| **Raw/processed data files** (.h5ad, .npz, .h5) | Large file size (hundreds of MB to GB). Data is publicly available from 10x Genomics / Seurat. |
| **Pre-trained model weights** (.pt, .pth, .ckpt) | Large file size. Weights can be reproduced by running the provided training scripts. scGPT pre-trained weights are available from the official scGPT repository. |
| **Training logs and intermediate results** | Temporary/voluminous files from server runs |
| **WandB / TensorBoard logs** | Run-specific artifacts not needed for reproduction |
| **Private datasets** | Only publicly available benchmark data is used |
| **Server-specific scripts** | Paths and configurations tied to specific servers |

## Why Large Files Are Excluded

Single-cell RNA-seq datasets (h5ad files) and model checkpoints (.pth files) are typically hundreds of megabytes to several gigabytes. They exceed GitHub's file size limits and are standard practice to exclude from code repositories. The provided scripts allow reviewers to reproduce the full workflow by:

1. Downloading the public PBMC 10X dataset
2. Running the preprocessing pipeline
3. Training models using the provided configurations

## How to Obtain Public Data

The PBMC 10X benchmark dataset is publicly available through:
- The Seurat v4 R package (`pbmc_multimodal_2023.rds`)
- Public single-cell data portals providing h5ad-formatted PBMC reference data

See [data/README.md](data/README.md) for detailed instructions.

## Quick Test

A minimal test configuration is provided at `examples/minimal_config.yaml`. It uses a small model (2 layers, 32-dim) for 3 epochs to verify the pipeline works correctly:

```bash
python experiments/train_student_kdmt.py \
  --config examples/minimal_config.yaml \
  --data_path outputs/processed/pbmc10x \
  --output_dir outputs/quick_test
```

## Reproducing Full Experiments

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete step-by-step reproduction guide, including:
1. Environment setup
2. Data download and preprocessing
3. Teacher fine-tuning
4. Student KDMT training (frozen KD and fine-tuned KD)
5. Student baseline and vanilla transformer baseline
6. Ablation experiments (9 configurations)
7. Multi-seed evaluation
8. Train-only HVG experiment
9. Gene-number fair comparison
10. Result aggregation, table generation, and figure generation

## File Count

| Directory | Files |
|-----------|-------|
| configs/ | 12 YAML files |
| src/scgpt_kdmt/ | 16 Python files |
| experiments/ | 8 Python scripts |
| scripts/ | 6 Python scripts |
| tests/ | 3 Python test files |
| docs/ | 3 Markdown files |
| examples/ | 2 files |
| Root | 8 files |
| **Total** | **60+ files** |
