# Model Overview

## Overall Framework

scGPT-KDMT is a knowledge distillation framework that transfers biological representations from a large pre-trained scGPT teacher model to a lightweight Transformer student model for efficient single-cell RNA-seq cell type classification. The student is trained with multi-task joint optimization combining supervised classification, knowledge distillation objectives, and unsupervised gene expression modeling tasks.

## Teacher Model: scGPT

The teacher is a fine-tuned Transformer model with the following characteristics:
- **Architecture**: 12 Transformer encoder layers, 512-dimensional hidden states, 8 attention heads
- **Parameters**: ~51.3M
- **Input**: Gene expression vector over full gene vocabulary (e.g., 16,545 genes)
- **Pre-training**: Self-supervised masked gene modeling on large-scale single-cell atlases
- **Fine-tuning**: Supervised cell type classification on the target PBMC dataset

The teacher encodes each gene through a gene-index embedding, where genes are identified by their position in the gene vocabulary. Individual gene expression values are projected through this embedding layer before being processed by the stacked Transformer encoder.

## Student Model

The student is a compact Transformer designed for efficient inference:
- **Architecture**: 3 Transformer encoder layers, 128-dimensional hidden states, 4 attention heads
- **Parameters**: ~2.92M (17.6x compression vs. teacher)
- **Input**: Gene expression vector over the top 2,000 highly variable genes (HVGs)
- **Multi-head outputs**: The student produces four outputs from a single forward pass:
  - `embedding`: Cell-level representation (128-dim, mean-pooled across genes)
  - `logits`: Cell type class probabilities (31 classes)
  - `recon`: Reconstructed gene expression (2,000-dim)
  - `masked_pred`: Predicted expression for randomly masked genes (2,000-dim)

The student also uses gene-index embeddings where each gene is identified by its position in the HVG list. No explicit positional encoding is used beyond the gene identity embedding.

## Multi-Task Learning Objectives

The student is trained with the following six objectives, all combined into a single training loop:

1. **Supervised Classification Loss** (cross-entropy): Standard cell type classification using ground-truth labels.

2. **Feature-level Knowledge Distillation Loss** (MSE): Aligns the student's cell embeddings with the teacher's cell embeddings in the hidden representation space.

3. **Logit-level Knowledge Distillation Loss** (KL divergence with temperature): Transfers the teacher's softened class probability distribution to the student. Temperature tau controls the softness of the target distribution.

4. **Reconstruction Loss** (MSE): The student reconstructs the original gene expression values from its cell embedding via a linear decoder.

5. **Masked Gene Prediction Loss** (MSE): 15% of input genes are randomly masked (set to zero). A mask predictor head reconstructs the original values at masked positions.

6. **Supervised Contrastive Loss**: Pulls together cell embeddings belonging to the same cell type while pushing apart embeddings from different types. Based on Khosla et al. (NeurIPS 2020).

## Total Loss

```
L_total = L_cls + lambda_feat * L_feat_kd + lambda_logit * L_logit_kd
          + lambda_recon * L_recon + lambda_mask * L_mask + lambda_contrast * L_contrast
```

All lambda weights are set to 0.5 by default. Each loss component can be toggled independently for ablation analysis.

## Feature-Level Knowledge Distillation

Feature-level KD aligns the internal representations between teacher and student. The teacher's output cell embedding serves as a target for the student's embedding via mean squared error. This encourages the student to learn similar feature representations to the teacher, capturing the biological patterns encoded in the teacher's larger feature space.

## Cell Type Classification Head

A two-layer MLP (hidden_dim=256, ReLU activation) maps the cell embedding to class logits over 31 fine-grained immune cell types.

## Expression Reconstruction / Mask Prediction Auxiliary Objectives

These unsupervised objectives encourage the student to learn biologically meaningful representations:

- **Reconstruction**: Ensures the cell embedding retains sufficient information to reconstruct the full gene expression profile.
- **Masked Gene Prediction**: Trains the model to impute expression values of held-out genes, similar to the pre-training objective used by scGPT.

## Evaluation Metrics

### Classification
- Accuracy, Macro-F1, macro-precision, macro-recall

### Clustering Quality
- Adjusted Rand Index (ARI), Normalized Mutual Information (NMI)
- Computed via KMeans clustering on learned cell embeddings

### Efficiency
- Parameter count comparison
- Inference throughput (cells/second)
- Compression ratio (student params / teacher params)

## Relationship to Manuscript Experiments

| Experiment | Script | Description |
|-----------|--------|-------------|
| Teacher Fine-tuning | `train_teacher_scgpt.py` | Fine-tune teacher for classification |
| Student KDMT | `train_student_kdmt.py` | Full multi-task KD training |
| Ablation Study | `run_ablation.py` | 9 configurations removing components |
| Evaluation | `evaluate.py` | Test set metrics |
| Quick Test | `minimal_config.yaml` | Sanity check with small model |