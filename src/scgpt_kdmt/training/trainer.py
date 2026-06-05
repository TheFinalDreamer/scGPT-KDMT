"""Training loop for multi-task knowledge distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..data.masking import mask_input
from ..losses.kd_loss import kd_loss, feature_distillation_loss


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    teacher_model=None,
    # Ablation switches
    use_kd_feat: bool = True,
    use_kd_logit: bool = True,
    use_recon: bool = True,
    use_mask: bool = True,
    use_contrast: bool = True,
    # Loss weights
    lambda_feat: float = 0.5,
    lambda_logit: float = 0.5,
    lambda_recon: float = 0.5,
    lambda_mask: float = 0.5,
    lambda_contrast: float = 0.5,
    # KD temperature
    kd_temperature: float = 2.0,
    mask_ratio: float = 0.15,
):
    """Train the student model for one epoch.

    Args:
        model: Student model.
        loader: Training DataLoader.
        optimizer: Optimizer.
        device: torch device.
        teacher_model: Frozen teacher model (optional).
        use_kd_feat: Enable feature-level distillation.
        use_kd_logit: Enable logit-level distillation.
        use_recon: Enable reconstruction loss.
        use_mask: Enable masked gene prediction.
        use_contrast: Enable contrastive loss.
        lambda_feat: Feature KD weight (default: 0.5).
        lambda_logit: Logit KD weight (default: 0.5).
        lambda_recon: Reconstruction weight (default: 0.5).
        lambda_mask: Mask prediction weight (default: 0.5).
        lambda_contrast: Contrastive weight (default: 0.5).
        kd_temperature: Temperature for logit-level KD (default: 2.0).
        mask_ratio: Fraction of genes to mask (default: 0.15).

    Returns:
        Average total loss over the epoch.
    """
    model.train()
    if teacher_model is not None:
        teacher_model.eval()

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()

        x_masked, mask = mask_input(x, mask_ratio=mask_ratio)
        x_masked = x_masked.to(device)
        mask = mask.to(device)
        if mask.dtype != torch.bool:
            mask = mask.bool()

        outputs = model(x_masked)

        # Supervised classification
        cls_loss = ce(outputs["logits"], y)
        loss = cls_loss

        # Reconstruction
        if use_recon:
            recon_loss = mse(outputs["recon"], x)
            loss = loss + lambda_recon * recon_loss

        # Masked gene prediction
        if use_mask and mask.sum() > 0:
            masked_loss = mse(outputs["masked_pred"][mask], x[mask])
            loss = loss + lambda_mask * masked_loss

        # Contrastive learning
        if use_contrast:
            from ..losses.contrastive import supervised_contrastive_loss
            cont_loss = supervised_contrastive_loss(
                outputs["embedding"], y
            )
            loss = loss + lambda_contrast * cont_loss

        # Knowledge distillation from teacher
        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(x)

            if use_kd_logit:
                loss_logit = kd_loss(
                    outputs["logits"],
                    teacher_outputs["logits"],
                    temperature=kd_temperature,
                )
                loss = loss + lambda_logit * loss_logit

            if use_kd_feat:
                loss_feat = feature_distillation_loss(
                    outputs["embedding"],
                    teacher_outputs["embedding"],
                )
                loss = loss + lambda_feat * loss_feat

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)