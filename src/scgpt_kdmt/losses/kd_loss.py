"""Knowledge distillation losses."""

import torch
import torch.nn.functional as F


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Logit-level knowledge distillation loss (KL divergence).

    Args:
        student_logits: Student model logits (batch_size, num_classes).
        teacher_logits: Teacher model logits (batch_size, num_classes).
        temperature: Distillation temperature (default: 4.0).

    Returns:
        Scalar KL divergence loss scaled by temperature^2.
    """
    T = temperature
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(
        student_log_probs, teacher_probs, reduction="batchmean"
    ) * (T * T)


def feature_distillation_loss(
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
) -> torch.Tensor:
    """Feature-level distillation loss (MSE between embeddings).

    Args:
        student_embedding: Student cell embeddings (batch_size, hidden_dim).
        teacher_embedding: Teacher cell embeddings (batch_size, hidden_dim).

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(student_embedding, teacher_embedding)