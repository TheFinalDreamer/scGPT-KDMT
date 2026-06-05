"""Supervised contrastive representation learning loss.

Implements the supervised contrastive loss from Khosla et al. (NeurIPS 2020).
"""

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss for cell type representation learning.

    Pulls together embeddings of cells with the same label while pushing
    apart embeddings of cells with different labels.

    Args:
        embeddings: Normalized embeddings of shape (batch_size, hidden_dim).
        labels: Integer class labels of shape (batch_size,).
        temperature: Contrastive temperature (default: 0.07).

    Returns:
        Scalar supervised contrastive loss.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    embeddings = F.normalize(embeddings, dim=1)

    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask

    exp_logits = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    loss = -mean_log_prob_pos.mean()
    return loss