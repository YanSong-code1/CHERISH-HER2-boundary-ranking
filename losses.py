"""Core multitask and hierarchy consistency objectives for CHERISH."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_hierarchy_loss(
    out: dict[str, torch.Tensor],
    task_b_mask: torch.Tensor,
    task_c_mask: torch.Tensor,
    masked_only: bool = True,
    base_consistency_weight: float = 1.0,
    final_consistency_weight: float = 0.0,
) -> torch.Tensor:
    """Align supervised intermediate readouts with primary state masses."""

    if masked_only and float(task_c_mask.detach().cpu().item()) <= 0.0:
        return out["task_a_logit"].new_zeros(())

    p_a = torch.sigmoid(out["task_a_logit"]).squeeze()
    p_b = torch.sigmoid(out["task_b_logit"]).squeeze()
    use_task_b_term = float(task_b_mask.detach().cpu().item()) > 0.0
    loss = out["task_a_logit"].new_zeros(())

    def consistency_term(task_c_prob: torch.Tensor) -> torch.Tensor:
        term = F.mse_loss(p_a, task_c_prob[1] + task_c_prob[2])
        if use_task_b_term:
            denom = (task_c_prob[1] + task_c_prob[2]).clamp_min(1e-6)
            term = term + F.mse_loss(p_b, task_c_prob[1] / denom)
        return term

    if float(base_consistency_weight) > 0.0:
        loss = loss + float(base_consistency_weight) * consistency_term(
            out["task_c_base_prob"]
        )
    if float(final_consistency_weight) > 0.0:
        loss = loss + float(final_consistency_weight) * consistency_term(
            torch.softmax(out["task_c_logit"], dim=0)
        )
    return loss


def combine_weighted_losses(
    *,
    loss_a: torch.Tensor,
    loss_b: torch.Tensor,
    loss_c: torch.Tensor,
    hierarchy_loss: torch.Tensor,
    prototype_loss: torch.Tensor,
    task_a_weight: float,
    task_b_weight: float,
    task_c_weight: float,
    hierarchy_weight: float,
    prototype_weight: float,
) -> torch.Tensor:
    """Combine the task and structural losses used during training."""

    return (
        task_a_weight * loss_a
        + task_b_weight * loss_b
        + task_c_weight * loss_c
        + hierarchy_weight * hierarchy_loss
        + prototype_weight * prototype_loss
    )


def scheduled_hierarchy_weight(
    *,
    epoch: int,
    target_weight: float,
    warmup_epochs: int,
    start_weight: float = 0.0,
) -> float:
    """Linearly warm up the hierarchy term to its target weight."""

    target_weight = float(target_weight)
    start_weight = float(start_weight)
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 1:
        return target_weight
    if epoch <= 0:
        return start_weight
    if epoch >= warmup_epochs - 1:
        return target_weight
    progress = float(epoch) / float(warmup_epochs - 1)
    return start_weight + (target_weight - start_weight) * progress
