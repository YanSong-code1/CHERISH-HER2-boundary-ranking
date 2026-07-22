"""Probability-conserving four-state inference for CHERISH."""

from __future__ import annotations

import torch


def probability_conserving_readout(
    pathway_entry_prob: torch.Tensor | float,
    state_probs: torch.Tensor,
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Compose the negative-side readout and primary three-state output.

    ``state_probs`` follows the order IHC 2+/FISH negative,
    IHC 2+/FISH amplified and IHC 3+. The returned four-state order is
    IHC 0/1+ followed by those three states.
    """

    if not isinstance(state_probs, torch.Tensor):
        raise TypeError("state_probs must be a torch.Tensor")
    if state_probs.ndim != 1 or state_probs.shape[0] != 3:
        raise ValueError("state_probs must have shape [3]")
    if not torch.isfinite(state_probs).all():
        raise ValueError("state_probs must contain finite values")
    if torch.any(state_probs < 0.0) or torch.any(state_probs > 1.0):
        raise ValueError("state_probs must lie in [0, 1]")
    if not torch.isclose(
        state_probs.sum(),
        state_probs.new_tensor(1.0),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError("state_probs must sum to one")

    entry = torch.as_tensor(
        pathway_entry_prob,
        dtype=state_probs.dtype,
        device=state_probs.device,
    )
    if entry.numel() != 1 or not torch.isfinite(entry).all():
        raise ValueError("pathway_entry_prob must be one finite scalar")
    entry = entry.reshape(())
    if entry < 0.0 or entry > 1.0:
        raise ValueError("pathway_entry_prob must lie in [0, 1]")

    four_state_probs = torch.cat(
        [(1.0 - entry).reshape(1), entry * state_probs]
    )
    if not torch.isclose(
        four_state_probs.sum(),
        four_state_probs.new_tensor(1.0),
        atol=atol,
        rtol=0.0,
    ):
        raise RuntimeError("four-state probability mass is not conserved")

    ihc2_amplification_prob = state_probs[1] / (
        state_probs[0] + state_probs[1] + float(eps)
    )
    return {
        "four_state_probs": four_state_probs,
        "ihc2_amplification_prob": ihc2_amplification_prob,
    }
