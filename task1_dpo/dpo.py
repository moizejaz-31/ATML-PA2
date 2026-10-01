from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    The DPO objective (Rafailov et al., 2023):
        L_DPO = -E[ log σ( β · ( log π_θ(y+|x)/π_ref(y+|x) - log π_θ(y-|x)/π_ref(y-|x) ) ) ]

    Which simplifies to:
        logits = β * ( (π_θ_chosen - π_θ_rejected) - (π_ref_chosen - π_ref_rejected) )
               = β * ( policy_margin - ref_margin )

    BUG FIX: The starter code used (+) instead of (-). The correct formulation
    subtracts the reference margin so that the implicit reward reflects the
    log-ratio advantage over the reference, not the sum.
    """
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp

    # CORRECTED: subtract ref_margin (was incorrectly adding)
    logits = beta * (policy_margin - ref_margin)

    loss = -F.logsigmoid(logits).mean()

    # Compute the implicit reward for analysis: r(x,y) = β · log(π_θ(y|x) / π_ref(y|x))
    implicit_chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
    implicit_rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)
    reward_margin = implicit_chosen_reward - implicit_rejected_reward

    return loss, {
        "logit_mean": logits.detach().mean(),
        "logit_std": logits.detach().std(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "ref_margin_mean": ref_margin.detach().mean(),
        "preference_accuracy": (logits > 0).float().mean().detach(),
        "implicit_chosen_reward_mean": implicit_chosen_reward.detach().mean(),
        "implicit_rejected_reward_mean": implicit_rejected_reward.detach().mean(),
        "reward_margin_mean": reward_margin.detach().mean(),
    }
