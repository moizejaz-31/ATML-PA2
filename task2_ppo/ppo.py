from __future__ import annotations

import torch

from common.metrics import masked_mean


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level GAE over response positions.

    rewards, values, mask: [batch, response_steps]. Padding positions must have mask=0.
    The final valid response position bootstraps with zero.
    """
    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(steps)):
        current_valid = mask[:, t]
        if t + 1 < steps:
            next_valid = mask[:, t + 1]
            next_value = values[:, t + 1] * next_valid
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(last_adv)

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * next_valid * last_adv
        last_adv = last_adv * current_valid
        advantages[:, t] = last_adv

    returns = advantages + values
    return advantages, returns


def shaped_rewards(task_reward, policy_logp, ref_logp, response_mask, beta_kl):
    """Sampled-action KL shaping plus terminal learned reward."""
    rewards = -float(beta_kl) * (policy_logp - ref_logp) * response_mask
    for b in range(rewards.shape[0]):
        valid = int(response_mask[b].sum().item())
        if valid > 0:
            rewards[b, valid - 1] += task_reward[b]
    return rewards


def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    """Return PPO clipped policy loss and diagnostics.

    Validate this starter implementation against the clipped surrogate in the assignment manual.
    """
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantage
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage

    # BUG FIX: Starter code used torch.maximum (optimistic bound).
    # PPO establishes a pessimistic lower bound on the unclipped objective by
    # taking the minimum of surr1 and surr2:
    # L_CLIP(θ) = E[ min( r_t(θ) A_t, clip(r_t(θ), 1-eps, 1+eps) A_t ) ]
    objective = torch.minimum(surr1, surr2)

    loss = -masked_mean(objective, mask)
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    clip_fraction = masked_mean(affected, mask)

    # Detailed clipping metrics for research questions:
    clip_high = ((ratio > (1.0 + eps)) & (advantage > 0)).float()
    clip_low = ((ratio < (1.0 - eps)) & (advantage < 0)).float()
    return loss, ratio.detach(), {
        "clip_fraction": clip_fraction.detach(),
        "clip_high_fraction": masked_mean(clip_high, mask).detach(),
        "clip_low_fraction": masked_mean(clip_low, mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), mask),
        "ratio_min": (ratio * mask + (1 - mask) * 1.0).min().detach(),
        "ratio_max": (ratio * mask).max().detach(),
        "policy_loss": loss.detach(),
    }


def value_mse_loss(predicted_values, returns, mask):
    return masked_mean((predicted_values - returns) ** 2, mask)


def normalize_advantages(advantages, mask, eps=1e-6):
    valid = advantages[mask.bool()]
    if valid.numel() <= 1:
        return advantages
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    return ((advantages - mean) / std) * mask
