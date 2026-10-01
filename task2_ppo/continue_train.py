from __future__ import annotations

import argparse
from pathlib import Path
import time
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import (
    count_parameters,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def plot_ppo_continuation(log_path: Path, fig_dir: Path, run_name: str = "standard"):
    """Plot multi-panel PPO training progression."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        return

    updates = [r["update"] for r in records]
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)

    # 1. Learned Task Reward
    axes[0, 0].plot(updates, [r["reward_mean"] for r in records], color="tab:blue", marker="o")
    axes[0, 0].set_ylabel("Learned Reward", fontsize=10)
    axes[0, 0].set_title("Terminal Reward Model Score", fontsize=11, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Reference Policy KL Drift
    axes[0, 1].plot(updates, [r["kl_mean"] for r in records], color="tab:red", marker="s")
    axes[0, 1].set_ylabel("KL(π_θ || π_ref)", fontsize=10)
    axes[0, 1].set_title("Policy Drift (KL from Reference)", fontsize=11, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Policy Loss & Value Loss
    axes[1, 0].plot(updates, [r["policy_loss"] for r in records], color="tab:purple", label="Policy Loss")
    axes[1, 0].set_ylabel("Policy Loss", fontsize=10)
    axes[1, 0].set_title("PPO Clipped Surrogate Policy Loss", fontsize=11, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(updates, [r["value_loss"] for r in records], color="tab:brown", label="Value Loss")
    axes[1, 1].set_ylabel("Value Loss (MSE)", fontsize=10)
    axes[1, 1].set_title("Critic Value Loss (Return MSE)", fontsize=11, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    # 4. Clip Fraction & Entropy
    axes[2, 0].plot(updates, [r["clip_fraction"] for r in records], color="tab:orange", marker="^")
    axes[2, 0].set_ylabel("Clip Fraction", fontsize=10)
    axes[2, 0].set_xlabel("Update Step", fontsize=10)
    axes[2, 0].set_title("Fraction of Tokens Clipped", fontsize=11, fontweight="bold")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(updates, [r["response_length_mean"] for r in records], color="tab:green", marker="d")
    axes[2, 1].set_ylabel("Response Length (tokens)", fontsize=10)
    axes[2, 1].set_xlabel("Update Step", fontsize=10)
    axes[2, 1].set_title("Mean Response Length", fontsize=11, fontweight="bold")
    axes[2, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Task 2 — PPO Continuation Dashboard ({run_name})", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig_path = fig_dir / f"task2_ppo_continuation_{run_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved PPO dashboard to {fig_path}")


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    clip_epsilon: float | None = None,
    kl_beta: float | None = None,
    run_name: str = "standard",
):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)

    policy = bundle["policy"]
    value_model = bundle["value_model"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    tokenizer = bundle["tokenizer"]
    prompts = bundle["prompt_rows"]
    opt_p = bundle["policy_optimizer"]
    opt_v = bundle["value_optimizer"]

    num_updates = int(cfg["updates"])
    eps = float(cfg["clip_epsilon"])
    beta_kl = float(cfg["kl_beta"])
    gamma = float(cfg.get("gamma", 1.0))
    lam = float(cfg.get("gae_lambda", 0.95))
    val_coef = float(cfg.get("value_coef", 0.50))
    ppo_epochs = int(cfg.get("ppo_epochs", 2))
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    max_resp_len = int(cfg.get("max_response_length", 512))
    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    missing_eos_pen = float(cfg.get("missing_eos_penalty", 1.0))

    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = repo_path("report/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / f"ppo_train_{run_name}.jsonl"

    print(f"============================================================")
    print(f"[PPO Continuation] run={run_name} updates={num_updates} eps={eps} beta_kl={beta_kl}")
    print(f"Policy trainable params: {count_parameters(policy)[1]:,}")
    print(f"Value trainable params:  {count_parameters(value_model)[1]:,}")
    print(f"Output adapter:          {out}")
    print(f"============================================================")

    timer = wall_timer()
    prompt_idx = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for update in range(1, num_updates + 1):
        # 1. Select prompts
        batch_prompts = []
        for _ in range(prompts_per_update):
            row = prompts[prompt_idx % len(prompts)]
            batch_prompts.append(prompt_messages(row))
            prompt_idx += 1

        # 2. On-policy rollout generation
        gen_data = batch_generate(
            policy,
            tokenizer,
            batch_prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_resp_len,
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            do_sample=bool(cfg["generation"]["do_sample"]),
        )

        sequences = gen_data["sequences"]
        attention_mask = gen_data["attention_mask"]
        prompt_width = gen_data["prompt_width"]
        response_ids = gen_data["response_ids"]
        response_mask = gen_data["response_mask"]
        responses = gen_data["responses"]
        terminated = gen_data["terminated_with_eos"]

        # 3. Terminal reward scoring
        with torch.no_grad():
            raw_rewards = score_reward_pairs(
                reward_model, reward_tokenizer, batch_prompts, responses, max_length=int(cfg["reward_max_length"])
            )
            # Apply missing EOS penalty
            for b_idx, has_eos in enumerate(terminated):
                if not has_eos:
                    raw_rewards[b_idx] -= missing_eos_pen

        # 4. Old policy logprobs and reference policy logprobs
        with torch.no_grad():
            old_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)

        # 5. KL shaping and GAE advantage estimation
        shaped_rew = shaped_rewards(raw_rewards, old_logp, ref_logp, response_mask, beta_kl)

        with torch.no_grad():
            token_vals = token_values(value_model, sequences, attention_mask)
            # Response-level values only
            resp_values = token_vals[:, prompt_width - 1 : prompt_width - 1 + response_ids.shape[1]]
            advantages, returns = compute_gae(shaped_rew, resp_values, response_mask, gamma=gamma, lam=lam)
            norm_advantages = normalize_advantages(advantages, response_mask)

        # 6. PPO Update Epochs
        policy.train()
        value_model.train()

        last_p_loss = 0.0
        last_v_loss = 0.0
        last_clip_fraction = 0.0
        last_entropy = 0.0

        for _ in range(ppo_epochs):
            # Policy forward
            new_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
            p_loss, _, clip_diag = ppo_policy_loss(new_logp, old_logp, norm_advantages, response_mask, eps=eps)

            opt_p.zero_grad()
            p_loss.backward()
            p_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm).item()
            opt_p.step()

            # Value forward
            cur_vals = token_values(value_model, sequences, attention_mask)[:, prompt_width - 1 : prompt_width - 1 + response_ids.shape[1]]
            v_loss = value_mse_loss(cur_vals, returns, response_mask)

            opt_v.zero_grad()
            (v_loss * val_coef).backward()
            v_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_grad_norm).item()
            opt_v.step()

            last_p_loss = float(p_loss.item())
            last_v_loss = float(v_loss.item())
            last_clip_fraction = float(clip_diag["clip_fraction"].item())
            last_entropy = float(sample_entropy(new_logp.detach(), response_mask).item())

        # Metrics for the update
        kl_div = float(sampled_kl(old_logp, ref_logp, response_mask).item())
        mean_rew = float(raw_rewards.mean().item())
        mean_len = float(response_mask.sum(-1).mean().item())
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

        record = {
            "update": update,
            "reward_mean": mean_rew,
            "kl_mean": kl_div,
            "policy_loss": last_p_loss,
            "value_loss": last_v_loss,
            "clip_fraction": last_clip_fraction,
            "entropy": last_entropy,
            "policy_grad_norm": p_norm,
            "value_grad_norm": v_norm,
            "response_length_mean": mean_len,
            "peak_vram_mb": peak_vram,
            "wall_time": float(timer()),
        }
        append_jsonl(log_path, record)

        if update % 2 == 0 or update == 1 or update == num_updates:
            print(
                f"[Update {update:2d}/{num_updates}] "
                f"R={mean_rew:+.3f} | KL={kl_div:.4f} | "
                f"P_loss={last_p_loss:+.4f} | V_loss={last_v_loss:.4f} | "
                f"Clip={last_clip_fraction:.2%} | Len={mean_len:.0f}tok | "
                f"VRAM={peak_vram:.0f}MB | Time={timer():.1f}s"
            )

    # Save policy adapter and tokenizer
    print(f"\n[PPO] Saving continued policy adapter to {out}")
    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    summary = {
        "run_name": run_name,
        "updates": num_updates,
        "clip_epsilon": eps,
        "kl_beta": beta_kl,
        "final_reward": mean_rew,
        "final_kl": kl_div,
        "final_policy_loss": last_p_loss,
        "final_value_loss": last_v_loss,
        "peak_vram_mb": peak_vram,
        "wall_time_seconds": timer(),
        "adapter_path": str(out),
    }
    save_json(results_dir / f"ppo_summary_{run_name}.json", summary)

    # Generate plots
    try:
        plot_ppo_continuation(log_path, fig_dir, run_name)
    except Exception as e:
        print(f"[PPO plot] Warning: plotting failed: {e}")

    print(f"[PPO] Continuation complete in {timer():.1f}s.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
