from __future__ import annotations

import argparse
from pathlib import Path
import time
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import count_parameters, load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.grpo import (
    group_relative_advantages,
    grpo_policy_loss,
    mask_truncated_sequences,
)


def plot_grpo_continuation(log_path: Path, fig_dir: Path, run_name: str = "standard"):
    """Plot multi-panel GRPO training progression."""
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

    # 1. Mean Reward
    axes[0, 0].plot(updates, [r["reward_mean"] for r in records], color="tab:blue", marker="o")
    axes[0, 0].set_ylabel("Learned Reward", fontsize=10)
    axes[0, 0].set_title("Mean Reward Across Groups", fontsize=11, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Reference Policy KL
    axes[0, 1].plot(updates, [r["sampled_kl"] for r in records], color="tab:red", marker="s")
    axes[0, 1].set_ylabel("KL(π_θ || π_ref)", fontsize=10)
    axes[0, 1].set_title("Sampled KL Divergence from Reference", fontsize=11, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Within-Group Reward Std & Uninformative Groups
    axes[1, 0].plot(updates, [r["within_group_reward_std"] for r in records], color="tab:purple", marker="^")
    axes[1, 0].set_ylabel("Within-Group Std", fontsize=10)
    axes[1, 0].set_title("Within-Group Reward Standard Deviation (σ_r)", fontsize=11, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(updates, [r["uninformative_group_fraction"] * 100 for r in records], color="tab:orange", marker="v")
    axes[1, 1].set_ylabel("Uninformative Groups (%)", fontsize=10)
    axes[1, 1].set_title("Fraction of Uninformative Groups (σ_r ≈ 0)", fontsize=11, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    # 4. Policy Loss & Response Length
    axes[2, 0].plot(updates, [r["policy_loss"] for r in records], color="tab:brown")
    axes[2, 0].set_ylabel("Policy Loss", fontsize=10)
    axes[2, 0].set_xlabel("Update Step", fontsize=10)
    axes[2, 0].set_title("GRPO Policy Loss", fontsize=11, fontweight="bold")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(updates, [r["response_length_mean"] for r in records], color="tab:green", marker="d")
    axes[2, 1].set_ylabel("Length (tokens)", fontsize=10)
    axes[2, 1].set_xlabel("Update Step", fontsize=10)
    axes[2, 1].set_title("Mean Response Length", fontsize=11, fontweight="bold")
    axes[2, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Task 3 — GRPO Continuation Dashboard ({run_name})", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig_path = fig_dir / f"task3_grpo_continuation_{run_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved GRPO dashboard to {fig_path}")


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)

    policy = bundle["policy"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    tokenizer = bundle["tokenizer"]
    prompts = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]

    num_updates = int(cfg["updates"])
    k_generations = int(cfg.get("num_generations", 4))
    eps = float(cfg.get("clip_epsilon", 0.20))
    beta = float(cfg.get("kl_beta", 0.10))
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    max_comp_len = int(cfg.get("max_completion_length", 512))
    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    mask_truncated = bool(cfg.get("mask_truncated_completions", True))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = repo_path("report/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / f"grpo_train_{run_name}.jsonl"

    print("=" * 65)
    print(f"[GRPO Continuation] run={run_name} updates={num_updates} K={k_generations} loss_type={loss_type}")
    print(f"Policy trainable params: {count_parameters(policy)[1]:,}")
    print(f"Output adapter:          {out}")
    print("=" * 65)

    timer = wall_timer()
    prompt_idx = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for update in range(1, num_updates + 1):
        # 1. Form batch of prompts repeated K times for group sampling
        batch_prompts = []
        group_ids_list = []
        for g_id in range(prompts_per_update):
            p_row = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1
            p_msgs = prompt_messages(p_row)
            for _ in range(k_generations):
                batch_prompts.append(p_msgs)
                group_ids_list.append(g_id)

        # 2. Rollout generation of K completions per prompt
        gen_data = batch_generate(
            policy,
            tokenizer,
            batch_prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_comp_len,
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
        truncated = gen_data["truncated"]

        # 3. Optional max-length completion masking
        token_mask = response_mask
        if mask_truncated:
            token_mask = mask_truncated_sequences(token_mask, truncated)

        # 4. Terminal reward evaluation
        with torch.no_grad():
            raw_rewards = score_reward_pairs(
                reward_model, reward_tokenizer, batch_prompts, responses, max_length=1024
            )

        # 5. Group-relative advantages
        device = sequences.device
        group_tensor = torch.tensor(group_ids_list, device=device, dtype=torch.long)
        advantages = group_relative_advantages(raw_rewards, group_tensor)

        # Compute within-group reward standard deviation and uninformative group rate
        group_stds = []
        for g_id in range(prompts_per_update):
            g_idx = (group_tensor == g_id).nonzero(as_tuple=True)[0]
            g_stds = raw_rewards[g_idx].std(unbiased=False).item()
            group_stds.append(g_stds)
        within_group_std = float(torch.tensor(group_stds).mean().item())
        uninformative_frac = float(sum(1 for s in group_stds if s < 1e-4) / max(len(group_stds), 1))

        # 6. Old logprobs and reference policy logprobs
        with torch.no_grad():
            old_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)

        # 7. Policy optimization step
        policy.train()
        new_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)

        loss, diag = grpo_policy_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            seq_adv=advantages,
            token_mask=token_mask,
            ref_logp=ref_logp,
            eps=eps,
            beta=beta,
            loss_type=loss_type,
            max_completion_length=max_comp_len,
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm).item()
        optimizer.step()

        # Update metrics
        mean_rew = float(raw_rewards.mean().item())
        mean_len = float(response_mask.sum(-1).mean().item())
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

        record = {
            "update": update,
            "reward_mean": mean_rew,
            "within_group_reward_std": within_group_std,
            "uninformative_group_fraction": uninformative_frac,
            "policy_loss": float(loss.item()),
            "sampled_kl": float(diag["sampled_kl"].item()),
            "clip_fraction": float(diag["clip_fraction"].item()),
            "entropy": float(diag["sample_entropy"].item()),
            "grad_norm": grad_norm,
            "response_length_mean": mean_len,
            "peak_vram_mb": peak_vram,
            "wall_time": float(timer()),
        }
        append_jsonl(log_path, record)

        if update % 2 == 0 or update == 1 or update == num_updates:
            print(
                f"[Update {update:2d}/{num_updates}] "
                f"R={mean_rew:+.3f} | σ_r={within_group_std:.3f} | "
                f"Uninf={uninformative_frac:.0%} | KL={diag['sampled_kl'].item():.4f} | "
                f"Loss={loss.item():+.4f} | Clip={diag['clip_fraction'].item():.2%} | "
                f"Len={mean_len:.0f}tok | VRAM={peak_vram:.0f}MB | Time={timer():.1f}s"
            )

    # Save continued adapter
    print(f"\n[GRPO] Saving continued policy adapter to {out}")
    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    summary = {
        "run_name": run_name,
        "updates": num_updates,
        "k_generations": k_generations,
        "loss_type": loss_type,
        "final_reward": mean_rew,
        "final_within_group_std": within_group_std,
        "final_uninformative_fraction": uninformative_frac,
        "final_kl": float(diag["sampled_kl"].item()),
        "final_loss": float(loss.item()),
        "peak_vram_mb": peak_vram,
        "wall_time_seconds": timer(),
        "adapter_path": str(out),
    }
    save_json(results_dir / f"grpo_summary_{run_name}.json", summary)

    # Generate plots
    try:
        plot_grpo_continuation(log_path, fig_dir, run_name)
    except Exception as e:
        print(f"[GRPO plot] Warning: plotting failed: {e}")

    print(f"[GRPO] Continuation complete in {timer():.1f}s.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
