from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, reset_file, save_json, set_seed, wall_timer
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import (
    count_parameters,
    disable_dropout,
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
    """Multi-panel PPO trajectory: every quantity the manual lists for the standard continuation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(l) for l in log_path.open(encoding="utf-8") if l.strip()]
    if not recs:
        return
    u = [r["update"] for r in recs]
    panels = [
        (["reward_rm", "reward_effective"], "Reward (RM / after EOS penalty)"),
        (["kl_token_mean"], "Sampled KL to reference (token mean)"),
        (["policy_loss"], "Policy loss (clipped surrogate)"),
        (["value_loss"], "Value loss (MSE)"),
        (["entropy_exact"], "Policy entropy (exact, nats/token)"),
        (["clip_fraction_last_epoch"], "Clip fraction (last PPO epoch)"),
        (["policy_grad_norm", "value_grad_norm"], "Grad norm (pre-clip)"),
        (["response_length"], "Response length (tokens)"),
        (["value_explained_variance"], "Critic explained variance (returns)"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    for ax, (keys, title) in zip(axes.flat, panels):
        for k, c in zip(keys, ["#2563EB", "#DC2626"]):
            if k in recs[0]:
                ax.plot(u, [r[k] for r in recs], "-o", ms=3, color=c, label=k)
        if len(keys) > 1:
            ax.legend(frameon=False, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("update")
    fig.suptitle(f"Task 2 — PPO continuation ({run_name}, ε={recs[0]['clip_epsilon']}, β_KL={recs[0]['kl_beta']})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"task2_ppo_continuation_{run_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved PPO dashboard for {run_name}")


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
    if bool(cfg.get("disable_dropout", True)):
        disable_dropout(policy)
        disable_dropout(value_model)
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


def explained_variance(values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.bool()
    v, r = values[m].float(), returns[m].float()
    var_r = r.var(unbiased=False)
    if r.numel() < 2 or var_r <= 0:
        return float("nan")
    return float(1.0 - (r - v).var(unbiased=False) / var_r)


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

    log_path = reset_file(results_dir / f"ppo_train_{run_name}.jsonl")
    rollout_path = reset_file(results_dir / f"ppo_rollouts_{run_name}.jsonl")

    print("=" * 60)
    print(f"[PPO] run={run_name} updates={num_updates} eps={eps} beta_kl={beta_kl} epochs={ppo_epochs}")
    print(f"Policy trainable params: {count_parameters(policy)[1]:,}")
    print(f"Value trainable params:  {count_parameters(value_model)[1]:,}")
    print(f"Output adapter:          {out}")
    print("=" * 60)

    timer = wall_timer()
    prompt_idx = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    generated_tokens = 0

    for update in range(1, num_updates + 1):
        # 1. Prompts (identical order for every fork: index 0, 1, 2, ...)
        batch_rows, batch_prompts = [], []
        for _ in range(prompts_per_update):
            row = prompts[prompt_idx % len(prompts)]
            batch_rows.append(row)
            batch_prompts.append(prompt_messages(row))
            prompt_idx += 1

        # 2. On-policy rollouts
        gen = batch_generate(
            policy, tokenizer, batch_prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_resp_len,
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            do_sample=bool(cfg["generation"]["do_sample"]),
        )
        sequences, attention_mask = gen["sequences"], gen["attention_mask"]
        prompt_width, response_ids = gen["prompt_width"], gen["response_ids"]
        response_mask = gen["response_mask"]
        generated_tokens += int(response_mask.sum().item())

        # 3. Terminal learned reward (+ released missing-EOS penalty)
        with torch.no_grad():
            rm_scores = score_reward_pairs(
                reward_model, reward_tokenizer, batch_prompts, gen["responses"], max_length=int(cfg["reward_max_length"])
            ).float()
            penalty = torch.tensor([0.0 if t else missing_eos_pen for t in gen["terminated_with_eos"]],
                                   device=rm_scores.device)
            task_reward = rm_scores - penalty

        # 4. Old-policy and reference log-probs (+ exact rollout entropy)
        with torch.no_grad():
            policy.eval()
            old_logp, entropy_tok = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids, return_entropy=True
            )
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)

        # 5. KL-shaped rewards, values, GAE
        shaped = shaped_rewards(task_reward, old_logp, ref_logp, response_mask, beta_kl)
        with torch.no_grad():
            value_model.eval()
            vals = token_values(value_model, sequences, attention_mask)
            resp_values = vals[:, prompt_width - 1 : prompt_width - 1 + response_ids.shape[1]].float()
            advantages, returns = compute_gae(shaped, resp_values, response_mask, gamma=gamma, lam=lam)
            norm_adv = normalize_advantages(advantages, response_mask)
            ev = explained_variance(resp_values, returns, response_mask)

        # 6. PPO epochs on this rollout batch
        policy.train()
        value_model.train()
        ep = []
        for epoch in range(ppo_epochs):
            new_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
            p_loss, ratio, diag = ppo_policy_loss(new_logp, old_logp, norm_adv, response_mask, eps=eps)
            opt_p.zero_grad()
            p_loss.backward()
            p_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm).item()
            opt_p.step()

            cur_vals = token_values(value_model, sequences, attention_mask)[
                :, prompt_width - 1 : prompt_width - 1 + response_ids.shape[1]
            ].float()
            v_loss = value_mse_loss(cur_vals, returns, response_mask)
            opt_v.zero_grad()
            (v_loss * val_coef).backward()
            v_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_grad_norm).item()
            opt_v.step()

            log_ratio = (new_logp - old_logp).detach()
            ep.append({
                "policy_loss": float(p_loss.item()),
                "value_loss": float(v_loss.item()),
                "clip_fraction": float(diag["clip_fraction"].item()),
                "clip_high_fraction": float(diag["clip_high_fraction"].item()),
                "clip_low_fraction": float(diag["clip_low_fraction"].item()),
                # k3 estimator of KL(pi_old || pi_new) at the start of this epoch
                "approx_kl_old_new": float(masked_mean(torch.exp(log_ratio) - 1 - log_ratio, response_mask).item()),
                "ratio_max": float(diag["ratio_max"].item()),
                "ratio_min": float(diag["ratio_min"].item()),
                "policy_grad_norm": float(p_norm),
                "value_grad_norm": float(v_norm),
            })

        last = ep[-1]
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        record = {
            "update": update,
            "prompt_id": str(batch_rows[0].get("prompt_id", batch_rows[0].get("source_index"))),
            "clip_epsilon": eps,
            "kl_beta": beta_kl,
            "reward_rm": float(rm_scores.mean().item()),
            "reward_effective": float(task_reward.mean().item()),
            "reward_mean": float(task_reward.mean().item()),
            "kl_token_mean": float(sampled_kl(old_logp, ref_logp, response_mask).item()),
            "kl_seq_sum": float(((old_logp - ref_logp) * response_mask).sum(-1).mean().item()),
            "policy_loss": float(np.mean([e["policy_loss"] for e in ep])),
            "value_loss": float(np.mean([e["value_loss"] for e in ep])),
            "clip_fraction": float(np.mean([e["clip_fraction"] for e in ep])),
            "clip_fraction_last_epoch": last["clip_fraction"],
            "approx_kl_old_new_last_epoch": last["approx_kl_old_new"],
            "ratio_max_last_epoch": last["ratio_max"],
            "ratio_min_last_epoch": last["ratio_min"],
            "entropy_exact": float(masked_mean(entropy_tok, response_mask).item()),
            "entropy_sampled": float(sample_entropy(old_logp, response_mask).item()),
            "policy_grad_norm": float(np.mean([e["policy_grad_norm"] for e in ep])),
            "value_grad_norm": float(np.mean([e["value_grad_norm"] for e in ep])),
            "response_length": float(response_mask.sum(-1).float().mean().item()),
            "terminated_with_eos": float(np.mean(gen["terminated_with_eos"])),
            "hit_max_tokens": float(np.mean(gen["truncated"])),
            "prompt_truncated": float(np.mean(gen["prompt_truncated"])),
            "value_mean": float(masked_mean(resp_values, response_mask).item()),
            "return_mean": float(masked_mean(returns, response_mask).item()),
            "value_explained_variance": ev,
            "value_first_token": float(resp_values[:, 0].mean().item()),
            "generated_tokens_cum": generated_tokens,
            "epochs": ep,
            "peak_vram_mb": peak_vram,
            "wall_time": float(timer()),
        }
        record["response_length_mean"] = record["response_length"]
        append_jsonl(log_path, record)
        append_jsonl(rollout_path, {
            "update": update, "prompt_id": record["prompt_id"], "prompt": batch_prompts[0][-1]["content"],
            "response": gen["responses"][0], "reward_rm": record["reward_rm"], "kl_token_mean": record["kl_token_mean"],
            "response_length": record["response_length"], "terminated_with_eos": bool(gen["terminated_with_eos"][0]),
        })

        print(
            f"[{update:2d}/{num_updates}] RM={record['reward_rm']:+.3f} KL={record['kl_token_mean']:.4f} "
            f"Lp={record['policy_loss']:+.4f} Lv={record['value_loss']:.3f} clip(ep{ppo_epochs})={last['clip_fraction']:.2%} "
            f"H={record['entropy_exact']:.3f} len={record['response_length']:.0f} EV={ev:.2f} "
            f"VRAM={peak_vram:.0f}MB t={timer():.0f}s", flush=True
        )

    print(f"\n[PPO] Saving continued policy adapter to {out}")
    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    recs = [json.loads(l) for l in log_path.open(encoding="utf-8") if l.strip()]
    summary = {
        "run_name": run_name,
        "updates": num_updates,
        "clip_epsilon": eps,
        "kl_beta": beta_kl,
        "ppo_epochs": ppo_epochs,
        "prompts_per_update": prompts_per_update,
        "disable_dropout": bool(cfg.get("disable_dropout", True)),
        "prompt_ids": [r["prompt_id"] for r in recs],
        "generated_tokens": generated_tokens,
        "final_reward": recs[-1]["reward_rm"],
        "final_kl": recs[-1]["kl_token_mean"],
        "mean_reward_last5": float(np.mean([r["reward_rm"] for r in recs[-5:]])),
        "mean_kl_last5": float(np.mean([r["kl_token_mean"] for r in recs[-5:]])),
        # Stability statistics (defined once, used for every fork):
        "stability_max_approx_kl_old_new": float(max(r["approx_kl_old_new_last_epoch"] for r in recs)),
        "stability_mean_clip_fraction_last_epoch": float(np.mean([r["clip_fraction_last_epoch"] for r in recs])),
        "stability_policy_grad_norm_cv": float(np.std([r["policy_grad_norm"] for r in recs]) /
                                               max(np.mean([r["policy_grad_norm"] for r in recs]), 1e-12)),
        "stability_max_ratio": float(max(r["ratio_max_last_epoch"] for r in recs)),
        "peak_vram_mb": recs[-1]["peak_vram_mb"],
        "wall_time_seconds": timer(),
        "adapter_path": str(out),
    }
    save_json(results_dir / f"ppo_summary_{run_name}.json", summary)
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
