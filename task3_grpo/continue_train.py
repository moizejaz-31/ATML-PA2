from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs, single_sequence
from common.logging_utils import append_jsonl, reset_file, save_json, set_seed, wall_timer
from common.models import (
    count_parameters,
    disable_dropout,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task3_grpo.grpo import (
    group_relative_advantages,
    grpo_policy_loss,
    mask_truncated_sequences,
)

from common.metrics import GRPO_ZERO_STD_TOL as ZERO_STD_TOL


def plot_grpo_continuation(log_path: Path, fig_dir: Path, run_name: str = "standard"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(l) for l in log_path.open(encoding="utf-8") if l.strip()]
    if not recs:
        return
    u = [r["update"] for r in recs]
    panels = [
        ("reward_mean", "Mean RM reward (K completions)"),
        ("kl_token_mean", "Sampled KL to reference (token mean)"),
        ("within_group_reward_std", "Within-group reward std"),
        ("uninformative_group_fraction", "Uninformative-group fraction"),
        ("policy_loss", "Policy loss"),
        ("grad_norm", "Grad norm (pre-clip)"),
        ("entropy_exact", "Policy entropy (exact)"),
        ("response_length", "Mean completion length (tokens)"),
        ("masked_fraction", "Completions masked (hit cap)"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    for ax, (k, t) in zip(axes.flat, panels):
        ax.plot(u, [r[k] for r in recs], "-o", ms=3, color="#2563EB")
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("update")
    fig.suptitle(f"Task 3 — GRPO continuation ({run_name}, loss={recs[0]['loss_type']}, K={recs[0]['k']})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"task3_grpo_continuation_{run_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved GRPO dashboard for {run_name}")


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    if bool(cfg.get("disable_dropout", True)):
        disable_dropout(policy)
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


def _grads(params):
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]


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
    params = trainable_parameters(policy)

    num_updates = int(cfg["updates"])
    K = int(cfg.get("num_generations", 4))
    eps = float(cfg.get("clip_epsilon", 0.20))
    beta = float(cfg.get("kl_beta", 0.10))
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    policy_epochs = int(cfg.get("policy_epochs", 1))
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
    log_path = reset_file(results_dir / f"grpo_train_{run_name}.jsonl")
    rollout_path = reset_file(results_dir / f"grpo_rollouts_{run_name}.jsonl")

    print("=" * 60)
    print(f"[GRPO] run={run_name} updates={num_updates} K={K} loss_type={loss_type} eps={eps} beta={beta}")
    print(f"Policy trainable params: {count_parameters(policy)[1]:,}")
    print("=" * 60)

    timer = wall_timer()
    prompt_idx = 0
    generated_tokens = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for update in range(1, num_updates + 1):
        # 1. Prompts repeated K times (identical prompt order for every run/fork)
        batch_prompts, group_ids, prompt_ids_used = [], [], []
        for g in range(prompts_per_update):
            row = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1
            prompt_ids_used.append(str(row.get("prompt_id", row.get("source_index"))))
            for _ in range(K):
                batch_prompts.append(prompt_messages(row))
                group_ids.append(g)
        n_seq = len(batch_prompts)

        # 2. Group rollouts
        gen = batch_generate(
            policy, tokenizer, batch_prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_comp_len,
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            do_sample=bool(cfg["generation"]["do_sample"]),
        )
        generated_tokens += int(gen["response_mask"].sum().item())
        token_mask_full = gen["response_mask"]
        if mask_truncated:
            token_mask_full = mask_truncated_sequences(token_mask_full, gen["truncated"])
        keep = (token_mask_full.sum(-1) > 0).tolist()

        # 3. Rewards and group-relative advantages
        with torch.no_grad():
            rewards = score_reward_pairs(reward_model, reward_tokenizer, batch_prompts, gen["responses"],
                                         max_length=int(cfg.get("reward_max_length", 1280))).float()
        gt = torch.tensor(group_ids, device=rewards.device)
        advantages = group_relative_advantages(rewards, gt)
        group_std = [float(rewards[gt == g].std(unbiased=False).item()) for g in range(prompts_per_update)]
        uninformative = float(np.mean([s <= ZERO_STD_TOL for s in group_std]))

        # 4. Per-completion old / reference log-probs (padding stripped)
        seqs = [single_sequence(gen, i) for i in range(n_seq)]
        old_lp, ref_lp, ent = [], [], []
        with torch.no_grad():
            policy.eval()
            for seq, attn, pw, rids in seqs:
                lp, h = response_token_logprobs(policy, seq, attn, pw, rids, return_entropy=True)
                with reference_mode(policy):
                    rl, _ = response_token_logprobs(policy, seq, attn, pw, rids)
                old_lp.append(lp[0])
                ref_lp.append(rl[0])
                ent.append(h[0])

        # 5. Policy optimisation. The batch loss of grpo_policy_loss decomposes exactly over completions:
        #      L = sum_k [ policy_term_k / n_seq  +  beta * sum_t k3_kt * m_kt / N_valid ],
        #    so each completion is forwarded/backpropagated on its own (same gradient, lower memory) and
        #    the size of its gradient contribution is recorded against its length.
        n_valid = float(token_mask_full.sum().item())
        policy.train()
        seq_stats = [dict() for _ in range(n_seq)]
        for epoch in range(policy_epochs):
            optimizer.zero_grad()
            pol_total, kl_num, clip_num, ratio_sum = 0.0, 0.0, 0.0, 0.0
            for i, (seq, attn, pw, rids) in enumerate(seqs):
                m = token_mask_full[i, : rids.shape[1]].float()
                st = seq_stats[i]
                st.update({"length": int(rids.shape[1]), "reward": float(rewards[i]), "advantage": float(advantages[i]),
                           "masked": not keep[i], "hit_cap": bool(gen["truncated"][i])})
                if not keep[i]:
                    st.update({"grad_norm": 0.0, "token_weight": 0.0, "sequence_weight": 0.0})
                    continue
                new, _ = response_token_logprobs(policy, seq, attn, pw, rids)
                single_loss, diag = grpo_policy_loss(
                    new_logp=new, old_logp=old_lp[i][None], seq_adv=advantages[i : i + 1],
                    token_mask=m[None], ref_logp=ref_lp[i][None], eps=eps, beta=0.0,
                    loss_type=loss_type, max_completion_length=max_comp_len,
                )
                policy_part = single_loss / n_seq
                lr_ = ref_lp[i][None] - new                                  # same k3 estimator as grpo_policy_loss
                k3 = torch.exp(lr_) - lr_ - 1.0
                kl_part = beta * (k3 * m[None]).sum() / max(n_valid, 1.0)
                before = _grads(params)
                (policy_part + kl_part).backward()
                after = _grads(params)
                gnorm = float(torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(after, before))).item())
                denom = float(m.sum()) if loss_type == "grpo" else float(max_comp_len)
                st.update({
                    "grad_norm": gnorm,
                    "token_weight": abs(float(advantages[i])) / max(denom, 1.0) / n_seq,
                    "sequence_weight": abs(float(advantages[i])) * float(m.sum()) / max(denom, 1.0) / n_seq,
                })
                pol_total += float(policy_part.item())
                kl_num += float((k3 * m[None]).sum().item())
                clip_num += float(diag["clip_fraction"].item()) * float(m.sum())
                ratio_sum += float(diag["ratio_mean"].item()) * float(m.sum())
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm).item()
            optimizer.step()

        mask_all = gen["response_mask"]
        kl_tok = sum(float(((o - r)).sum()) for o, r in zip(old_lp, ref_lp)) / max(float(mask_all.sum()), 1.0)
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        record = {
            "update": update,
            "prompt_ids": prompt_ids_used,
            "loss_type": loss_type,
            "k": K,
            "reward_mean": float(rewards.mean()),
            "reward_std_all": float(rewards.std(unbiased=False)),
            "within_group_reward_std": float(np.mean(group_std)),
            "uninformative_group_fraction": uninformative,
            "policy_loss": pol_total,
            "kl_k3_token_mean": kl_num / max(n_valid, 1.0),
            "kl_token_mean": kl_tok,
            "sampled_kl": kl_tok,
            "clip_fraction": clip_num / max(n_valid, 1.0),
            "ratio_mean": ratio_sum / max(n_valid, 1.0),
            "entropy_exact": float(torch.cat(ent).mean()),
            "entropy": float(torch.cat([-o for o in old_lp]).mean()),
            "grad_norm": grad_norm,
            "response_length": float(mask_all.sum(-1).float().mean()),
            "response_length_mean": float(mask_all.sum(-1).float().mean()),
            "masked_fraction": float(np.mean([not k for k in keep])),
            "prompt_truncated": float(np.mean(gen["prompt_truncated"])),
            "length_reward_corr": float(np.corrcoef(rewards.cpu().numpy(), mask_all.sum(-1).cpu().numpy())[0, 1])
            if n_seq > 2 and float(rewards.std()) > 0 and float(mask_all.sum(-1).float().std()) > 0 else float("nan"),
            "sequences": seq_stats,
            "generated_tokens_cum": generated_tokens,
            "peak_vram_mb": peak_vram,
            "wall_time": float(timer()),
        }
        append_jsonl(log_path, record)
        for i in range(n_seq):
            append_jsonl(rollout_path, {"update": update, "prompt_id": prompt_ids_used[group_ids[i]],
                                        "prompt": batch_prompts[i][-1]["content"], "completion": gen["responses"][i],
                                        **seq_stats[i]})
        print(
            f"[{update:2d}/{num_updates}] R={record['reward_mean']:+.3f} σ_g={record['within_group_reward_std']:.3f} "
            f"uninf={uninformative:.0%} KL={kl_tok:.4f} L={pol_total:+.4f} clip={record['clip_fraction']:.2%} "
            f"gn={grad_norm:.3f} len={record['response_length']:.0f} masked={record['masked_fraction']:.0%} "
            f"VRAM={peak_vram:.0f}MB t={timer():.0f}s", flush=True
        )

    print(f"\n[GRPO] Saving continued policy adapter to {out}")
    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    recs = [json.loads(l) for l in log_path.open(encoding="utf-8") if l.strip()]
    summary = {
        "run_name": run_name,
        "updates": num_updates,
        "k_generations": K,
        "loss_type": loss_type,
        "clip_epsilon": eps,
        "kl_beta": beta,
        "policy_epochs": policy_epochs,
        "disable_dropout": bool(cfg.get("disable_dropout", True)),
        "prompt_ids": [p for r in recs for p in r["prompt_ids"]],
        "generated_tokens": generated_tokens,
        "mean_reward": float(np.mean([r["reward_mean"] for r in recs])),
        "mean_within_group_std": float(np.mean([r["within_group_reward_std"] for r in recs])),
        "mean_uninformative_fraction": float(np.mean([r["uninformative_group_fraction"] for r in recs])),
        "mean_masked_fraction": float(np.mean([r["masked_fraction"] for r in recs])),
        "final_kl": recs[-1]["kl_token_mean"],
        "peak_vram_mb": recs[-1]["peak_vram_mb"],
        "wall_time_seconds": timer(),
        "adapter_path": str(out),
    }
    save_json(results_dir / f"grpo_summary_{run_name}.json", summary)
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
