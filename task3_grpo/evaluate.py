from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import sampled_kl, word_count
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode


def plot_grpo_evaluation(results: dict, fig_dir: Path, name: str):
    """Plot evaluation reward distribution and length statistics for GRPO."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    examples = results.get("per_example", [])
    if not examples:
        return

    rewards = [e["reward_score"] for e in examples]
    lengths = [e["response_length_tokens"] for e in examples]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reward distribution
    axes[0].hist(rewards, bins=25, color="tab:blue", edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(rewards), color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean = {np.mean(rewards):.3f}")
    axes[0].set_xlabel("Reward Model Score", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("GRPO Held-out Reward Distribution", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Length distribution
    axes[1].hist(lengths, bins=25, color="tab:green", edgecolor="white", alpha=0.85)
    axes[1].axvline(np.mean(lengths), color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean = {np.mean(lengths):.1f} tokens")
    axes[1].set_xlabel("Response Length (tokens)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("GRPO Response Length Distribution", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Task 3 — GRPO Held-Out Evaluation: {name}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig_path = fig_dir / f"task3_grpo_eval_{name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved GRPO eval figure to {fig_path}")


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_grpo(config_path: str, adapter: str, name: str = "standard", batch_size: int = 4):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model, reward_tokenizer = bundle["reward"]

    set_seed(int(cfg["seed"]))
    results_dir = repo_path(cfg["results_dir"])
    fig_dir = repo_path("report/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_resp_len = int(cfg.get("max_completion_length", 512))

    print(f"[GRPO Eval] Evaluating adapter '{adapter}' on {len(rows)} held-out prompts...")

    per_example = []
    all_rewards = []
    all_kls = []
    all_lengths = []
    terminated_count = 0

    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [prompt_messages(r) for r in chunk]

        gen = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_resp_len,
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            do_sample=bool(cfg["generation"]["do_sample"]),
        )

        with torch.no_grad():
            rews = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                prompts,
                gen["responses"],
                max_length=1024,
            )
            # Logprobs for KL estimation
            pol_logp, _ = response_token_logprobs(
                policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"]
            )
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"]
                )

        for i, (r, resp, rew, term, rlen) in enumerate(
            zip(chunk, gen["responses"], rews, gen["terminated_with_eos"], gen["response_lengths"])
        ):
            r_mask = gen["response_mask"][i : i + 1]
            kl_val = float(sampled_kl(pol_logp[i : i + 1], ref_logp[i : i + 1], r_mask).item())

            p_text = prompt_messages(r)[-1].get("content", "")
            rew_val = float(rew.item())
            all_rewards.append(rew_val)
            all_kls.append(kl_val)
            all_lengths.append(rlen)
            if term:
                terminated_count += 1

            per_example.append({
                "prompt": p_text[:200],
                "response": resp[:500],
                "reward_score": rew_val,
                "kl_divergence": kl_val,
                "response_length_tokens": int(rlen),
                "response_length_words": word_count(resp),
                "terminated_with_eos": bool(term),
            })

    results = {
        "eval_name": name,
        "adapter": adapter,
        "num_evaluated": len(per_example),
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_kl": float(np.mean(all_kls)),
        "std_kl": float(np.std(all_kls)),
        "mean_response_length_tokens": float(np.mean(all_lengths)),
        "eos_termination_rate": terminated_count / max(len(per_example), 1),
        "per_example": per_example,
    }

    out_file = results_dir / f"grpo_eval_{name}.json"
    save_json(out_file, results)
    print(f"\n[GRPO Eval] Complete. Saved results to {out_file}")
    print(f"  Mean Reward:     {results['mean_reward']:+.3f} ± {results['std_reward']:.3f}")
    print(f"  Mean KL:         {results['mean_kl']:.4f}")
    print(f"  Mean Length:     {results['mean_response_length_tokens']:.1f} tokens")
    print(f"  EOS Rate:        {results['eos_termination_rate']:.1%}")

    try:
        plot_grpo_evaluation(results, fig_dir, name)
    except Exception as e:
        print(f"[GRPO Eval] Warning: plotting failed: {e}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    evaluate_grpo(args.config, args.adapter, args.name, args.batch_size)


if __name__ == "__main__":
    main()
