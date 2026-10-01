from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.data import (
    load_yaml,
    prompt_messages_from_preference,
    preference_responses,
    read_jsonl,
    repo_path,
    render_prompt,
)
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import word_count, word_limit_compliance
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)
from common.generation import response_sequence_logprobs
from common.data import encode_prompt_response, pad_batch


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def plot_evaluation_results(results: dict, eval_name: str, fig_dir: Path):
    """Generate and save evaluation visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Reward distribution histogram ---
    if "per_example" in results and results["per_example"]:
        rewards = [ex.get("reward_score", 0) for ex in results["per_example"] if "reward_score" in ex]
        lengths = [ex.get("response_length_words", 0) for ex in results["per_example"] if "response_length_words" in ex]

        if rewards:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Reward distribution
            ax = axes[0]
            ax.hist(rewards, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
            ax.axvline(float(np.mean(rewards)), color="red", linestyle="--", linewidth=1.5,
                       label=f"Mean={np.mean(rewards):.3f}")
            ax.set_xlabel("Reward Score", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title(f"Reward Distribution — {eval_name}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Length distribution
            ax = axes[1]
            if lengths:
                ax.hist(lengths, bins=30, color="seagreen", edgecolor="white", alpha=0.85)
                ax.axvline(float(np.mean(lengths)), color="red", linestyle="--", linewidth=1.5,
                           label=f"Mean={np.mean(lengths):.1f}")
                ax.set_xlabel("Response Length (words)", fontsize=11)
                ax.set_ylabel("Count", fontsize=11)
                ax.set_title(f"Response Length Distribution — {eval_name}", fontsize=12, fontweight="bold")
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)

            fig.suptitle(f"Task 1 — DPO Evaluation: {eval_name}", fontsize=14, fontweight="bold", y=1.02)
            fig.tight_layout()
            fig.savefig(fig_dir / f"task1_dpo_eval_{eval_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # --- 2. Reward vs Length scatter ---
    if "per_example" in results and results["per_example"]:
        rs = [(ex.get("reward_score"), ex.get("response_length_words"))
              for ex in results["per_example"]
              if ex.get("reward_score") is not None and ex.get("response_length_words") is not None]
        if len(rs) > 5:
            rr, ll = zip(*rs)
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.scatter(ll, rr, alpha=0.5, s=20, color="steelblue")
            z = np.polyfit(list(ll), list(rr), 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(ll), max(ll), 100)
            ax2.plot(x_line, p(x_line), "r--", linewidth=1.5, label=f"Trend (slope={z[0]:.4f})")
            ax2.set_xlabel("Response Length (words)", fontsize=11)
            ax2.set_ylabel("Reward Score", fontsize=11)
            ax2.set_title(f"Reward vs Length — {eval_name}", fontsize=12, fontweight="bold")
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(fig_dir / f"task1_dpo_reward_vs_length_{eval_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig2)

    print(f"[plot] Evaluation figures saved to {fig_dir}")


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_dpo(config_path: str, adapter: str, name: str = "standard", gen_batch_size: int = 4):
    """Full evaluation: held-out preference accuracy, reward, KL, generation quality."""
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    model = bundle["policy"]
    tokenizer = bundle["tokenizer"]
    rm_model, rm_tokenizer = bundle["reward"]
    rows = bundle["rows"]

    set_seed(int(cfg["seed"]))
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = repo_path("report/figures")

    device = next(model.parameters()).device
    max_length = int(cfg["max_sequence_length"])

    print(f"[DPO eval] name={name} adapter={adapter} eval_rows={len(rows)}")

    # -----------------------------------------------------------------------
    # Part A: Held-out preference accuracy & KL
    # -----------------------------------------------------------------------
    from common.data import encode_prompt_response, pad_batch
    from common.generation import response_sequence_logprobs

    pref_correct = 0
    total_pairs = 0
    kl_sum = 0.0
    kl_count = 0

    eval_batch_size = int(cfg.get("batch_size", 2))
    for start in range(0, len(rows), eval_batch_size):
        chunk = rows[start : start + eval_batch_size]
        chosen_examples, rejected_examples = [], []
        for row in chunk:
            prompt_msgs = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen_examples.append(encode_prompt_response(tokenizer, prompt_msgs, yc, max_length))
            rejected_examples.append(encode_prompt_response(tokenizer, prompt_msgs, yr, max_length))

        chosen_batch = pad_batch(tokenizer, chosen_examples)
        rejected_batch = pad_batch(tokenizer, rejected_examples)
        chosen_batch = {k: v.to(device) for k, v in chosen_batch.items()}
        rejected_batch = {k: v.to(device) for k, v in rejected_batch.items()}

        with torch.no_grad():
            # Policy log-probs
            policy_chosen, _, _ = response_sequence_logprobs(model, chosen_batch)
            policy_rejected, _, _ = response_sequence_logprobs(model, rejected_batch)

            # Reference log-probs
            with reference_mode(model):
                ref_chosen, _, _ = response_sequence_logprobs(model, chosen_batch)
                ref_rejected, _, _ = response_sequence_logprobs(model, rejected_batch)

        # Preference accuracy: does the model prefer chosen over rejected?
        margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
        pref_correct += int((margin > 0).sum().item())
        total_pairs += len(chunk)

        # KL(π_θ || π_ref) on chosen
        kl_batch = (policy_chosen - ref_chosen).sum().item()
        kl_sum += kl_batch
        kl_count += len(chunk)

    held_out_pref_acc = pref_correct / max(total_pairs, 1)
    mean_kl = kl_sum / max(kl_count, 1)

    print(f"  Held-out preference accuracy: {held_out_pref_acc:.4f}")
    print(f"  Mean KL(policy || ref):        {mean_kl:.4f}")

    # -----------------------------------------------------------------------
    # Part B: Generate responses and score with reward model
    # -----------------------------------------------------------------------
    per_example = []
    max_new_tokens = int(cfg.get("eval_max_new_tokens", 256))
    num_eval_gen = min(len(rows), int(cfg.get("eval_generation_limit", 100)))

    print(f"  Generating {num_eval_gen} responses for reward evaluation...")
    for start in range(0, num_eval_gen, gen_batch_size):
        chunk = rows[start : start + gen_batch_size]
        prompts = [prompt_messages_from_preference(r) for r in chunk]

        gen_out = batch_generate(
            model, tokenizer, prompts,
            max_prompt_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        # Score with reward model
        rewards = score_reward_pairs(rm_model, rm_tokenizer, prompts, gen_out["responses"], max_length=max_length)

        for i, (row, resp, rew) in enumerate(zip(chunk, gen_out["responses"], rewards)):
            prompt_text = ""
            prompt_msgs = prompt_messages_from_preference(row)
            if prompt_msgs:
                prompt_text = prompt_msgs[-1].get("content", "")

            wc = word_count(resp)
            wl_comp = word_limit_compliance(prompt_text, resp)

            per_example.append({
                "index": start + i,
                "prompt": prompt_text[:200],
                "response": resp[:500],
                "reward_score": float(rew.item()),
                "response_length_words": wc,
                "response_length_tokens": int(gen_out["response_lengths"][i]),
                "terminated_with_eos": gen_out["terminated_with_eos"][i],
                "word_limit_compliance": wl_comp,
            })

    # Aggregate stats
    rewards_all = [ex["reward_score"] for ex in per_example]
    lengths_all = [ex["response_length_words"] for ex in per_example]
    import numpy as np
    mean_reward = float(np.mean(rewards_all)) if rewards_all else 0.0
    std_reward = float(np.std(rewards_all)) if rewards_all else 0.0
    mean_length = float(np.mean(lengths_all)) if lengths_all else 0.0

    wl_scores = [ex["word_limit_compliance"] for ex in per_example if ex["word_limit_compliance"] is not None]
    wl_rate = float(np.mean(wl_scores)) if wl_scores else None

    results = {
        "eval_name": name,
        "adapter": adapter,
        "held_out_preference_accuracy": held_out_pref_acc,
        "mean_kl_from_reference": mean_kl,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_response_length_words": mean_length,
        "num_evaluated": len(per_example),
        "word_limit_compliance_rate": wl_rate,
        "per_example": per_example,
    }

    # Save
    save_json(results_dir / f"dpo_eval_{name}.json", results)
    print(f"  Mean reward:    {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"  Mean length:    {mean_length:.1f} words")
    if wl_rate is not None:
        print(f"  Word-limit compliance: {wl_rate:.3f}")

    # Qualitative examples
    sorted_by_reward = sorted(per_example, key=lambda x: x["reward_score"], reverse=True)
    print("\n  --- Top 3 by reward ---")
    for ex in sorted_by_reward[:3]:
        print(f"    R={ex['reward_score']:.3f} | {ex['response'][:100]}...")
    print("  --- Bottom 3 by reward ---")
    for ex in sorted_by_reward[-3:]:
        print(f"    R={ex['reward_score']:.3f} | {ex['response'][:100]}...")

    # Generate plots
    try:
        plot_evaluation_results(results, name, fig_dir)
    except Exception as e:
        print(f"[plot] Warning: could not generate plots: {e}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--gen-batch-size", type=int, default=4)
    args = ap.parse_args()
    evaluate_dpo(args.config, args.adapter, args.name, args.gen_batch_size)


if __name__ == "__main__":
    main()
