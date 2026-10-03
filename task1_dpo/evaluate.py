from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import load_json, save_json, set_seed
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from common.policy_eval import generate_and_score
from task1_dpo.preprocess import load_filtered_pairs, pair_id, stratum_of

NO_ADAPTER = {None, "", "none", "sft", "base"}


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def plot_evaluation_results(results: dict, eval_name: str, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ex = results.get("generations", [])
    if not ex:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    rewards = np.array([e["reward_score"] for e in ex])
    lengths = np.array([e["response_length_tokens"] for e in ex])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].hist(rewards, bins=30, color="#2563EB", alpha=0.85)
    axes[0].axvline(rewards.mean(), color="red", ls="--", label=f"mean={rewards.mean():.3f}")
    axes[0].set_xlabel("RM score")
    axes[0].legend(frameon=False)
    axes[1].hist(lengths, bins=30, color="#16A34A", alpha=0.85)
    axes[1].axvline(lengths.mean(), color="red", ls="--", label=f"mean={lengths.mean():.1f}")
    axes[1].set_xlabel("response tokens")
    axes[1].legend(frameon=False)
    axes[2].scatter(lengths, rewards, s=12, alpha=0.5, color="#2563EB")
    if len(ex) > 5 and lengths.std() > 0:
        z = np.polyfit(lengths, rewards, 1)
        xs = np.linspace(lengths.min(), lengths.max(), 50)
        axes[2].plot(xs, np.poly1d(z)(xs), "r--", label=f"slope={z[0]:.4f}/token")
        axes[2].legend(frameon=False)
    axes[2].set_xlabel("response tokens")
    axes[2].set_ylabel("RM score")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Task 1 — DPO held-out generations: {eval_name}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"task1_dpo_eval_{eval_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Held-out preference metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def preference_metrics(model, tokenizer, rows: list[dict], max_length: int, beta: float, batch_size: int = 2):
    """Held-out DPO loss / preference accuracy. margin = [log pi/pi_ref](y+) - [log pi/pi_ref](y-),
    sequence log-probs summed over response tokens only (manual definition)."""
    device = next(model.parameters()).device
    per_pair = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        ch, rj, infos = [], [], []
        for row in chunk:
            msgs = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            ic, mc, info_c = encode_prompt_response(tokenizer, msgs, yc, max_length, return_info=True)
            ir, mr, info_r = encode_prompt_response(tokenizer, msgs, yr, max_length, return_info=True)
            ch.append((ic, mc))
            rj.append((ir, mr))
            infos.append((info_c, info_r))
        cb = {k: v.to(device) for k, v in pad_batch(tokenizer, ch).items()}
        rb = {k: v.to(device) for k, v in pad_batch(tokenizer, rj).items()}
        pc, _, _ = response_sequence_logprobs(model, cb)
        pr, _, _ = response_sequence_logprobs(model, rb)
        with reference_mode(model):
            rc, _, _ = response_sequence_logprobs(model, cb)
            rr, _, _ = response_sequence_logprobs(model, rb)
        margin = ((pc - rc) - (pr - rr)).float()
        loss = -torch.nn.functional.logsigmoid(beta * margin)
        for row, m, l, lc, lr_, (info_c, info_r) in zip(
            chunk, margin.cpu().tolist(), loss.cpu().tolist(), (pc - rc).cpu().tolist(), (pr - rr).cpu().tolist(), infos
        ):
            per_pair.append({
                "prompt_id": pair_id(row),
                "stratum": stratum_of(row),
                "margin": m,
                "dpo_loss": l,
                "logratio_chosen": lc,
                "logratio_rejected": lr_,
                "chosen_tokens_full": info_c["response_tokens_full"],
                "rejected_tokens_full": info_r["response_tokens_full"],
                "chosen_tokens_kept": info_c["response_tokens_kept"],
                "rejected_tokens_kept": info_r["response_tokens_kept"],
                "any_response_truncated": bool(info_c["response_truncated"] or info_r["response_truncated"]),
            })
    margins = np.array([p["margin"] for p in per_pair])
    return {
        "num_pairs": len(per_pair),
        "beta_for_loss": beta,
        "heldout_dpo_loss": float(np.mean([p["dpo_loss"] for p in per_pair])),
        "heldout_preference_accuracy": float((margins > 0).mean()),
        "heldout_tie_fraction": float((margins == 0).mean()),
        "mean_margin": float(margins.mean()),
        "mean_implicit_reward_margin": float(beta * margins.mean()),
    }, per_pair


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def resolve_beta(cfg: dict, name: str, beta: float | None) -> float:
    if beta is not None:
        return float(beta)
    summary = repo_path(cfg["results_dir"]) / f"dpo_summary_{name.replace('ablation_', '')}.json"
    if summary.exists():
        return float(load_json(summary)["beta"])
    return float(cfg["beta"])


def evaluate_dpo(config_path: str, adapter: str | None, name: str = "standard", beta: float | None = None,
                 gen_batch_size: int | None = None):
    """Held-out DPO loss / preference accuracy + the common generation protocol (RM score,
    sampled-response KL, entropy, length). `adapter=None` evaluates the untouched SFT policy."""
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    beta = resolve_beta(cfg, name, beta)
    tokenizer = load_tokenizer(cfg["base_model"])
    use_adapter = adapter not in NO_ADAPTER
    model = load_policy(cfg, adapter_path=adapter if use_adapter else None, trainable=False)
    rows, data_info = load_filtered_pairs(cfg, tokenizer, cfg["paths"]["dpo_standard_eval"])
    max_length = int(cfg["max_sequence_length"])
    results_dir = repo_path(cfg["results_dir"])
    print(f"[DPO eval] name={name} adapter={adapter if use_adapter else 'none (SFT)'} beta={beta} "
          f"pairs={len(rows)} (dropped {data_info['dropped_overlength']} overlength)")

    pref, per_pair = preference_metrics(model, tokenizer, rows, max_length, beta, int(cfg.get("batch_size", 2)))
    print(f"  held-out DPO loss={pref['heldout_dpo_loss']:.4f} pref_acc={pref['heldout_preference_accuracy']:.4f}")

    n_gen = min(len(rows), int(cfg.get("eval_generation_prompts", 200)))
    gen_rows = rows[:n_gen]
    reward_model, reward_tokenizer = load_reward_model(cfg)
    gen_summary, generations = generate_and_score(
        model, tokenizer, reward_model, reward_tokenizer,
        [prompt_messages_from_preference(r) for r in gen_rows],
        [pair_id(r) for r in gen_rows],
        max_prompt_length=max_length,
        max_new_tokens=int(cfg.get("max_generation_tokens", 256)),
        reward_max_length=int(cfg.get("reward_max_length", 1280)),
        generation=cfg["generation"],
        seed=int(cfg["seed"]),
        batch_size=int(gen_batch_size or cfg.get("eval_batch_size", 8)),
        has_reference=use_adapter,
    )
    clear_gpu(reward_model)

    results = {
        "eval_name": name,
        "adapter": adapter if use_adapter else None,
        "beta": beta,
        "data": data_info,
        **pref,
        **gen_summary,
        # Backwards-compatible aliases used by older plotting code.
        "held_out_preference_accuracy": pref["heldout_preference_accuracy"],
        "mean_kl_from_reference": gen_summary["kl_token_mean"],
        "mean_response_length_tokens": gen_summary["length_tokens"]["mean"],
        "mean_response_length_words": gen_summary["length_words"]["mean"],
        "per_pair": per_pair,
        "generations": generations,
    }
    save_json(results_dir / f"dpo_eval_{name}.json", results)
    print(f"  RM={gen_summary['mean_reward']:.4f}±{gen_summary['std_reward']:.4f} "
          f"KL(token)={gen_summary['kl_token_mean']:.4f} KL(seq)={gen_summary['kl_sequence_mean']:.3f} "
          f"len={gen_summary['length_tokens']['mean']:.1f}±{gen_summary['length_tokens']['std']:.1f} tok "
          f"EOS={gen_summary['eos_rate']:.2f}")
    try:
        plot_evaluation_results(results, name, repo_path("report/figures"))
    except Exception as e:
        print(f"[plot] Warning: could not generate plots: {e}")
    clear_gpu(model)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the untouched SFT policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used for the held-out DPO loss (default: the run's training beta)")
    ap.add_argument("--gen-batch-size", type=int)
    args = ap.parse_args()
    evaluate_dpo(args.config, args.adapter, args.name, args.beta, args.gen_batch_size)


if __name__ == "__main__":
    main()
