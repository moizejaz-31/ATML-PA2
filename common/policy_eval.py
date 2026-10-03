"""Common held-out generation protocol shared by the DPO, PPO and GRPO evaluations.

For every condition the same fixed prompt list, decoding config, cap and seed are used, and:
  * reward      = fixed course RM score of the generated response (RM input left-truncated);
  * KL          = released sampled-response estimator: log pi_theta - log pi_ref on the policy's OWN
                  generated tokens, aggregated as one token-level mean over the whole evaluation set
                  (= common.metrics.sampled_kl over all tokens). The per-response summed log-ratio is
                  also stored as a secondary, sequence-level view;
  * entropy     = exact mean token-level policy entropy over valid response tokens (plus the sampled
                  -log pi estimate from common.metrics.sample_entropy);
  * length      = generated response tokens (excluding padding), mean/std/IQR.
"""

from __future__ import annotations

import numpy as np
import torch

from common.generation import batch_generate, response_token_logprobs, score_reward_pairs, single_sequence
from common.logging_utils import set_seed
from common.metrics import TokenKLAccumulator, length_stats, word_count
from common.models import reference_mode


@torch.no_grad()
def generate_and_score(
    model,
    tokenizer,
    reward_model,
    reward_tokenizer,
    prompts: list[list[dict]],
    prompt_ids: list[str],
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    reward_max_length: int,
    generation: dict,
    seed: int,
    batch_size: int = 8,
    has_reference: bool = True,
):
    set_seed(seed)
    model.eval()
    kl = TokenKLAccumulator()
    ent_sum, ent_samp_sum, ent_count = 0.0, 0.0, 0.0
    per_example = []

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        ids_chunk = prompt_ids[start : start + batch_size]
        gen = batch_generate(
            model,
            tokenizer,
            chunk,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=float(generation.get("temperature", 0.7)),
            top_p=float(generation.get("top_p", 0.9)),
            do_sample=bool(generation.get("do_sample", True)),
        )
        rewards = score_reward_pairs(reward_model, reward_tokenizer, chunk, gen["responses"], max_length=reward_max_length)
        # Log-probs one sample at a time (padding stripped): a batched full-vocabulary log-softmax
        # over 8 x ~1000 tokens would not fit on a 16 GB GPU.
        seq_kl, tok_kl = [], []
        for i in range(len(chunk)):
            seq, attn, pw, rids = single_sequence(gen, i)
            pol_logp, entropy = response_token_logprobs(model, seq, attn, pw, rids, return_entropy=True)
            if has_reference:
                with reference_mode(model):
                    ref_logp, _ = response_token_logprobs(model, seq, attn, pw, rids)
            else:
                ref_logp = pol_logp
            mask = torch.ones_like(pol_logp)
            seq_kl.extend(kl.add(pol_logp, ref_logp, mask))
            tok_kl.append(float((pol_logp - ref_logp).mean().item()))
            ent_sum += float(entropy.sum().item())
            ent_samp_sum += float((-pol_logp).sum().item())
            ent_count += float(mask.sum().item())

        for i, (pid, msgs, resp) in enumerate(zip(ids_chunk, chunk, gen["responses"])):
            per_example.append({
                "prompt_id": pid,
                "prompt": str(msgs[-1].get("content", "")),
                "response": resp,
                "reward_score": float(rewards[i].item()),
                "kl_seq_sum": float(seq_kl[i]),
                "kl_token_mean": float(tok_kl[i]),
                "response_length_tokens": int(gen["response_lengths"][i]),
                "response_length_words": word_count(resp),
                "terminated_with_eos": bool(gen["terminated_with_eos"][i]),
                "hit_max_tokens": bool(gen["truncated"][i]),
                "prompt_truncated": bool(gen["prompt_truncated"][i]),
            })
        print(f"    generated {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)

    rewards = np.asarray([e["reward_score"] for e in per_example], float)
    summary = {
        "num_prompts": len(per_example),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "sem_reward": float(rewards.std(ddof=1) / np.sqrt(len(rewards))) if len(rewards) > 1 else float("nan"),
        "kl_token_mean": kl.token_mean(),
        "kl_sequence_mean": kl.sequence_mean(),
        "entropy_exact": ent_sum / max(ent_count, 1.0),
        "entropy_sampled": ent_samp_sum / max(ent_count, 1.0),
        "length_tokens": length_stats(e["response_length_tokens"] for e in per_example),
        "length_words": length_stats(e["response_length_words"] for e in per_example),
        "eos_rate": float(np.mean([e["terminated_with_eos"] for e in per_example])),
        "hit_max_tokens_rate": float(np.mean([e["hit_max_tokens"] for e in per_example])),
        "prompt_truncated_count": int(sum(e["prompt_truncated"] for e in per_example)),
        "reward_length_corr": float(np.corrcoef(rewards, [e["response_length_tokens"] for e in per_example])[0, 1])
        if len(rewards) > 2 else float("nan"),
        "protocol": {
            "max_prompt_length": max_prompt_length,
            "max_new_tokens": max_new_tokens,
            "reward_max_length": reward_max_length,
            "generation": generation,
            "seed": seed,
            "batch_size": batch_size,
        },
    }
    return summary, per_example


def plot_rl_eval(results: dict, fig_path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ex = results.get("generations", [])
    if not ex:
        return
    rewards = np.array([e["reward_score"] for e in ex])
    lengths = np.array([e["response_length_tokens"] for e in ex])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(rewards, bins=25, color="#2563EB", alpha=0.85)
    axes[0].axvline(rewards.mean(), color="red", ls="--", label=f"mean={rewards.mean():.3f}")
    axes[0].set_xlabel("RM score")
    axes[1].hist(lengths, bins=25, color="#16A34A", alpha=0.85)
    axes[1].axvline(lengths.mean(), color="red", ls="--", label=f"mean={lengths.mean():.1f}")
    axes[1].set_xlabel("response tokens")
    axes[2].scatter(lengths, rewards, s=12, alpha=0.6, c=["#DC2626" if e["hit_max_tokens"] else "#2563EB" for e in ex])
    axes[2].set_xlabel("response tokens (red = hit cap)")
    axes[2].set_ylabel("RM score")
    for ax in axes[:2]:
        ax.legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_rl_policy(cfg: dict, adapter: str | None, name: str, out_json, fig_path, max_new_tokens: int, title: str):
    """Held-out evaluation for PPO/GRPO-family adapters on the fixed first `eval_num_prompts` prompts
    of the held-out RL prompt pool. `adapter=None` evaluates the untouched SFT policy."""
    from common.data import prompt_messages, read_jsonl
    from common.logging_utils import save_json
    from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer

    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    n = int(cfg.get("eval_num_prompts", len(rows)))
    rows = rows[:n]
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    print(f"[RL eval] {name}: adapter={adapter or 'none (SFT)'} prompts={len(rows)} cap={max_new_tokens}")
    summary, generations = generate_and_score(
        policy, tokenizer, reward_model, reward_tokenizer,
        [prompt_messages(r) for r in rows],
        [str(r.get("prompt_id", r.get("source_index"))) for r in rows],
        max_prompt_length=int(cfg.get("max_prompt_length", 256)),
        max_new_tokens=max_new_tokens,
        reward_max_length=int(cfg.get("reward_max_length", 1280)),
        generation=cfg["generation"],
        seed=int(cfg["seed"]),
        batch_size=int(cfg.get("eval_batch_size", 8)),
        has_reference=adapter is not None,
    )
    results = {
        "eval_name": name,
        "adapter": adapter,
        **summary,
        # Aliases kept for older plotting code.
        "mean_kl": summary["kl_token_mean"],
        "mean_response_length_tokens": summary["length_tokens"]["mean"],
        "eos_termination_rate": summary["eos_rate"],
        "generations": generations,
    }
    save_json(out_json, results)
    print(f"  RM={summary['mean_reward']:+.3f}±{summary['sem_reward']:.3f}(sem) KL_tok={summary['kl_token_mean']:.4f} "
          f"H={summary['entropy_exact']:.3f} len={summary['length_tokens']['mean']:.1f} EOS={summary['eos_rate']:.2f}")
    try:
        plot_rl_eval(results, fig_path, title)
    except Exception as e:
        print(f"[plot] Warning: {e}")
    clear_gpu(policy, reward_model)
    return results
