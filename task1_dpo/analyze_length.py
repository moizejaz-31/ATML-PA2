from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate
from common.logging_utils import load_json, save_json, set_seed
from common.metrics import length_stats, parse_word_limit, wilson_interval, word_count
from common.models import clear_gpu, load_policy, load_tokenizer
from task1_dpo.evaluate import evaluate_dpo, preference_metrics, resolve_beta
from task1_dpo.preprocess import load_filtered_pairs
from task1_dpo.train import run_training

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]
STRATUM_LABEL = {"preferred_longer": "preferred longer", "length_matched": "length matched", "rejected_longer": "rejected longer"}
MODELS = [("sft", "SFT (no adapter)", "#9CA3AF"), ("standard", "standard DPO", "#2563EB"), ("length_balanced", "length-balanced DPO", "#16A34A")]


def stratified_summary(per_pair: list[dict]) -> dict:
    by = defaultdict(list)
    for p in per_pair:
        by[p["stratum"]].append(p)
    out = {}
    for s, ps in by.items():
        m = np.array([p["margin"] for p in ps])
        k = int((m > 0).sum())
        lo, hi = wilson_interval(k, len(m))
        out[s] = {
            "total_pairs": len(ps),
            "accuracy": float(k / len(m)),
            "accuracy_ci95": [lo, hi],
            "mean_margin": float(m.mean()),
            "median_margin": float(np.median(m)),
            "mean_dpo_loss": float(np.mean([p["dpo_loss"] for p in ps])),
            "pairs_with_truncated_response": int(sum(p["any_response_truncated"] for p in ps)),
            "mean_chosen_tokens": float(np.mean([p["chosen_tokens_full"] for p in ps])),
            "mean_rejected_tokens": float(np.mean([p["rejected_tokens_full"] for p in ps])),
        }
    return out


def word_limit_eval(model, tokenizer, rows: list[dict], max_new_tokens: int, n_samples: int, generation: dict, seed: int):
    """Greedy (primary) + `n_samples` sampled completions per word-limit prompt."""
    prompts = [prompt_messages(r) for r in rows]
    greedy = batch_generate(model, tokenizer, prompts, max_prompt_length=256, max_new_tokens=max_new_tokens,
                            temperature=0.0, do_sample=False)
    records = []
    for r, msgs, resp, ntok, hit in zip(rows, prompts, greedy["responses"], greedy["response_lengths"], greedy["truncated"]):
        limit = parse_word_limit(msgs[-1]["content"])
        wc = word_count(resp)
        records.append({"prompt_id": r.get("prompt_id"), "prompt": msgs[-1]["content"], "limit": limit,
                        "response": resp, "word_count": wc, "tokens": int(ntok), "hit_max_tokens": bool(hit),
                        "compliant": bool(limit is not None and wc <= limit), "excess_ratio": wc / limit if limit else None})
    sampled = []
    if n_samples > 0:
        set_seed(seed)
        rep = [m for m in prompts for _ in range(n_samples)]
        rep_rows = [r for r in rows for _ in range(n_samples)]
        for start in range(0, len(rep), 16):
            g = batch_generate(model, tokenizer, rep[start:start + 16], max_prompt_length=256, max_new_tokens=max_new_tokens,
                               temperature=float(generation["temperature"]), top_p=float(generation["top_p"]), do_sample=True)
            for r, msgs, resp in zip(rep_rows[start:start + 16], rep[start:start + 16], g["responses"]):
                limit = parse_word_limit(msgs[-1]["content"])
                sampled.append({"prompt_id": r.get("prompt_id"), "word_count": word_count(resp),
                                "compliant": bool(limit is not None and word_count(resp) <= limit)})
    k = sum(r["compliant"] for r in records)
    ks = sum(r["compliant"] for r in sampled)
    return {
        "greedy_compliance_rate": k / max(len(records), 1),
        "greedy_compliance_ci95": list(wilson_interval(k, len(records))),
        "greedy_word_count": length_stats(r["word_count"] for r in records),
        "greedy_token_count": length_stats(r["tokens"] for r in records),
        "mean_excess_ratio": float(np.mean([r["excess_ratio"] for r in records if r["excess_ratio"] is not None])),
        "sampled_n_per_prompt": n_samples,
        "sampled_compliance_rate": ks / max(len(sampled), 1) if sampled else None,
        "sampled_compliance_ci95": list(wilson_interval(ks, len(sampled))) if sampled else None,
        "sampled_word_count": length_stats(r["word_count"] for r in sampled) if sampled else None,
        # Backwards-compatible keys.
        "compliance_rate": k / max(len(records), 1),
        "mean_word_count": float(np.mean([r["word_count"] for r in records])) if records else 0.0,
        "details": records,
    }


def plot_length_analysis(res: dict, fig_dir: Path, results_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(24, 4.6))

    # (a) per-stratum held-out accuracy with Wilson CIs
    x = np.arange(len(STRATA))
    w = 0.38
    for j, key in enumerate(["standard", "length_balanced"]):
        st = res[key]["stratified"]
        acc = [st.get(s, {}).get("accuracy", np.nan) for s in STRATA]
        ci = np.array([st.get(s, {}).get("accuracy_ci95", [np.nan, np.nan]) for s in STRATA])
        err = np.vstack([np.array(acc) - ci[:, 0], ci[:, 1] - np.array(acc)])
        _, label, color = MODELS[j + 1]
        axes[0].bar(x + (j - 0.5) * w, acc, w, yerr=err, capsize=4, color=color, alpha=0.85, label=label)
    axes[0].axhline(0.5, color="black", lw=0.8, ls=":")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([STRATUM_LABEL[s] for s in STRATA])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("held-out preference accuracy")
    axes[0].set_title("(a) Accuracy by length stratum (95% CI)")
    axes[0].legend(frameon=False, fontsize=8)

    # (b) margin vs length difference
    for key in ["standard", "length_balanced"]:
        _, label, color = [m for m in MODELS if m[0] == key][0]
        pp = res[key]["per_pair"]
        d = np.array([p["chosen_tokens_full"] - p["rejected_tokens_full"] for p in pp], float)
        m = np.array([p["margin"] for p in pp], float)
        axes[1].scatter(d, m, s=8, alpha=0.3, color=color)
        edges = np.percentile(d, np.linspace(0, 100, 9))
        idx = np.clip(np.digitize(d, edges[1:-1]), 0, 7)
        centers = [d[idx == b].mean() for b in range(8) if (idx == b).any()]
        means = [m[idx == b].mean() for b in range(8) if (idx == b).any()]
        axes[1].plot(centers, means, "-o", color=color, lw=2, label=f"{label} (binned mean)")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("tokens(chosen) − tokens(rejected)")
    axes[1].set_ylabel("DPO margin Δ log-ratio")
    axes[1].set_title("(b) Held-out margin vs. pair length difference")
    axes[1].legend(frameon=False, fontsize=8)

    # (c) word-limit prompts: greedy word count vs limit
    det = {k: res[k]["word_limit"]["details"] for k, _, _ in MODELS if k in res}
    n = len(next(iter(det.values())))
    xx = np.arange(n)
    for j, (key, label, color) in enumerate([m for m in MODELS if m[0] in det]):
        axes[2].bar(xx + (j - 1) * 0.27, [r["word_count"] for r in det[key]], 0.27, color=color, label=label)
    limits = [r["limit"] for r in next(iter(det.values()))]
    axes[2].scatter(xx, limits, marker="_", s=400, color="#DC2626", zorder=5, label="word limit")
    axes[2].set_xticks(xx)
    axes[2].set_xticklabels([r["prompt_id"] for r in next(iter(det.values()))], rotation=45, fontsize=8)
    axes[2].set_ylabel("words (greedy)")
    axes[2].set_title("(c) Word-limit prompts")
    axes[2].legend(frameon=False, fontsize=7)

    # (d) generated length on the held-out prompts
    for key, label, color in MODELS:
        f = results_dir / f"dpo_eval_{'sft_reference' if key == 'sft' else key}.json"
        if f.exists():
            g = load_json(f)["generations"]
            axes[3].hist([e["response_length_tokens"] for e in g], bins=30, histtype="step", lw=2, color=color, label=label)
    axes[3].set_xlabel("generated tokens (held-out prompts, cap 256)")
    axes[3].set_title("(d) Generated length distribution")
    axes[3].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Task 1 — Length-confounding study", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "task1_dpo_length_confounding.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {fig_dir / 'task1_dpo_length_confounding.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="Reuse an existing length-balanced adapter")
    ap.add_argument("--skip-heldout-eval", action="store_true", help="Skip the standard held-out eval of the length-balanced model")
    ap.add_argument("--word-limit-samples", type=int, default=8)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    results_dir = repo_path(cfg["results_dir"])
    fig_dir = repo_path("report/figures")
    std_adapter = repo_path(cfg["standard_output"])
    len_adapter = repo_path(cfg["length_output"])

    if not (args.skip_train and (len_adapter / "adapter_config.json").exists()):
        print("\n>>> Training length-balanced DPO")
        run_training(args.config, "length_balanced", dataset_path=cfg["paths"]["dpo_length_train"], output_path=str(len_adapter))
    if not args.skip_heldout_eval:
        evaluate_dpo(args.config, str(len_adapter), name="length_balanced")

    tokenizer = load_tokenizer(cfg["base_model"])
    strat_rows, strat_info = load_filtered_pairs(cfg, tokenizer, cfg["paths"]["dpo_length_eval"])
    wl_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    max_len = int(cfg["max_sequence_length"])
    max_new = int(cfg.get("max_generation_tokens", 256))

    combined = {"stratified_eval_data": strat_info}
    for key, adapter in [("sft", None), ("standard", std_adapter), ("length_balanced", len_adapter)]:
        print(f"\n>>> {key}: stratified held-out pairs ({len(strat_rows)}) + word-limit prompts")
        model = load_policy(cfg, adapter_path=str(adapter) if adapter else None, trainable=False)
        entry = {}
        if adapter is not None:
            beta = resolve_beta(cfg, key, None)
            pref, per_pair = preference_metrics(model, tokenizer, strat_rows, max_len, beta, int(cfg.get("batch_size", 2)))
            entry.update({"overall": pref, "stratified": stratified_summary(per_pair), "per_pair": per_pair})
        set_seed(int(cfg["seed"]))
        entry["word_limit"] = word_limit_eval(model, tokenizer, wl_rows, max_new, args.word_limit_samples,
                                              cfg["generation"], int(cfg["seed"]))
        combined[key] = entry
        clear_gpu(model)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Backwards-compatible aliases.
    combined["standard_dpo"] = {k: v for k, v in combined["standard"].items() if k != "per_pair"}
    combined["length_balanced_dpo"] = {k: v for k, v in combined["length_balanced"].items() if k != "per_pair"}
    save_json(results_dir / "dpo_length_analysis.json", combined)

    try:
        plot_length_analysis(combined, fig_dir, results_dir)
    except Exception as e:
        print(f"[Length Analysis] Warning: plotting failed: {e}")

    print("\n" + "=" * 92)
    print(f"{'Stratum':<18} | {'n':>4} | {'std acc':>8} | {'LB acc':>8} | {'std margin':>10} | {'LB margin':>10} | {'trunc':>5}")
    for s in STRATA:
        a, b = combined["standard"]["stratified"].get(s, {}), combined["length_balanced"]["stratified"].get(s, {})
        print(f"{s:<18} | {a.get('total_pairs', 0):4d} | {a.get('accuracy', float('nan')):8.3f} | {b.get('accuracy', float('nan')):8.3f} | "
              f"{a.get('mean_margin', float('nan')):10.3f} | {b.get('mean_margin', float('nan')):10.3f} | {a.get('pairs_with_truncated_response', 0):5d}")
    for key, label, _ in MODELS:
        wl = combined[key]["word_limit"]
        print(f"word-limit {label:<22}: greedy compliance {wl['greedy_compliance_rate']:.0%} "
              f"(mean {wl['greedy_word_count']['mean']:.1f} words); sampled {wl['sampled_compliance_rate'] or 0:.0%}")
    print("=" * 92)


if __name__ == "__main__":
    main()
