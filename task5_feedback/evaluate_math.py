from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.metrics import word_count
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def plot_math_evaluation(results: dict, dataset: str, fig_dir: Path):
    """Plot accuracy, format compliance, and pairwise win rates."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    policies = ["sft", "rlvr", "rlaif"]
    labels = ["SFT Base", "RLVR (Exact)", "RLAIF (AI Judge)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Exact Final-Answer Accuracy
    accs = [results["per_policy"][p]["exact_accuracy"] * 100 for p in policies]
    axes[0].bar(labels, accs, color=["tab:blue", "tab:green", "tab:purple"], alpha=0.85, width=0.45)
    axes[0].set_ylabel("Exact Accuracy (%)", fontsize=11)
    axes[0].set_title(f"Exact Final Answer Accuracy ({dataset.upper()})", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    # 2. Format compliance & Response length
    fmts = [results["per_policy"][p]["format_compliance_rate"] * 100 for p in policies]
    axes[1].bar(labels, fmts, color=["tab:blue", "tab:green", "tab:purple"], alpha=0.85, width=0.45)
    axes[1].set_ylabel("Format Compliance (%)", fontsize=11)
    axes[1].set_title("Designated Format (#### <number>) Rate", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(fmts):
        axes[1].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    # 3. Pairwise Head-to-Head Win Rates
    if "pairwise_comparisons" in results:
        pw = results["pairwise_comparisons"]
        pairs = list(pw.keys())
        p_labels = [p.replace("_vs_", " vs ").upper() for p in pairs]
        win_a = [pw[p]["win_rate_a"] * 100 for p in pairs]
        ties = [pw[p]["tie_rate"] * 100 for p in pairs]
        win_b = [pw[p]["win_rate_b"] * 100 for p in pairs]

        x = np.arange(len(pairs))
        width = 0.5
        axes[2].bar(x, win_a, width, label="Win Policy A", color="tab:green", alpha=0.85)
        axes[2].bar(x, ties, width, bottom=win_a, label="Tie", color="tab:gray", alpha=0.6)
        axes[2].bar(x, win_b, width, bottom=np.array(win_a) + np.array(ties), label="Win Policy B", color="tab:red", alpha=0.85)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(p_labels, fontsize=10)
        axes[2].set_ylabel("Proportion (%)", fontsize=11)
        axes[2].set_title("AI Judge Head-to-Head Outcomes", fontsize=12, fontweight="bold")
        axes[2].legend(fontsize=9)
        axes[2].set_ylim(0, 105)
        axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Task 5 — Math Reasoning Evaluation ({dataset.upper()} Dataset)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig_path = fig_dir / f"task5_math_eval_{dataset}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved math evaluation dashboard to {fig_path}")


def evaluate_dataset(config_path: str, dataset: str, batch_size: int = 4):
    cfg, rows, tokenizer = load_math_evaluation(config_path, dataset)
    set_seed(int(cfg["seed"]))
    policies = ["sft", "rlvr", "rlaif"]
    max_tokens = int(cfg.get("math_max_new_tokens", 512))

    results_dir = repo_path(cfg["results_dir"]) / "task5_feedback"
    fig_dir = repo_path("report/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Task 5: Math Reasoning Evaluation ({dataset.upper()})")
    print(f"Evaluating {len(rows)} test problems across: {policies}")
    print("=" * 70)

    policy_generations = {}
    policy_metrics = {}

    for pol in policies:
        print(f"\nGenerating responses for policy: {pol.upper()}...")
        model = load_frozen_policy(cfg, pol)
        records = []

        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            prompts = []
            for r in chunk:
                if "messages" in r and isinstance(r["messages"], list):
                    prompts.append(r["messages"])
                else:
                    q = str(r.get("question", r.get("problem", "")))
                    prompts.append([{
                        "role": "user",
                        "content": f"{q}\n\nShow your reasoning and end your response with exactly `#### <number>`."
                    }])
            gen = batch_generate(
                model,
                tokenizer,
                prompts,
                max_prompt_length=256,
                max_new_tokens=max_tokens,
                temperature=0.0,
                do_sample=False,
            )
            for r, resp, n_tok in zip(chunk, gen["responses"], gen["response_lengths"]):
                gold = str(r.get("gold_final", r.get("answer", r.get("gold", ""))))
                pred = extract_designated_final(resp)
                corr = exact_reward(resp, gold)
                records.append({
                    "problem": str(r.get("question", r.get("problem", ""))),
                    "gold": gold,
                    "response": resp,
                    "pred_final": pred,
                    "exact_correct": bool(corr),
                    "response_tokens": int(n_tok),
                    "response_words": word_count(resp),
                })

        clear_gpu(model)
        policy_generations[pol] = records

        exact_acc = sum(1 for r in records if r["exact_correct"]) / max(len(records), 1)
        fmt_rate = sum(1 for r in records if r["pred_final"] is not None) / max(len(records), 1)
        mean_tok = float(np.mean([r["response_tokens"] for r in records]))
        mean_word = float(np.mean([r["response_words"] for r in records]))

        policy_metrics[pol] = {
            "exact_accuracy": exact_acc,
            "format_compliance_rate": fmt_rate,
            "mean_tokens": mean_tok,
            "mean_words": mean_word,
        }
        print(f"  {pol.upper()}: Exact Acc = {exact_acc:.1%}, Format = {fmt_rate:.1%}, Mean Length = {mean_tok:.1f} tokens")

    # Pairwise AI judge comparisons
    judge_cache_file = results_dir / f"judge_cache_{dataset}.json"
    judge = PairwiseAIJudge(cfg, judge_cache_file)

    pairwise_comparisons = {}
    pair_specs = [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]

    for p_a, p_b in pair_specs:
        key = f"{p_a}_vs_{p_b}"
        print(f"\nRunning AI Judge pairwise comparison: {p_a.upper()} vs {p_b.upper()}...")
        wins_a = 0
        wins_b = 0
        ties = 0

        for r_a, r_b in zip(policy_generations[p_a], policy_generations[p_b]):
            pref = judge.compare(r_a["problem"], r_a["response"], r_b["response"])
            if pref == "A":
                wins_a += 1
            elif pref == "B":
                wins_b += 1
            else:
                ties += 1

        n_total = len(rows)
        pairwise_comparisons[key] = {
            "policy_a": p_a,
            "policy_b": p_b,
            "win_rate_a": wins_a / max(n_total, 1),
            "win_rate_b": wins_b / max(n_total, 1),
            "tie_rate": ties / max(n_total, 1),
        }
        print(f"  {p_a.upper()} Win: {wins_a / n_total:.1%}, Tie: {ties / n_total:.1%}, {p_b.upper()} Win: {wins_b / n_total:.1%}")

    clear_gpu(judge.model)

    combined = {
        "dataset": dataset,
        "num_problems": len(rows),
        "per_policy": policy_metrics,
        "pairwise_comparisons": pairwise_comparisons,
    }

    out_file = results_dir / f"math_eval_{dataset}.json"
    save_json(out_file, combined)
    print(f"\n[Math Eval] Complete. Saved results to {out_file}")

    try:
        plot_math_evaluation(combined, dataset, fig_dir)
    except Exception as e:
        print(f"[Math Eval] Warning: plotting failed: {e}")

    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    evaluate_dataset(args.config, args.dataset, args.batch_size)


if __name__ == "__main__":
    main()
