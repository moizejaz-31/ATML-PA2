from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import batch_generate, response_sequence_logprobs
from common.logging_utils import load_json, save_json, set_seed
from common.metrics import word_count, word_limit_compliance
from common.models import load_policy, load_tokenizer, reference_mode
from task1_dpo.train import run_training


def get_stratum(row: dict) -> str:
    """Extract or infer stratum: preferred_longer, matched_length, rejected_longer."""
    if "stratum" in row:
        return str(row["stratum"])
    if "length_stratum" in row:
        return str(row["length_stratum"])
    # Infer from word counts
    yc, yr = preference_responses(row)
    wc = word_count(yc)
    wr = word_count(yr)
    diff = wc - wr
    if diff > 15:
        return "preferred_longer"
    elif diff < -15:
        return "rejected_longer"
    else:
        return "matched_length"


def evaluate_on_stratified_set(model, tokenizer, rows: list[dict], max_length: int, batch_size: int = 2):
    """Compute preference accuracy and implicit margin per stratum."""
    device = next(model.parameters()).device
    stratum_data = defaultdict(lambda: {"correct": 0, "total": 0, "margins": [], "chosen_lens": [], "rejected_lens": []})

    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        chosen_exs, rejected_exs = [], []
        for r in chunk:
            prompt_msgs = prompt_messages_from_preference(r)
            yc, yr = preference_responses(r)
            chosen_exs.append(encode_prompt_response(tokenizer, prompt_msgs, yc, max_length))
            rejected_exs.append(encode_prompt_response(tokenizer, prompt_msgs, yr, max_length))

        cb = {k: v.to(device) for k, v in pad_batch(tokenizer, chosen_exs).items()}
        rb = {k: v.to(device) for k, v in pad_batch(tokenizer, rejected_exs).items()}

        with torch.no_grad():
            pol_c, _, _ = response_sequence_logprobs(model, cb)
            pol_r, _, _ = response_sequence_logprobs(model, rb)
            with reference_mode(model):
                ref_c, _, _ = response_sequence_logprobs(model, cb)
                ref_r, _, _ = response_sequence_logprobs(model, rb)

        margins = ((pol_c - ref_c) - (pol_r - ref_r)).cpu().tolist()

        for r, m in zip(chunk, margins):
            s = get_stratum(r)
            yc, yr = preference_responses(r)
            stratum_data[s]["total"] += 1
            stratum_data[s]["margins"].append(m)
            if m > 0:
                stratum_data[s]["correct"] += 1
            stratum_data[s]["chosen_lens"].append(word_count(yc))
            stratum_data[s]["rejected_lens"].append(word_count(yr))

    summary = {}
    for s, d in stratum_data.items():
        summary[s] = {
            "total_pairs": d["total"],
            "accuracy": d["correct"] / max(d["total"], 1),
            "mean_margin": float(np.mean(d["margins"])) if d["margins"] else 0.0,
            "mean_chosen_words": float(np.mean(d["chosen_lens"])) if d["chosen_lens"] else 0.0,
            "mean_rejected_words": float(np.mean(d["rejected_lens"])) if d["rejected_lens"] else 0.0,
        }
    return summary


def evaluate_word_limit_compliance(model, tokenizer, prompt_rows: list[dict], max_new_tokens: int = 128):
    """Generate responses to word-limit constrained prompts and evaluate compliance."""
    prompts = [prompt_messages(r) for r in prompt_rows]
    gen_out = batch_generate(
        model, tokenizer, prompts,
        max_prompt_length=256,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )
    records = []
    for r, resp in zip(prompt_rows, gen_out["responses"]):
        p_text = prompt_messages(r)[-1].get("content", "")
        comp = word_limit_compliance(p_text, resp)
        wc = word_count(resp)
        records.append({
            "prompt": p_text,
            "response": resp,
            "word_count": wc,
            "compliant": bool(comp) if comp is not None else False,
        })
    compliant_count = sum(1 for rec in records if rec["compliant"])
    mean_wc = float(np.mean([r["word_count"] for r in records])) if records else 0.0
    return {
        "compliance_rate": compliant_count / max(len(records), 1),
        "mean_word_count": mean_wc,
        "details": records,
    }


def plot_length_analysis(standard_res: dict, length_res: dict, wl_std: dict, wl_len: dict, fig_dir: Path):
    """Generate multi-panel comparison for length bias analysis."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Per-stratum accuracy comparison
    strata = ["preferred_longer", "matched_length", "rejected_longer"]
    labels = ["Pref Longer", "Length Matched", "Rej Longer"]
    std_accs = [standard_res.get(s, {}).get("accuracy", 0.0) for s in strata]
    len_accs = [length_res.get(s, {}).get("accuracy", 0.0) for s in strata]

    x = np.arange(len(strata))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width/2, std_accs, width, label="Standard DPO", color="tab:blue", alpha=0.85)
    ax.bar(x + width/2, len_accs, width, label="Length-Balanced DPO", color="tab:green", alpha=0.85)
    ax.set_ylabel("Preference Accuracy", fontsize=11)
    ax.set_title("Preference Accuracy Across Length Strata", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # 2. Per-stratum margin comparison
    std_margins = [standard_res.get(s, {}).get("mean_margin", 0.0) for s in strata]
    len_margins = [length_res.get(s, {}).get("mean_margin", 0.0) for s in strata]
    ax = axes[1]
    ax.bar(x - width/2, std_margins, width, label="Standard DPO", color="tab:blue", alpha=0.85)
    ax.bar(x + width/2, len_margins, width, label="Length-Balanced DPO", color="tab:green", alpha=0.85)
    ax.set_ylabel("Mean Implicit Margin (Log-Ratio Advantage)", fontsize=11)
    ax.set_title("Implicit Margin Across Length Strata", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Word limit compliance & Length
    ax = axes[2]
    categories = ["Compliance Rate", "Mean Words (scaled /50)"]
    std_wl_vals = [wl_std["compliance_rate"], wl_std["mean_word_count"] / 50.0]
    len_wl_vals = [wl_len["compliance_rate"], wl_len["mean_word_count"] / 50.0]
    x_wl = np.arange(len(categories))
    ax.bar(x_wl - width/2, std_wl_vals, width, label="Standard DPO", color="tab:blue", alpha=0.85)
    ax.bar(x_wl + width/2, len_wl_vals, width, label="Length-Balanced DPO", color="tab:green", alpha=0.85)
    ax.set_title("Word-Limit Compliance & Concision", fontsize=12, fontweight="bold")
    ax.set_xticks(x_wl)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Task 1 — Length Bias & Confounding Study", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig_path = fig_dir / "task1_dpo_length_confounding.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved length confounding dashboard to {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="Skip training length-balanced model")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    results_dir = repo_path(cfg["results_dir"])
    fig_dir = repo_path("report/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    std_adapter = repo_path(cfg["standard_output"])
    len_adapter = repo_path(cfg["length_output"])

    # 1. Train length-balanced DPO condition
    if not args.skip_train:
        print("\n>>> Training Length-Balanced DPO Model <<<")
        run_training(
            config_path=args.config,
            run_name="length_balanced",
            dataset_path=cfg["paths"]["dpo_length_train"],
            output_path=str(len_adapter),
        )

    # 2. Evaluate both models on length-stratified eval set
    stratified_rows = read_jsonl(cfg["paths"]["dpo_length_eval"])
    wl_prompts = read_jsonl(cfg["paths"]["word_limit_prompts"])
    tokenizer = load_tokenizer(cfg["base_model"])
    max_len = int(cfg["max_sequence_length"])

    print(f"\n>>> Evaluating Standard DPO on Length-Stratified Set ({len(stratified_rows)} rows) <<<")
    model_std = load_policy(cfg, adapter_path=str(std_adapter), trainable=False)
    std_stratified = evaluate_on_stratified_set(model_std, tokenizer, stratified_rows, max_len)
    wl_std = evaluate_word_limit_compliance(model_std, tokenizer, wl_prompts)
    del model_std
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n>>> Evaluating Length-Balanced DPO on Length-Stratified Set ({len(stratified_rows)} rows) <<<")
    model_len = load_policy(cfg, adapter_path=str(len_adapter), trainable=False)
    len_stratified = evaluate_on_stratified_set(model_len, tokenizer, stratified_rows, max_len)
    wl_len = evaluate_word_limit_compliance(model_len, tokenizer, wl_prompts)
    del model_len
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    combined = {
        "standard_dpo": {
            "stratified": std_stratified,
            "word_limit": wl_std,
        },
        "length_balanced_dpo": {
            "stratified": len_stratified,
            "word_limit": wl_len,
        },
    }

    out_file = results_dir / "dpo_length_analysis.json"
    save_json(out_file, combined)
    print(f"\n[Length Analysis] Saved results to {out_file}")

    # Generate plots
    try:
        plot_length_analysis(std_stratified, len_stratified, wl_std, wl_len, fig_dir)
    except Exception as e:
        print(f"[Length Analysis] Warning: plotting failed: {e}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Stratum':<20} | {'Std DPO Acc':>12} | {'Length DPO Acc':>15} | {'Std Margin':>12} | {'Length Margin':>14}")
    print("-" * 80)
    for s in ["preferred_longer", "matched_length", "rejected_longer"]:
        s_acc = std_stratified.get(s, {}).get("accuracy", 0.0)
        l_acc = len_stratified.get(s, {}).get("accuracy", 0.0)
        s_m = std_stratified.get(s, {}).get("mean_margin", 0.0)
        l_m = len_stratified.get(s, {}).get("mean_margin", 0.0)
        print(f"{s:<20} | {s_acc:12.4f} | {l_acc:15.4f} | {s_m:12.4f} | {l_m:14.4f}")
    print("-" * 80)
    print(f"Word-limit compliance: Standard DPO = {wl_std['compliance_rate']:.1%}, Length-Balanced DPO = {wl_len['compliance_rate']:.1%}")
    print(f"Word-limit mean words: Standard DPO = {wl_std['mean_word_count']:.1f}, Length-Balanced DPO = {wl_len['mean_word_count']:.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
