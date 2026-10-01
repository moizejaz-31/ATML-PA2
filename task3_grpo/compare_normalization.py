from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_grpo


def plot_normalization_comparison(results: dict, fig_dir: Path):
    """Plot comparison between canonical GRPO (1/Tk) and Dr-GRPO (1/Lmax)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = ["grpo", "dr_grpo"]
    labels = ["Canonical GRPO (1/Tk)", "Dr-GRPO (1/L_max)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Held-out response length
    lengths = [results[m]["eval"]["mean_response_length_tokens"] for m in methods]
    axes[0].bar(labels, lengths, color=["tab:blue", "tab:green"], alpha=0.85, width=0.4)
    axes[0].set_ylabel("Response Length (tokens)", fontsize=11)
    axes[0].set_title("Response Length Under Normalization", fontsize=12, fontweight="bold")
    for i, v in enumerate(lengths):
        axes[0].text(i, v + 5, f"{v:.1f}", ha="center", fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    # 2. Held-out reward score
    rewards = [results[m]["eval"]["mean_reward"] for m in methods]
    axes[1].bar(labels, rewards, color=["tab:blue", "tab:green"], alpha=0.85, width=0.4)
    axes[1].set_ylabel("Held-Out Reward Score", fontsize=11)
    axes[1].set_title("Held-Out Reward Score Comparison", fontsize=12, fontweight="bold")
    for i, v in enumerate(rewards):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    # 3. Policy Drift (KL from reference)
    kls = [results[m]["eval"]["mean_kl"] for m in methods]
    axes[2].bar(labels, kls, color=["tab:blue", "tab:green"], alpha=0.85, width=0.4)
    axes[2].set_ylabel("Mean KL(π_θ || π_ref)", fontsize=11)
    axes[2].set_title("Policy Drift (KL) Comparison", fontsize=12, fontweight="bold")
    for i, v in enumerate(kls):
        axes[2].text(i, v + 0.002, f"{v:.4f}", ha="center", fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Task 3 — GRPO vs Dr-GRPO Sequence Normalization Study", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig_path = fig_dir / "task3_grpo_normalization_study.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved normalization comparison dashboard to {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="Skip training if forks exist")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    fork_updates = int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg["results_dir"])
    fig_dir = repo_path("report/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Task 3: GRPO vs Dr-GRPO Normalization Study")
    print(f"Updates per condition: {fork_updates}")
    print("=" * 70)

    loss_types = ["grpo", "dr_grpo"]
    results = {}

    for ltype in loss_types:
        run_name = f"norm_{ltype}"
        adapter_out = repo_path(f"outputs/task3_grpo/fork_{run_name}")

        if not args.skip_train:
            print(f"\n>>> Running Continuation Fork: loss_type = {ltype} ({fork_updates} updates) <<<")
            train_sum = run_grpo(
                config_path=args.config,
                output=str(adapter_out),
                updates=fork_updates,
                loss_type=ltype,
                run_name=run_name,
            )
        else:
            train_file = results_dir / f"grpo_summary_{run_name}.json"
            train_sum = load_json(train_file) if train_file.exists() else {"loss_type": ltype}

        print(f">>> Evaluating Fork loss_type = {ltype} <<<")
        eval_sum = evaluate_grpo(
            config_path=args.config,
            adapter=str(adapter_out),
            name=f"fork_{run_name}",
        )

        results[ltype] = {
            "loss_type": ltype,
            "train": train_sum,
            "eval": eval_sum,
        }

    out_file = results_dir / "grpo_normalization_comparison.json"
    save_json(out_file, results)
    print(f"\n[Normalization Study] Complete. Saved results to {out_file}")

    try:
        plot_normalization_comparison(results, fig_dir)
    except Exception as e:
        print(f"[Normalization Study] Warning: plotting failed: {e}")

    print("\n" + "=" * 80)
    print(f"{'Condition':<25} | {'Reward':>12} | {'KL (drift)':>12} | {'Length (tokens)':>16}")
    print("-" * 80)
    for ltype in loss_types:
        ev = results[ltype]["eval"]
        name_str = "Canonical GRPO (1/Tk)" if ltype == "grpo" else "Dr-GRPO (1/L_max)"
        print(f"{name_str:<25} | {ev['mean_reward']:12.4f} | {ev['mean_kl']:12.4f} | {ev['mean_response_length_tokens']:16.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
