from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task1_dpo.evaluate import evaluate_dpo
from task1_dpo.train import run_training


def plot_beta_ablation(results: dict[str, dict], fig_dir: Path):
    """Plot multi-panel comparison across beta values."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    betas = sorted(float(b) for b in results.keys())
    beta_strs = [f"{b:.2f}" for b in betas]

    pref_accs = [results[str(b)]["eval"]["held_out_preference_accuracy"] for b in betas]
    kls = [results[str(b)]["eval"]["mean_kl_from_reference"] for b in betas]
    rewards = [results[str(b)]["eval"]["mean_reward"] for b in betas]
    reward_stds = [results[str(b)]["eval"].get("std_reward", 0.0) for b in betas]
    lengths = [results[str(b)]["eval"]["mean_response_length_words"] for b in betas]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Preference Accuracy vs Beta
    axes[0, 0].plot(beta_strs, pref_accs, marker="o", color="tab:blue", linewidth=2, markersize=8)
    axes[0, 0].set_title("Held-out Preference Accuracy vs β", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Regularization β", fontsize=11)
    axes[0, 0].set_ylabel("Preference Accuracy", fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Reference Policy KL vs Beta
    axes[0, 1].plot(beta_strs, kls, marker="s", color="tab:orange", linewidth=2, markersize=8)
    axes[0, 1].set_title("KL Divergence from Reference Policy vs β", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Regularization β", fontsize=11)
    axes[0, 1].set_ylabel("Mean KL(π_θ || π_ref)", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Reward Score vs Beta
    axes[1, 0].errorbar(beta_strs, rewards, yerr=reward_stds, marker="^", color="tab:green",
                        linewidth=2, markersize=8, capsize=5)
    axes[1, 0].set_title("Reward Model Score vs β", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Regularization β", fontsize=11)
    axes[1, 0].set_ylabel("Mean RM Score", fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Response Length vs Beta
    axes[1, 1].plot(beta_strs, lengths, marker="d", color="tab:purple", linewidth=2, markersize=8)
    axes[1, 1].set_title("Mean Response Length (words) vs β", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Regularization β", fontsize=11)
    axes[1, 1].set_ylabel("Length (words)", fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Task 1 — DPO Regularization Strength Study (β Ablation)", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    plot_path = fig_dir / "task1_dpo_beta_ablation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved beta ablation dashboard to {plot_path}")

    # Trade-off curve: Reward vs KL
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    scatter = ax2.scatter(kls, rewards, c=betas, cmap="viridis", s=150, zorder=5)
    for b, kl_val, r_val in zip(betas, kls, rewards):
        ax2.annotate(f"β={b}", (kl_val, r_val), textcoords="offset points", xytext=(8, 8),
                     fontweight="bold")
    ax2.plot(kls, rewards, "k--", alpha=0.4)
    ax2.set_xlabel("Mean KL from Reference (Policy Drift)", fontsize=11)
    ax2.set_ylabel("Reward Model Score", fontsize=11)
    ax2.set_title("DPO Pareto Frontier: Reward vs Policy Drift", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    cbar = fig2.colorbar(scatter, ax=ax2)
    cbar.set_label("β", fontsize=11)
    fig2.tight_layout()
    frontier_path = fig_dir / "task1_dpo_reward_vs_kl_frontier.png"
    fig2.savefig(frontier_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[plot] Saved Pareto frontier to {frontier_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="Skip training if checkpoints exist")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    betas = [float(b) for b in cfg.get("betas", [0.03, 0.10, 0.30])]
    max_examples = int(cfg.get("short_ablation_examples", 600))
    results_dir = repo_path(cfg["results_dir"])
    fig_dir = repo_path("report/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"============================================================")
    print(f"Task 1: DPO Regularization Strength Study")
    print(f"Testing β values: {betas}")
    print(f"Training subset cap: {max_examples} examples")
    print(f"============================================================")

    combined_results = {}

    for beta in betas:
        run_name = f"beta_{beta:.2f}".replace(".", "_")
        adapter_output = repo_path(f"outputs/task1_dpo/ablation_{run_name}")

        if not args.skip_train:
            print(f"\n>>> Running DPO training fork: β = {beta} <<<")
            train_summary = run_training(
                config_path=args.config,
                run_name=run_name,
                dataset_path=cfg["paths"]["dpo_standard_train"],
                output_path=str(adapter_output),
                beta=beta,
                max_examples=max_examples,
            )
        else:
            train_summary_file = results_dir / f"dpo_summary_{run_name}.json"
            if train_summary_file.exists():
                train_summary = load_json(train_summary_file)
            else:
                train_summary = {"beta": beta, "run_name": run_name}

        print(f"\n>>> Running evaluation: β = {beta} <<<")
        eval_summary = evaluate_dpo(
            config_path=args.config,
            adapter=str(adapter_output),
            name=f"ablation_{run_name}",
        )

        combined_results[str(beta)] = {
            "beta": beta,
            "train": train_summary,
            "eval": eval_summary,
        }

    # Save overall ablation summary
    ablation_json = results_dir / "dpo_beta_ablation_results.json"
    save_json(ablation_json, combined_results)
    print(f"\n[Ablation] Saved complete beta ablation results to {ablation_json}")

    # Generate plots
    try:
        plot_beta_ablation(combined_results, fig_dir)
    except Exception as e:
        print(f"[Ablation] Warning: plotting failed: {e}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'β':>6} | {'Pref Acc':>10} | {'KL (drift)':>10} | {'Mean Reward':>12} | {'Length (words)':>14}")
    print("-" * 70)
    for b in betas:
        res = combined_results[str(b)]["eval"]
        print(f"{b:6.2f} | {res['held_out_preference_accuracy']:10.4f} | {res['mean_kl_from_reference']:10.4f} | "
              f"{res['mean_reward']:12.4f} | {res['mean_response_length_words']:14.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
