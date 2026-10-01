from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_ppo


def plot_kl_ablation(results: dict[str, dict], fig_dir: Path):
    """Plot multi-panel KL penalty ablation study."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    kl_betas = sorted(float(b) for b in results.keys())
    beta_strs = [f"{b:.2f}" for b in kl_betas]

    rewards = [results[str(b)]["eval"]["mean_reward"] for b in kl_betas]
    reward_stds = [results[str(b)]["eval"].get("std_reward", 0.0) for b in kl_betas]
    kl_drifts = [results[str(b)]["eval"]["mean_kl"] for b in kl_betas]
    lengths = [results[str(b)]["eval"]["mean_response_length_tokens"] for b in kl_betas]
    eos_rates = [results[str(b)]["eval"]["eos_termination_rate"] for b in kl_betas]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Reward vs beta_KL
    axes[0, 0].errorbar(beta_strs, rewards, yerr=reward_stds, marker="o", color="tab:blue",
                        linewidth=2, markersize=8, capsize=5)
    axes[0, 0].set_xlabel("KL Penalty Coefficient β_KL", fontsize=11)
    axes[0, 0].set_ylabel("Held-Out Reward Score", fontsize=11)
    axes[0, 0].set_title("Reward Model Score vs β_KL", fontsize=12, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Reference Policy KL Drift vs beta_KL
    axes[0, 1].plot(beta_strs, kl_drifts, marker="s", color="tab:red", linewidth=2, markersize=8)
    axes[0, 1].set_xlabel("KL Penalty Coefficient β_KL", fontsize=11)
    axes[0, 1].set_ylabel("Mean KL(π_θ || π_ref)", fontsize=11)
    axes[0, 1].set_title("Policy Drift from Reference vs β_KL", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Response Length vs beta_KL
    axes[1, 0].plot(beta_strs, lengths, marker="^", color="tab:green", linewidth=2, markersize=8)
    axes[1, 0].set_xlabel("KL Penalty Coefficient β_KL", fontsize=11)
    axes[1, 0].set_ylabel("Mean Response Length (tokens)", fontsize=11)
    axes[1, 0].set_title("Response Length vs β_KL", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. EOS Termination Rate vs beta_KL
    axes[1, 1].plot(beta_strs, [r * 100 for r in eos_rates], marker="d", color="tab:purple", linewidth=2, markersize=8)
    axes[1, 1].set_xlabel("KL Penalty Coefficient β_KL", fontsize=11)
    axes[1, 1].set_ylabel("EOS Termination Rate (%)", fontsize=11)
    axes[1, 1].set_title("Proper EOS Termination Rate vs β_KL", fontsize=12, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Task 2 — PPO KL Regularization & Overoptimization Study", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig_path = fig_dir / "task2_ppo_kl_ablation.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved KL ablation dashboard to {fig_path}")

    # Pareto trade-off: Reward vs Policy Drift
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(kl_drifts, rewards, marker="o", markersize=10, linewidth=2, color="tab:blue")
    for b, kl_val, r_val in zip(kl_betas, kl_drifts, rewards):
        ax2.annotate(f"β_KL={b}", (kl_val, r_val), textcoords="offset points", xytext=(8, 8),
                     fontweight="bold", fontsize=10)
    ax2.set_xlabel("Policy Drift: Mean KL(π_θ || π_ref)", fontsize=11)
    ax2.set_ylabel("Mean Reward Model Score", fontsize=11)
    ax2.set_title("PPO Reward vs Drift Trade-Off (Goodhart / Overoptimization)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    tradeoff_path = fig_dir / "task2_ppo_reward_vs_drift_tradeoff.png"
    fig2.savefig(tradeoff_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[plot] Saved Reward vs Drift trade-off to {tradeoff_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="Skip training if forks exist")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    kl_values = [float(b) for b in cfg.get("kl_values", [0.0, 0.10, 0.20])]
    fork_updates = int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg["results_dir"])
    fig_dir = repo_path("report/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"Task 2: PPO KL Regularization Study (Reward Overoptimization)")
    print(f"Testing β_KL values: {kl_values}")
    print(f"Fork budget:         {fork_updates} updates")
    print("=" * 65)

    results = {}

    for beta_kl in kl_values:
        run_name = f"kl_{beta_kl:.2f}".replace(".", "_")
        adapter_out = repo_path(f"outputs/task2_ppo/fork_{run_name}")

        if not args.skip_train:
            print(f"\n>>> Running PPO Fork: β_KL = {beta_kl} ({fork_updates} updates) <<<")
            train_sum = run_ppo(
                config_path=args.config,
                output=str(adapter_out),
                updates=fork_updates,
                kl_beta=beta_kl,
                run_name=run_name,
            )
        else:
            train_file = results_dir / f"ppo_summary_{run_name}.json"
            train_sum = load_json(train_file) if train_file.exists() else {"kl_beta": beta_kl}

        print(f">>> Evaluating Fork β_KL = {beta_kl} <<<")
        eval_sum = evaluate_ppo(
            config_path=args.config,
            adapter=str(adapter_out),
            name=f"fork_{run_name}",
        )

        results[str(beta_kl)] = {
            "beta_kl": beta_kl,
            "train": train_sum,
            "eval": eval_sum,
        }

    out_file = results_dir / "ppo_kl_ablation_results.json"
    save_json(out_file, results)
    print(f"\n[KL Ablation] Complete. Saved results to {out_file}")

    try:
        plot_kl_ablation(results, fig_dir)
    except Exception as e:
        print(f"[KL Ablation] Warning: plotting failed: {e}")

    print("\n" + "=" * 80)
    print(f"{'β_KL':>6} | {'Held-out Reward':>18} | {'Policy Drift (KL)':>20} | {'Length (tokens)':>18} | {'EOS Rate':>10}")
    print("-" * 80)
    for b in kl_values:
        ev = results[str(b)]["eval"]
        print(f"{b:6.2f} | {ev['mean_reward']:17.4f} | {ev['mean_kl']:19.4f} | {ev['mean_response_length_tokens']:17.1f} | {ev['eos_termination_rate']:9.1%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
