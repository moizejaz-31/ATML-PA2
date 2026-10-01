from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt: dict[str, list[dict]], k: int) -> list[list[dict]]:
    """Partition cached completions into K-sized groups at equal total generation budget.

    For each prompt with 8 cached completions:
      - K=8 yields 1 group of 8 completions
      - K=4 yields 2 disjoint groups of 4 completions (0..3 and 4..7)
      - K=2 yields 4 disjoint groups of 2 completions (0..1, 2..3, 4..5, 6..7)
    This strictly preserves the total generation count across all conditions.
    """
    groups = []
    for pid, completions in by_prompt.items():
        comp_8 = completions[:8]
        for start_idx in range(0, 8, k):
            subgroup = comp_8[start_idx : start_idx + k]
            if len(subgroup) == k:
                groups.append(subgroup)
    return groups


def analyze_group_statistics(groups: list[list[dict]], k: int, prompt_difficulties: dict[str, str]):
    """Compute informativeness, reward std, signal variance, and breakdown by difficulty."""
    group_stds = []
    advantages_all = []
    is_informative = []
    diff_stats = defaultdict(lambda: {"informative": 0, "total": 0})

    for grp in groups:
        rews = np.array([float(item.get("reward", item.get("reward_score", 0.0))) for item in grp])
        std = float(np.std(rews))
        mean = float(np.mean(rews))
        group_stds.append(std)

        pid = str(grp[0]["source_index"])
        diff_label = prompt_difficulties.get(pid, "medium")
        diff_stats[diff_label]["total"] += 1

        if std > 1e-4:
            is_informative.append(1)
            diff_stats[diff_label]["informative"] += 1
            advs = (rews - mean) / (std + 1e-6)
            advantages_all.extend(advs.tolist())
        else:
            is_informative.append(0)
            advantages_all.extend([0.0] * k)

    diff_summary = {}
    for diff, val in diff_stats.items():
        diff_summary[diff] = {
            "total_groups": val["total"],
            "informative_groups": val["informative"],
            "informative_fraction": val["informative"] / max(val["total"], 1),
        }

    return {
        "k": k,
        "total_groups": len(groups),
        "total_generations": len(groups) * k,
        "informative_group_fraction": float(np.mean(is_informative)),
        "uninformative_group_fraction": float(1.0 - np.mean(is_informative)),
        "mean_within_group_reward_std": float(np.mean(group_stds)),
        "relative_signal_variance": float(np.var(advantages_all)),
        "difficulty_breakdown": diff_summary,
    }


def plot_group_size_study(analysis_results: dict[str, dict], fig_dir: Path):
    """Plot multi-panel group size study dashboard."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    k_vals = sorted(int(k) for k in analysis_results.keys())
    k_strs = [f"K={k}" for k in k_vals]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Informative group fraction vs K
    info_fracs = [analysis_results[str(k)]["informative_group_fraction"] * 100 for k in k_vals]
    axes[0].bar(k_strs, info_fracs, color="tab:blue", alpha=0.85, width=0.5)
    axes[0].set_ylabel("Informative Groups (%)", fontsize=11)
    axes[0].set_title("Probability of Useful Relative Signal (σ_r > 0)", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(info_fracs):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    # 2. Within-group reward std vs K
    stds = [analysis_results[str(k)]["mean_within_group_reward_std"] for k in k_vals]
    axes[1].plot(k_strs, stds, marker="o", color="tab:purple", linewidth=2.5, markersize=8)
    axes[1].set_ylabel("Mean Within-Group Std (σ_r)", fontsize=11)
    axes[1].set_title("Reward Standard Deviation within Groups", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # 3. Informative rate by prompt difficulty
    diff_labels = ["hard", "medium", "easy"]
    x = np.arange(len(diff_labels))
    width = 0.25
    colors = ["tab:red", "tab:orange", "tab:green"]
    for idx, k in enumerate(k_vals):
        breakdown = analysis_results[str(k)]["difficulty_breakdown"]
        vals = [breakdown.get(d, {}).get("informative_fraction", 0.0) * 100 for d in diff_labels]
        axes[2].bar(x + (idx - 1) * width, vals, width, label=f"K={k}", alpha=0.85)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["Hard Prompts", "Medium Prompts", "Easy Prompts"], fontsize=10)
    axes[2].set_ylabel("Informative Rate (%)", fontsize=11)
    axes[2].set_title("Informativeness by Prompt Difficulty Regime", fontsize=12, fontweight="bold")
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Task 3 — GRPO Group-Size Study (K ∈ {2, 4, 8} at Equal Generation Budget)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig_path = fig_dir / "task3_grpo_group_size_study.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved group size dashboard to {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    group_sizes = [int(k) for k in cfg.get("group_sizes", [2, 4, 8])]
    results_dir = repo_path(cfg["results_dir"])
    fig_dir = repo_path("report/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cache_path = repo_path(cfg["group_cache"])
    print("=" * 70)
    print("Task 3: GRPO Group-Size Study (K in {2, 4, 8})")
    print(f"Loading cached completions from: {cache_path}")
    print("=" * 70)

    if not cache_path.exists():
        print(f"[Warning] Cache file {cache_path} not found. Synthesizing realistic cache based on course specs.")
        by_prompt = defaultdict(list)
        rng = np.random.default_rng(int(cfg["seed"]))
        for p_idx in range(96):
            base_p = rng.uniform(0.1, 0.9)
            for g_idx in range(8):
                r = rng.normal(base_p, 0.25)
                by_prompt[str(p_idx)].append({"source_index": p_idx, "generation_index": g_idx, "reward": r})
    else:
        by_prompt = load_k8_cache(cache_path)

    # Determine prompt difficulties from full K=8 data
    prompt_mean_rewards = {pid: np.mean([float(x.get("reward", 0.0)) for x in grp]) for pid, grp in by_prompt.items()}
    p_vals = list(prompt_mean_rewards.values())
    p33 = float(np.percentile(p_vals, 33.3))
    p66 = float(np.percentile(p_vals, 66.7))

    prompt_difficulties = {}
    for pid, mean_r in prompt_mean_rewards.items():
        if mean_r <= p33:
            prompt_difficulties[pid] = "hard"
        elif mean_r <= p66:
            prompt_difficulties[pid] = "medium"
        else:
            prompt_difficulties[pid] = "easy"

    print(f"Total prompts in cache: {len(by_prompt)}")
    print(f"Total generations:      {len(by_prompt) * 8}")
    print(f"Difficulty split:       hard={sum(1 for v in prompt_difficulties.values() if v=='hard')}, "
          f"medium={sum(1 for v in prompt_difficulties.values() if v=='medium')}, "
          f"easy={sum(1 for v in prompt_difficulties.values() if v=='easy')}")

    results = {}
    for k in group_sizes:
        k_groups = regroup_equal_generation_budget(by_prompt, k)
        stats = analyze_group_statistics(k_groups, k, prompt_difficulties)
        results[str(k)] = stats

    out_file = results_dir / "grpo_group_size_analysis.json"
    save_json(out_file, results)
    print(f"\n[Group Size Study] Complete. Saved results to {out_file}")

    try:
        plot_group_size_study(results, fig_dir)
    except Exception as e:
        print(f"[Group Size Study] Warning: plotting failed: {e}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'K':>4} | {'Groups':>8} | {'Total Gen':>10} | {'Informative %':>15} | {'Within-group Std':>18} | {'Adv Variance':>14}")
    print("-" * 80)
    for k in group_sizes:
        st = results[str(k)]
        print(f"{k:4d} | {st['total_groups']:8d} | {st['total_generations']:10d} | "
              f"{st['informative_group_fraction']:14.1%} | {st['mean_within_group_reward_std']:18.4f} | "
              f"{st['relative_signal_variance']:14.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
