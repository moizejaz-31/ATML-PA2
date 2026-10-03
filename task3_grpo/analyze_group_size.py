"""Task 3 equal-generation group-size study on the supplied K-cache (no training).

The cache holds 8 completions + RM rewards for each of 24 held-out prompts (192 generations). For
K in {2, 4, 8} the same 192 generations are partitioned into disjoint groups (8/K groups per prompt),
so every condition spends exactly the same generation budget.

Per K we report the manual's quantities: informative-group rate (std > released tolerance), mean
within-group reward std, variance of the group-relative signal, plus three measures of how
*reliable* the relative signal is, because with a continuous RM reward almost every group is
technically informative:
  * baseline error: |group mean - mean of the prompt's OTHER cached completions| (noise in the GRPO
    baseline, measured against completions not used by the group);
  * sign agreement: fraction of completions whose advantage sign under the K-group equals the sign of
    (reward - mean of the prompt's other completions), i.e. whether the update pushes the completion the
    same way an independent baseline would. For K=8 no other completions exist, so both are n/a;
  * raw-advantage variance (r - group mean) before the std normalisation.
Difficulty bins (defined once): prompt tertiles of the 8-sample mean RM reward
(hard = lowest third, easy = highest third). Completions that hit the generation cap are reported
separately, because the standard continuation masks them out of the loss.
"""

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
from common.metrics import GRPO_ZERO_STD_TOL as ZERO_STD_TOL

DIFFS = ["hard", "medium", "easy"]


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    bad = {pid: len(g) for pid, g in by_prompt.items() if len(g) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for g in by_prompt.values():
        g.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt: dict[str, list[dict]], k: int) -> list[list[dict]]:
    """Disjoint consecutive groups of size k from each prompt's 8 completions (0..k-1, k..2k-1, ...)."""
    groups = []
    for _, comps in by_prompt.items():
        c8 = comps[:8]
        for s in range(0, 8, k):
            if len(c8[s : s + k]) == k:
                groups.append(c8[s : s + k])
    return groups


def difficulty_bins(by_prompt):
    means = {pid: float(np.mean([float(x["reward"]) for x in g[:8]])) for pid, g in by_prompt.items()}
    q1, q2 = np.percentile(list(means.values()), [100 / 3, 200 / 3])
    bins = {pid: ("hard" if m <= q1 else "medium" if m <= q2 else "easy") for pid, m in means.items()}
    return bins, means, {"rule": "tertiles of the 8-completion mean RM reward per prompt",
                         "hard_max": float(q1), "medium_max": float(q2)}


def group_stats(groups, k, bins, prompt_means, tol=ZERO_STD_TOL, meaningful_std=0.1, by_prompt=None):
    out_rows = []
    for g in groups:
        pid = str(g[0]["source_index"])
        r = np.array([float(x["reward"]) for x in g])
        mu, sd = r.mean(), r.std()
        informative = sd > tol
        adv = (r - mu) / (sd + 1e-6) if informative else np.zeros_like(r)
        in_group = {int(x.get("generation_index", -1)) for x in g}
        others = [float(x["reward"]) for x in (by_prompt or {}).get(pid, [])[:8] if int(x.get("generation_index", -1)) not in in_group]
        ref = float(np.mean(others)) if others else float("nan")
        ref_sign = np.sign(r - ref) if others else np.full_like(r, np.nan)
        agree = ((np.sign(adv) == ref_sign) & (ref_sign != 0)).astype(float) if others else np.full_like(r, np.nan)
        capped = np.array([bool(x.get("clipped_at_max", False)) for x in g])
        out_rows.append({
            "prompt": pid, "difficulty": bins[pid], "std": float(sd), "informative": bool(informative),
            "meaningful": bool(sd > meaningful_std), "adv": adv.tolist(), "raw_adv": (r - mu).tolist(),
            "baseline_error": float(abs(mu - ref)) if others else float("nan"), "sign_agree": agree.tolist(),
            "trainable_after_mask": int((~capped).sum()), "best_minus_worst": float(r.max() - r.min()),
        })

    def summarise(rs):
        if not rs:
            return {}
        adv = np.concatenate([x["adv"] for x in rs])
        raw = np.concatenate([x["raw_adv"] for x in rs])
        return {
            "groups": len(rs),
            "informative_group_fraction": float(np.mean([x["informative"] for x in rs])),
            "uninformative_group_fraction": float(1 - np.mean([x["informative"] for x in rs])),
            "meaningful_group_fraction": float(np.mean([x["meaningful"] for x in rs])),
            "mean_within_group_reward_std": float(np.mean([x["std"] for x in rs])),
            "relative_signal_variance": float(np.var(adv)),
            "raw_advantage_variance": float(np.var(raw)),
            "mean_baseline_error": float(np.nanmean([x["baseline_error"] for x in rs])) if any(np.isfinite(x["baseline_error"]) for x in rs) else None,
            "advantage_sign_agreement": float(np.nanmean(np.concatenate([x["sign_agree"] for x in rs])))
            if any(np.isfinite(np.asarray(x["sign_agree"])).any() for x in rs) else None,
            "mean_best_minus_worst": float(np.mean([x["best_minus_worst"] for x in rs])),
            "groups_with_lt2_trainable_after_cap_mask": int(sum(x["trainable_after_mask"] < 2 for x in rs)),
        }

    res = {"k": k, "total_generations": len(groups) * k, **summarise(out_rows)}
    res["difficulty_breakdown"] = {d: summarise([x for x in out_rows if x["difficulty"] == d]) for d in DIFFS}
    res["total_groups"] = len(groups)
    return res


def plot_group_size_study(results: dict, fig_dir: Path, meta: dict):
    fig_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(int(k) for k in results)
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.4))
    x = np.arange(len(DIFFS))
    w = 0.25
    colors = {2: "#93C5FD", 4: "#2563EB", 8: "#1E3A8A"}
    for metric, ax, title in [
        ("mean_within_group_reward_std", axes[0], "Mean within-group reward std"),
        ("advantage_sign_agreement", axes[1], "Advantage-sign agreement with leave-group-out baseline"),
        ("mean_baseline_error", axes[2], "|group mean − mean of the prompt's other completions|"),
    ]:
        for j, k in enumerate(ks):
            v = [results[str(k)]["difficulty_breakdown"].get(d, {}).get(metric) for d in DIFFS]
            v = [np.nan if x_ is None else x_ for x_ in v]
            ax.bar(x + (j - 1) * w, v, w, color=colors.get(k, "gray"), label=f"K={k}")
            if results[str(k)][metric] is not None:
                ax.axhline(results[str(k)][metric], color=colors.get(k, "gray"), ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}" for d in DIFFS])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(frameon=False)

    rows = ["informative_group_fraction", "meaningful_group_fraction", "relative_signal_variance", "raw_advantage_variance"]
    cell = [[f"{results[str(k)][r]:.3f}" for k in ks] for r in rows]
    axes[3].axis("off")
    t = axes[3].table(cellText=cell, rowLabels=["informative (std>1e-6)", "std > 0.1 RM units", "var(normalised adv)",
                                                "var(raw adv)"], colLabels=[f"K={k}" for k in ks], loc="center")
    t.scale(1, 1.6)
    axes[3].set_title(f"Equal budget: {results[str(ks[0])]['total_generations']} generations per K", fontsize=10)
    fig.suptitle(f"Task 3 — GRPO group-size study (difficulty = {meta['rule']}; dashed = all prompts)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "task3_grpo_group_size_study.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {fig_dir / 'task3_grpo_group_size_study.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    ks = [int(k) for k in cfg.get("group_sizes", [2, 4, 8])]
    results_dir = repo_path(cfg["results_dir"])
    cache_path = repo_path(cfg["group_cache"])
    if not cache_path.exists():
        raise SystemExit(f"GRPO K-cache not found at {cache_path}. Run python -m scripts.download_assets first.")
    by_prompt = load_k8_cache(cache_path)
    bins, means, meta = difficulty_bins(by_prompt)
    print(f"Prompts={len(by_prompt)} generations={8 * len(by_prompt)} difficulty bins: "
          + ", ".join(f"{d}={sum(v == d for v in bins.values())}" for d in DIFFS))

    results = {str(k): group_stats(regroup_equal_generation_budget(by_prompt, k), k, bins, means, by_prompt=by_prompt) for k in ks}
    capped = sum(bool(x.get("clipped_at_max")) for g in by_prompt.values() for x in g)
    out = {**results, "_meta": {**meta, "prompts": len(by_prompt), "capped_completions": capped,
                                "zero_std_tolerance": ZERO_STD_TOL, "prompt_difficulty": bins, "prompt_mean_reward": means}}
    save_json(results_dir / "grpo_group_size_analysis.json", out)
    plot_group_size_study(results, repo_path("report/figures"), meta)

    print("\n" + "=" * 110)
    print(f"{'K':>3} | {'groups':>6} | {'inform.':>7} | {'std>0.1':>7} | {'mean std':>8} | {'var(adv)':>8} | {'var(raw)':>8} | {'sign agree':>10} | {'baseline err':>12}")
    for k in ks:
        s = results[str(k)]
        print(f"{k:3d} | {s['groups']:6d} | {s['informative_group_fraction']:7.3f} | {s['meaningful_group_fraction']:7.3f} | "
              f"{s['mean_within_group_reward_std']:8.4f} | {s['relative_signal_variance']:8.4f} | {s['raw_advantage_variance']:8.4f} | "
              f"{s['advantage_sign_agreement'] if s['advantage_sign_agreement'] is not None else float('nan'):10.3f} | "
              f"{s['mean_baseline_error'] if s['mean_baseline_error'] is not None else float('nan'):12.4f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
