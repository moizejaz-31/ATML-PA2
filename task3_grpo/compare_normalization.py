"""Task 3 length-normalisation study: canonical GRPO (1/T_k) vs the supplied Dr. GRPO-style (1/L_max)
normalisation. Two matched short forks from the identical midpoint (same prompts, reward, beta, eps,
generation settings and budget); only `loss_type` differs.

Length-conditioned statistic (defined once): for every trained completion the training loop records
its length T_k, |A_k| and the L2 norm of its own gradient contribution. We report the Spearman
correlation between T_k and that gradient norm, the share of the total per-completion gradient mass
carried by the longest/shortest third of completions, and binned means. Canonical GRPO gives every
completion the same total weight |A_k| (per-token weight |A_k|/T_k), so long completions get smaller
per-token updates; Dr. GRPO gives every token weight |A_k|/L_max, so a completion's total weight grows
with T_k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_grpo

LOSS_TYPES = [("grpo", "canonical GRPO (1/T_k)", "#2563EB"), ("dr_grpo", "Dr. GRPO (1/L_max)", "#DC2626")]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def length_conditioned(log_file: Path) -> dict:
    recs = [json.loads(l) for l in log_file.open(encoding="utf-8") if l.strip()]
    seqs = [s for r in recs for s in r["sequences"] if not s["masked"]]
    L = np.array([s["length"] for s in seqs], float)
    g = np.array([s["grad_norm"] for s in seqs], float)
    a = np.array([abs(s["advantage"]) for s in seqs], float)
    gpa = g / np.maximum(a, 1e-6)
    t1, t2 = np.percentile(L, [100 / 3, 200 / 3])
    short, long_ = L <= t1, L > t2
    return {
        "trained_completions": int(len(seqs)),
        "masked_completions": int(sum(s["masked"] for r in recs for s in r["sequences"])),
        "spearman_length_vs_gradnorm": spearman(L, g),
        "spearman_length_vs_gradnorm_per_unit_adv": spearman(L, gpa),
        "grad_mass_share_longest_third": float(g[long_].sum() / max(g.sum(), 1e-12)),
        "grad_mass_share_shortest_third": float(g[short].sum() / max(g.sum(), 1e-12)),
        "mean_gradnorm_short_third": float(g[short].mean()) if short.any() else float("nan"),
        "mean_gradnorm_long_third": float(g[long_].mean()) if long_.any() else float("nan"),
        "length_tertile_edges": [float(t1), float(t2)],
        "train_length_trajectory": [r["response_length"] for r in recs],
        "train_reward_trajectory": [r["reward_mean"] for r in recs],
        "train_kl_trajectory": [r["kl_token_mean"] for r in recs],
        "_L": L.tolist(), "_g": g.tolist(), "_a": a.tolist(),
    }


def plot_normalization(res: dict, fig_dir: Path, results_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=(23, 4.5))
    for lt, label, color in LOSS_TYPES:
        lc = res[lt]["length_conditioned"]
        L, g, a = np.array(lc["_L"]), np.array(lc["_g"]), np.array(lc["_a"])
        axes[0].scatter(L, g / np.maximum(a, 1e-6), s=14, alpha=0.5, color=color,
                        label=f"{label}  ρ_s={lc['spearman_length_vs_gradnorm_per_unit_adv']:.2f}")
        u = np.arange(1, len(lc["train_length_trajectory"]) + 1)
        axes[1].plot(u, lc["train_length_trajectory"], "-o", ms=3, color=color, label=label)
        axes[2].plot(u, lc["train_reward_trajectory"], "-o", ms=3, color=color, label=label)
    axes[0].set_xlabel("completion length T_k (tokens)")
    axes[0].set_ylabel("‖∇ contribution‖ / |A_k|")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Per-completion gradient vs length")
    axes[1].set_title("(b) Mean completion length during training")
    axes[2].set_title("(c) Mean RM reward during training")
    for ax in axes[1:3]:
        ax.set_xlabel("update")
    mid_file = results_dir / "grpo_eval_midpoint.json"
    mid = load_json(mid_file) if mid_file.exists() else None
    keys = [("mean_reward", "RM"), ("kl_token_mean", "KL×100"), ("length_tokens", "len/100")]
    x = np.arange(len(keys))
    for j, (lt, label, color) in enumerate(LOSS_TYPES):
        e = res[lt]["eval"]
        vals = [e["mean_reward"], 100 * e["kl_token_mean"], e["length_tokens"]["mean"] / 100]
        axes[3].bar(x + (j - 0.5) * 0.38, vals, 0.38, color=color, label=label)
    if mid is not None:
        mv = [mid["mean_reward"], 100 * mid["kl_token_mean"], mid["length_tokens"]["mean"] / 100]
        axes[3].scatter(x, mv, marker="_", s=600, color="black", zorder=5, label="midpoint")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels([k[1] for k in keys])
    axes[3].set_title("(d) Held-out evaluation (scaled)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Task 3 — sequence-length normalisation study")
    fig.tight_layout()
    fig.savefig(fig_dir / "task3_grpo_normalization_study.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {fig_dir / 'task3_grpo_normalization_study.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="Reuse existing fork adapters/logs")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    fork_updates = int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg["results_dir"])
    fig_dir = repo_path("report/figures")

    res = {}
    for lt, label, _ in LOSS_TYPES:
        run_name = f"fork_norm_{lt}"
        adapter = repo_path(f"outputs/task3_grpo/{run_name}")
        if args.skip_train and (adapter / "adapter_config.json").exists():
            train = load_json(results_dir / f"grpo_summary_{run_name}.json")
        else:
            print(f"\n>>> GRPO fork {label} ({fork_updates} updates)")
            train = run_grpo(args.config, output=str(adapter), updates=fork_updates, loss_type=lt, run_name=run_name)
        ev = evaluate_grpo(args.config, str(adapter), name=run_name)
        res[lt] = {"loss_type": lt, "label": label, "train": train,
                   "eval": {k: v for k, v in ev.items() if k != "generations"},
                   "length_conditioned": length_conditioned(results_dir / f"grpo_train_{run_name}.jsonl")}

    try:
        plot_normalization(res, fig_dir, results_dir)
    except Exception as e:
        print(f"[Normalization Study] Warning: plotting failed: {e}")
    slim = {lt: {**v, "length_conditioned": {k: x for k, x in v["length_conditioned"].items() if not k.startswith("_")}}
            for lt, v in res.items()}
    slim["_per_completion"] = {lt: {"length": v["length_conditioned"]["_L"], "grad_norm": v["length_conditioned"]["_g"],
                                    "abs_advantage": v["length_conditioned"]["_a"]} for lt, v in res.items()}
    save_json(results_dir / "grpo_normalization_comparison.json", slim)

    print("\n" + "=" * 100)
    print(f"{'condition':<24} | {'RM':>7} | {'KL tok':>8} | {'len':>6} | {'ρ_s(T, g/|A|)':>13} | {'long-third mass':>15} | {'masked':>6}")
    for lt, label, _ in LOSS_TYPES:
        e, lc = res[lt]["eval"], res[lt]["length_conditioned"]
        print(f"{label:<24} | {e['mean_reward']:7.3f} | {e['kl_token_mean']:8.4f} | {e['length_tokens']['mean']:6.1f} | "
              f"{lc['spearman_length_vs_gradnorm_per_unit_adv']:13.3f} | {lc['grad_mass_share_longest_third']:15.3f} | {lc['masked_completions']:6d}")
    print("=" * 100)


if __name__ == "__main__":
    main()
