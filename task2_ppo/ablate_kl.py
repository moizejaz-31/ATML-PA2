"""Task 2 KL-pressure (reward over-optimisation) study: beta_KL in {0, 0.10, 0.20} at the reference eps,
matched short forks from the identical midpoint, common held-out protocol."""

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
from task2_ppo.forks import fork_name, run_or_load_fork

TRAJ = [("reward_rm", "RM reward (rollout)"), ("kl_token_mean", "Sampled KL to reference"),
        ("entropy_exact", "Policy entropy (exact)"), ("response_length", "Response length (tokens)")]


def plot_kl_ablation(res: dict, fig_dir: Path, results_dir: Path):
    kls = res["kl_values"]
    colors = dict(zip([str(k) for k in kls], ["#DC2626", "#2563EB", "#16A34A"]))
    fig, axes = plt.subplots(2, 4, figsize=(22, 8))
    # Row 1: training trajectories (which observable moves first when KL pressure is weakened?)
    for ax, (key, title) in zip(axes[0], TRAJ):
        for k in kls:
            f = results_dir / f"ppo_train_{fork_name(res['clip_epsilon'], k)}.jsonl"
            if not f.exists():
                continue
            recs = [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
            ax.plot([r["update"] for r in recs], [r[key] for r in recs], "-o", ms=3, color=colors[str(k)], label=f"β_KL={k}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("fork update")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False)

    # Row 2: held-out metrics (with the midpoint as the common starting point)
    mid_file = results_dir / "ppo_eval_midpoint.json"
    mid = load_json(mid_file) if mid_file.exists() else None
    x = np.arange(len(kls))
    held = [("mean_reward", "Held-out RM score", "sem_reward"), ("kl_token_mean", "Held-out sampled KL", None),
            ("entropy_exact", "Held-out entropy", None), ("length_tokens", "Held-out length (tokens)", None)]
    for ax, (key, title, err) in zip(axes[1], held):
        ev = [res["forks"][str(k)]["eval"] for k in kls]
        y = [e[key]["mean"] if key == "length_tokens" else e[key] for e in ev]
        yerr = [e[err] for e in ev] if err else ([e[key]["iqr"] / 2 for e in ev] if key == "length_tokens" else None)
        ax.bar(x, y, yerr=yerr, capsize=4, color=[colors[str(k)] for k in kls], alpha=0.85)
        if mid is not None:
            mv = mid[key]["mean"] if key == "length_tokens" else mid[key]
            ax.axhline(mv, color="black", ls="--", lw=1, label="midpoint (start)")
            ax.legend(frameon=False, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"β_KL={k}" for k in kls])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Task 2 — KL-pressure study (ε={res['clip_epsilon']}, {res['fork_updates']} updates per fork)")
    fig.tight_layout()
    fig.savefig(fig_dir / "task2_ppo_kl_ablation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    for k in kls:
        e = res["forks"][str(k)]["eval"]
        ax2.errorbar(e["kl_token_mean"], e["mean_reward"], yerr=e["sem_reward"], fmt="o", ms=9, color=colors[str(k)], label=f"β_KL={k}")
    if mid is not None:
        ax2.errorbar(mid["kl_token_mean"], mid["mean_reward"], yerr=mid["sem_reward"], fmt="s", color="black", label="midpoint")
    ax2.set_xlabel("held-out sampled KL to reference")
    ax2.set_ylabel("held-out RM score (± s.e.m.)")
    ax2.set_title("Reward bought with drift")
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig(fig_dir / "task2_ppo_reward_vs_drift_tradeoff.png", dpi=170, bbox_inches="tight")
    plt.close(fig2)
    print(f"[plot] Saved KL-study figures to {fig_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--no-reuse", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    kls = [float(b) for b in cfg.get("kl_values", [0.0, 0.10, 0.20])]
    eps = float(cfg["clip_epsilon"])
    results_dir = repo_path(cfg["results_dir"])
    fig_dir = repo_path("report/figures")

    forks = {str(k): run_or_load_fork(args.config, eps, k, reuse=not args.no_reuse) for k in kls}
    res = {"kl_values": kls, "clip_epsilon": eps, "fork_updates": int(cfg["fork_updates"]), "forks": forks}
    # Backwards-compatible flat view
    res.update({str(k): {"beta_kl": k, "train": forks[str(k)]["train"], "eval": forks[str(k)]["eval"]} for k in kls})
    save_json(results_dir / "ppo_kl_ablation_results.json", res)
    try:
        plot_kl_ablation(res, fig_dir, results_dir)
    except Exception as e:
        print(f"[KL Ablation] Warning: plotting failed: {e}")

    print("\n" + "=" * 90)
    print(f"{'β_KL':>5} | {'RM':>7} | {'KL tok':>8} | {'entropy':>7} | {'len':>6} | {'EOS':>5} | {'train KL last':>13}")
    for k in kls:
        e, t = forks[str(k)]["eval"], forks[str(k)]["train"]
        print(f"{k:5.2f} | {e['mean_reward']:7.3f} | {e['kl_token_mean']:8.4f} | {e['entropy_exact']:7.3f} | "
              f"{e['length_tokens']['mean']:6.1f} | {e['eos_rate']:5.2f} | {t['final_kl']:13.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
