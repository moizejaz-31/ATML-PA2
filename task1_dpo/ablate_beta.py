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
from task1_dpo.evaluate import evaluate_dpo
from task1_dpo.train import run_training

SLIM_DROP = {"per_pair", "generations"}


def run_name_for(beta: float) -> str:
    return f"beta_{beta:.2f}".replace(".", "_")


def slim(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in SLIM_DROP}


def plot_beta_ablation(results: dict[str, dict], fig_dir: Path, results_dir: Path):
    """β study panels. The standard one-epoch run (different budget) and the untouched SFT policy are
    drawn as separate reference markers, never on the β line."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    betas = sorted(float(b) for b in results)
    ev = [results[str(b)]["eval"] for b in betas]
    sft = load_json(results_dir / "dpo_eval_sft_reference.json") if (results_dir / "dpo_eval_sft_reference.json").exists() else None
    std = load_json(results_dir / "dpo_eval_standard.json") if (results_dir / "dpo_eval_standard.json").exists() else None

    panels = [
        ("heldout_preference_accuracy", "Held-out preference accuracy", None),
        ("heldout_dpo_loss", "Held-out DPO loss (own β)", None),
        ("kl_token_mean", "Sampled KL to reference (token mean)", None),
        ("mean_reward", "RM score of generations", "sem_reward"),
        ("length_tokens", "Generated length (tokens, mean ± IQR/2)", None),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    x = np.arange(len(betas))
    for ax, (key, title, err) in zip(axes, panels):
        if key == "length_tokens":
            y = [e[key]["mean"] for e in ev]
            yerr = [e[key]["iqr"] / 2 for e in ev]
        else:
            y = [e[key] for e in ev]
            yerr = [e[err] for e in ev] if err else None
        ax.errorbar(x, y, yerr=yerr, marker="o", color="#2563EB", capsize=4, lw=2, label="short forks (600 pairs)")
        for ref, style, label in [(sft, ":", "SFT (no adapter)"), (std, "--", "standard DPO β=0.10, 1 epoch")]:
            if ref is not None and key in ref:
                val = ref[key]["mean"] if key == "length_tokens" else ref[key]
                if key in ("heldout_preference_accuracy", "heldout_dpo_loss") and ref is sft:
                    continue  # margins are identically 0 for the reference policy itself
                ax.axhline(val, ls=style, color="gray", lw=1.2, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([f"β={b:g}" for b in betas])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7)
    axes[3].legend(frameon=False, fontsize=7)
    fig.suptitle("Task 1 — DPO regularization-strength study (only β changes)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "task1_dpo_beta_ablation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Fit-vs-drift view: preference accuracy and RM score against sampled KL.
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
    kls = [e["kl_token_mean"] for e in ev]
    for ax, key, lab in [(ax2[0], "heldout_preference_accuracy", "held-out preference accuracy"),
                         (ax2[1], "mean_reward", "RM score")]:
        ys = [e[key] for e in ev]
        ax.plot(kls, ys, "-o", color="#2563EB")
        for b, kx, yy in zip(betas, kls, ys):
            ax.annotate(f"β={b:g}", (kx, yy), textcoords="offset points", xytext=(6, 6))
        if std is not None:
            ax.scatter([std["kl_token_mean"]], [std[key]], marker="*", s=160, color="#DC2626", label="standard (1 epoch)")
            ax.legend(frameon=False)
        ax.set_xlabel("sampled KL to reference (token mean)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    fig2.suptitle("Preference fit / reward vs. drift across β")
    fig2.tight_layout()
    fig2.savefig(fig_dir / "task1_dpo_reward_vs_kl_frontier.png", dpi=170, bbox_inches="tight")
    plt.close(fig2)

    # Training curves of the forks overlaid.
    fig3, ax3 = plt.subplots(1, 3, figsize=(16, 3.8))
    for b, color in zip(betas, ["#93C5FD", "#2563EB", "#1E3A8A"]):
        log = results_dir / f"dpo_train_{run_name_for(b)}.jsonl"
        if not log.exists():
            continue
        recs = [json.loads(l) for l in log.open(encoding="utf-8") if l.strip()]
        st = [r["step"] for r in recs]
        ax3[0].plot(st, [r["loss"] for r in recs], color=color, label=f"β={b:g}")
        ax3[1].plot(st, [r["reward_margin_mean"] / r["beta"] for r in recs], color=color, label=f"β={b:g}")
        ax3[2].plot(st, [r["preference_accuracy"] for r in recs], color=color, label=f"β={b:g}")
    for ax, t in zip(ax3, ["train DPO loss", "train log-ratio margin (unscaled)", "train preference accuracy"]):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
    ax3[0].legend(frameon=False)
    fig3.tight_layout()
    fig3.savefig(fig_dir / "task1_dpo_beta_training_curves.png", dpi=170, bbox_inches="tight")
    plt.close(fig3)
    print(f"[plot] Saved β-study figures to {fig_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="Reuse existing fork adapters")
    ap.add_argument("--plot-only", action="store_true", help="Only redraw figures from saved results")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    betas = [float(b) for b in cfg.get("betas", [0.03, 0.10, 0.30])]
    max_examples = int(cfg.get("short_ablation_examples", 600))
    results_dir = repo_path(cfg["results_dir"])
    fig_dir = repo_path("report/figures")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        plot_beta_ablation(load_json(results_dir / "dpo_beta_ablation_results.json"), fig_dir, results_dir)
        return

    print(f"Task 1: DPO regularization study  β={betas}  short-run budget={max_examples} kept pairs")
    combined = {}
    for beta in betas:
        run_name = run_name_for(beta)
        adapter_output = repo_path(f"outputs/task1_dpo/ablation_{run_name}")
        if args.skip_train and (adapter_output / "adapter_config.json").exists():
            train_summary = load_json(results_dir / f"dpo_summary_{run_name}.json")
        else:
            print(f"\n>>> DPO fork β={beta}")
            train_summary = run_training(
                config_path=args.config,
                run_name=run_name,
                dataset_path=cfg["paths"]["dpo_standard_train"],
                output_path=str(adapter_output),
                beta=beta,
                max_examples=max_examples,
            )
        print(f"\n>>> Evaluating fork β={beta}")
        eval_summary = evaluate_dpo(args.config, str(adapter_output), name=f"ablation_{run_name}", beta=beta)
        combined[str(beta)] = {
            "beta": beta,
            "budget": f"{max_examples} pairs, 1 pass",
            "train": {k: v for k, v in train_summary.items() if k != "used_pair_ids"},
            "eval": slim(eval_summary),
        }

    save_json(results_dir / "dpo_beta_ablation_results.json", combined)
    try:
        plot_beta_ablation(combined, fig_dir, results_dir)
    except Exception as e:
        print(f"[Ablation] Warning: plotting failed: {e}")

    print("\n" + "=" * 96)
    print(f"{'β':>6} | {'loss':>7} | {'pref acc':>8} | {'KL tok':>8} | {'KL seq':>8} | {'RM':>7} | {'len tok':>8} | {'IQR':>6}")
    for b in betas:
        e = combined[str(b)]["eval"]
        print(f"{b:6.2f} | {e['heldout_dpo_loss']:7.4f} | {e['heldout_preference_accuracy']:8.3f} | "
              f"{e['kl_token_mean']:8.4f} | {e['kl_sequence_mean']:8.3f} | {e['mean_reward']:7.3f} | "
              f"{e['length_tokens']['mean']:8.1f} | {e['length_tokens']['iqr']:6.1f}")
    print("=" * 96)


if __name__ == "__main__":
    main()
