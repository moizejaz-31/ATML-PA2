"""Task 5 synthesis: in-domain vs transfer drops and a coverage / noise / exploitability / cost table
for the two feedback sources. Reads only saved results; it never substitutes placeholder numbers."""

from __future__ import annotations

import argparse
import timeit
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task5_feedback.rlvr import exact_reward

POLICIES = ["sft", "rlvr", "rlaif"]


def require(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the corresponding Task 5 step first (no placeholder numbers are used).")
    return load_json(path)


def plot_synthesis(gsm, tr, diag, table, fig_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    x = np.arange(len(POLICIES))
    labels = ["SFT", "RLVR", "RLAIF"]
    for off, data, name, c in [(-0.2, gsm, "GSM8K (in-domain)", "#2563EB"), (0.2, tr, "SVAMP (transfer)", "#F59E0B")]:
        acc = np.array([data["per_policy"][p]["exact_accuracy"] for p in POLICIES]) * 100
        ci = np.array([data["per_policy"][p]["exact_accuracy_ci95"] for p in POLICIES]) * 100
        axes[0].bar(x + off, acc, 0.4, yerr=[acc - ci[:, 0], ci[:, 1] - acc], capsize=4, color=c, label=name)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("exact accuracy (%)")
    axes[0].legend(frameon=False)
    axes[0].set_title("(a) In-domain vs transfer accuracy")

    for off, data, name, c in [(-0.2, gsm, "GSM8K", "#2563EB"), (0.2, tr, "SVAMP", "#F59E0B")]:
        wr = [data["pairwise_comparisons"][k]["win_rate_a_ties_half"] for k in ["rlvr_vs_sft", "rlaif_vs_sft"]]
        axes[1].bar(np.arange(2) + off, wr, 0.4, color=c, label=name)
    axes[1].axhline(0.5, color="black", ls=":")
    axes[1].set_xticks(range(2))
    axes[1].set_xticklabels(["RLVR vs SFT", "RLAIF vs SFT"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("(b) AI-judge win rate vs SFT (tie = 0.5)")
    axes[1].legend(frameon=False)

    axes[2].axis("off")
    rows = list(table)
    cell = [[table[r]["verifier"], table[r]["ai_judge"]] for r in rows]
    t = axes[2].table(cellText=cell, rowLabels=rows, colLabels=["exact verifier", "AI judge"], loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.5)
    axes[2].set_title("(c) Feedback-source properties (measured)")
    fig.suptitle("Task 5 — RLVR vs RLAIF synthesis")
    fig.tight_layout()
    fig.savefig(fig_dir / "task5_feedback_synthesis_dashboard.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rd = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm = require(rd / "math_eval_gsm.json")
    tr = require(rd / "math_eval_transfer.json")
    diag = require(rd / "perturbation_scores.json")

    drops = {}
    for p in POLICIES:
        g, t = gsm["per_policy"][p], tr["per_policy"][p]
        drops[p] = {"gsm_accuracy": g["exact_accuracy"], "transfer_accuracy": t["exact_accuracy"],
                    "accuracy_drop": g["exact_accuracy"] - t["exact_accuracy"],
                    "relative_retention": t["exact_accuracy"] / g["exact_accuracy"] if g["exact_accuracy"] > 0 else None,
                    "gsm_format": g["format_compliance_rate"], "transfer_format": t["format_compliance_rate"],
                    "gsm_len": g["length_tokens"]["mean"], "transfer_len": t["length_tokens"]["mean"]}
    for k in ["rlvr_vs_sft", "rlaif_vs_sft", "rlvr_vs_rlaif"]:
        drops[k] = {"gsm_win_rate": gsm["pairwise_comparisons"][k]["win_rate_a_ties_half"],
                    "transfer_win_rate": tr["pairwise_comparisons"][k]["win_rate_a_ties_half"],
                    "win_rate_drop": gsm["pairwise_comparisons"][k]["win_rate_a_ties_half"] - tr["pairwise_comparisons"][k]["win_rate_a_ties_half"]}

    # Coverage / noise / exploitability / cost, all measured from the saved results.
    pw = gsm["pairwise_comparisons"]["rlaif_vs_sft"]
    cont = pw["verifier_judge_contingency"]
    n = pw["n"]
    ver_decisive = (n - sum(cont["TIE"].values())) / n
    judge_decisive = 1 - pw["tie_rate"]
    comps = diag["comparisons"]
    sample = "Reasoning... so the answer is 42.\n#### 42"
    verifier_seconds = timeit.timeit(lambda: exact_reward(sample, "42"), number=2000) / 2000
    judge_sec = gsm["judge_cost"].get("mean_seconds_per_call")
    table = {
        "coverage: decisive on RLAIF-vs-SFT pairs": {"verifier": f"{ver_decisive:.2f}", "ai_judge": f"{judge_decisive:.2f}"},
        "coverage: S_reason (reasoning-only change)": {"verifier": f"{diag['s_reason']['rlvr']:.2f}", "ai_judge": f"{diag['s_reason']['rlaif']:.2f}"},
        "coverage: S_outcome (final-answer change)": {"verifier": f"{diag['s_outcome']['rlvr']:.2f}", "ai_judge": f"{diag['s_outcome']['rlaif']:.2f}"},
        "noise: A/B order-consistent decisions": {"verifier": "1.00 (deterministic)",
                                                  "ai_judge": f"{np.mean([c['rlaif_order_consistency'] for c in comps.values()]):.2f}"},
        "noise: agrees w/ verifier when verifier decides": {"verifier": "—", "ai_judge": f"{pw['agreement_on_verifier_decisive'] if pw['agreement_on_verifier_decisive'] is not None else float('nan'):.2f}"},
        "exploit: prefers persuasive filler": {"verifier": f"{comps['filler_susceptibility']['rlvr']['worse_rate']:.2f}",
                                              "ai_judge": f"{comps['filler_susceptibility']['rlaif']['worse_rate']:.2f}"},
        "exploit: prefers gold-distractor (wrong final)": {"verifier": f"{comps['distractor_robustness']['rlvr']['worse_rate']:.2f}",
                                                          "ai_judge": f"{comps['distractor_robustness']['rlaif']['worse_rate']:.2f}"},
        "cost: seconds per reward call": {"verifier": f"{verifier_seconds:.1e}", "ai_judge": f"{judge_sec:.2f}" if judge_sec else "n/a (cached)"},
        "cost: calls per K=4 group": {"verifier": "4", "ai_judge": "6 (all pairs)"},
    }
    synthesis = {"drops": drops, "feedback_source_table": table,
                 "verifier_judge_contingency_gsm_rlaif_vs_sft": cont,
                 "diagnostics": {"s_reason": diag["s_reason"], "s_outcome": diag["s_outcome"]},
                 # Backwards-compatible keys
                 "in_domain_gsm": gsm["per_policy"], "transfer_svamp": tr["per_policy"],
                 "retention_ratios": {p: drops[p]["relative_retention"] for p in POLICIES},
                 "pairwise_head_to_head": gsm["pairwise_comparisons"]}
    save_json(rd / "feedback_synthesis.json", synthesis)
    try:
        plot_synthesis(gsm, tr, diag, table, repo_path("report/figures"))
    except Exception as e:
        print(f"[Synthesis] Warning: plotting failed: {e}")

    print("\n" + "=" * 88)
    print(f"{'Policy':<6} | {'GSM acc':>8} | {'SVAMP acc':>9} | {'drop':>6} | {'GSM len':>7} | {'SVAMP len':>9}")
    for p in POLICIES:
        d = drops[p]
        print(f"{p.upper():<6} | {d['gsm_accuracy']:8.3f} | {d['transfer_accuracy']:9.3f} | {d['accuracy_drop']:6.3f} | "
              f"{d['gsm_len']:7.0f} | {d['transfer_len']:9.0f}")
    for k in ["rlvr_vs_sft", "rlaif_vs_sft"]:
        print(f"{k}: win rate GSM {drops[k]['gsm_win_rate']:.3f} -> SVAMP {drops[k]['transfer_win_rate']:.3f}")
    for k, v in table.items():
        print(f"  {k:<48} verifier={v['verifier']:<22} judge={v['ai_judge']}")
    print("=" * 88)


if __name__ == "__main__":
    main()
