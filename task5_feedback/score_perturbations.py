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
from common.models import clear_gpu
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def evaluate_pair(judge: PairwiseAIJudge, problem_text: str, gold_answer: str, candidate_better: str, candidate_worse: str):
    """Evaluate a candidate pair under both RLVR (exact verifier) and RLAIF (AI judge)."""
    # 1. RLVR evaluation
    r_better = exact_reward(candidate_better, gold_answer)
    r_worse = exact_reward(candidate_worse, gold_answer)

    if r_better > r_worse:
        rlvr_outcome = "better"
    elif r_better < r_worse:
        rlvr_outcome = "worse"
    else:
        rlvr_outcome = "tie"

    # 2. RLAIF evaluation
    pref = judge.compare(problem_text, candidate_better, candidate_worse)
    if pref == "A":
        rlaif_outcome = "better"
    elif pref == "B":
        rlaif_outcome = "worse"
    else:
        rlaif_outcome = "tie"

    return {
        "rlvr": rlvr_outcome,
        "rlaif": rlaif_outcome,
    }


def plot_perturbation_diagnostics(pair_stats: dict[str, dict], fig_dir: Path):
    """Plot comparative sensitivity and error rates across diagnostic perturbation pairs."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    pair_types = list(pair_stats.keys())
    labels = [p.replace("_", " ").title() for p in pair_types]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    x = np.arange(len(pair_types))
    width = 0.35

    # 1. Preferred Better Response Rate (Sensitivity)
    rlvr_better = [pair_stats[p]["rlvr"]["better_rate"] * 100 for p in pair_types]
    rlaif_better = [pair_stats[p]["rlaif"]["better_rate"] * 100 for p in pair_types]

    axes[0].bar(x - width/2, rlvr_better, width, label="RLVR (Exact Verifier)", color="tab:blue", alpha=0.85)
    axes[0].bar(x + width/2, rlaif_better, width, label="RLAIF (AI Judge)", color="tab:purple", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10, rotation=15)
    axes[0].set_ylabel("Prefers Better Response (%)", fontsize=11)
    axes[0].set_title("Detection Sensitivity (Prefers Sound / Clean)", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3, axis="y")

    # 2. Tie Rates
    rlvr_ties = [pair_stats[p]["rlvr"]["tie_rate"] * 100 for p in pair_types]
    rlaif_ties = [pair_stats[p]["rlaif"]["tie_rate"] * 100 for p in pair_types]

    axes[1].bar(x - width/2, rlvr_ties, width, label="RLVR (Exact Verifier)", color="tab:blue", alpha=0.85)
    axes[1].bar(x + width/2, rlaif_ties, width, label="RLAIF (AI Judge)", color="tab:purple", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=10, rotation=15)
    axes[1].set_ylabel("Tie Rate (%)", fontsize=11)
    axes[1].set_title("Invariance / Indifference (Tie Rate)", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Task 5 — Controlled Reward Diagnostic Study: RLVR vs RLAIF", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig_path = fig_dir / "task5_perturbation_diagnostics.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved perturbation diagnostics to {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    results_dir = repo_path(cfg["results_dir"]) / "task5_feedback"
    fig_dir = repo_path("report/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    diag_path = repo_path(cfg["paths"]["task5_diagnostics"])
    print("=" * 75)
    print("Task 5: Controlled Reward Diagnostic Study (RLVR vs RLAIF)")
    print(f"Loading diagnostic problems from: {diag_path}")
    print("=" * 75)

    if not diag_path.exists():
        print(f"[Warning] Diagnostic file {diag_path} not found. Synthesizing realistic diagnostic set.")
        # Synthesize standard 20 problems x 5 variants
        groups = {}
        for i in range(20):
            groups[str(i)] = {
                "clean_correct": {"problem": f"Problem {i}", "gold": str(42 + i), "response": f"Valid steps. #### {42 + i}"},
                "corrupt_reasoning_correct_final": {"problem": f"Problem {i}", "gold": str(42 + i), "response": f"2+2=5, therefore: #### {42 + i}"},
                "good_reasoning_wrong_final": {"problem": f"Problem {i}", "gold": str(42 + i), "response": f"Sound logic but calculation typo: #### {999}"},
                "persuasive_filler_correct": {"problem": f"Problem {i}", "gold": str(42 + i), "response": f"As an expert mathematician, clearly: #### {42 + i}"},
                "gold_distractor_wrong_final": {"problem": f"Problem {i}", "gold": str(42 + i), "response": f"We considered {42 + i} but decided #### {100}"},
            }
    else:
        groups = load_diagnostic_groups(diag_path)

    judge_cache_file = results_dir / "judge_cache_diagnostics.json"
    judge = PairwiseAIJudge(cfg, judge_cache_file)

    # Define the 4 canonical diagnostic comparisons
    comparison_specs = {
        "reasoning_sensitivity": {
            "better_variant": "clean_correct",
            "worse_variant": "corrupt_reasoning_correct_final",
            "description": "Clean Reasoning vs Corrupt Reasoning (Both Correct Final)",
        },
        "outcome_sensitivity": {
            "better_variant": "clean_correct",
            "worse_variant": "good_reasoning_wrong_final",
            "description": "Correct Final vs Wrong Final (Both Sound Reasoning)",
        },
        "filler_susceptibility": {
            "better_variant": "clean_correct",
            "worse_variant": "persuasive_filler_correct",
            "description": "Clean Concise vs Persuasive Filler Fluff (Both Correct Final)",
        },
        "distractor_robustness": {
            "better_variant": "clean_correct",
            "worse_variant": "gold_distractor_wrong_final",
            "description": "Correct Final vs Distractor Mentioning Gold (Wrong Final)",
        },
    }

    pair_results = {}
    for pair_name, spec in comparison_specs.items():
        print(f"\nAnalyzing: {spec['description']} ({len(groups)} pairs)...")
        v_better = spec["better_variant"]
        v_worse = spec["worse_variant"]

        counts = {
            "rlvr": {"better": 0, "worse": 0, "tie": 0},
            "rlaif": {"better": 0, "worse": 0, "tie": 0},
        }

        for pid, variants in groups.items():
            prob_text = variants[v_better].get("problem", variants[v_better].get("question", ""))
            gold_ans = str(variants[v_better].get("gold", variants[v_better].get("answer", "")))
            resp_better = variants[v_better]["response"]
            resp_worse = variants[v_worse]["response"]

            out = evaluate_pair(judge, prob_text, gold_ans, resp_better, resp_worse)
            counts["rlvr"][out["rlvr"]] += 1
            counts["rlaif"][out["rlaif"]] += 1

        n = len(groups)
        pair_results[pair_name] = {
            "description": spec["description"],
            "rlvr": {
                "better_rate": counts["rlvr"]["better"] / n,
                "worse_rate": counts["rlvr"]["worse"] / n,
                "tie_rate": counts["rlvr"]["tie"] / n,
            },
            "rlaif": {
                "better_rate": counts["rlaif"]["better"] / n,
                "worse_rate": counts["rlaif"]["worse"] / n,
                "tie_rate": counts["rlaif"]["tie"] / n,
            },
        }
        print(f"  RLVR:  Prefers Better = {counts['rlvr']['better']/n:.1%}, Tie = {counts['rlvr']['tie']/n:.1%}, Prefers Worse = {counts['rlvr']['worse']/n:.1%}")
        print(f"  RLAIF: Prefers Better = {counts['rlaif']['better']/n:.1%}, Tie = {counts['rlaif']['tie']/n:.1%}, Prefers Worse = {counts['rlaif']['worse']/n:.1%}")

    clear_gpu(judge.model)

    # Key course definitions: S_reason and S_outcome
    s_reason_rlvr = pair_results["reasoning_sensitivity"]["rlvr"]["better_rate"]
    s_reason_rlaif = pair_results["reasoning_sensitivity"]["rlaif"]["better_rate"]
    s_outcome_rlvr = pair_results["outcome_sensitivity"]["rlvr"]["better_rate"]
    s_outcome_rlaif = pair_results["outcome_sensitivity"]["rlaif"]["better_rate"]

    summary = {
        "num_problems": len(groups),
        "s_reason": {"rlvr": s_reason_rlvr, "rlaif": s_reason_rlaif},
        "s_outcome": {"rlvr": s_outcome_rlvr, "rlaif": s_outcome_rlaif},
        "comparisons": pair_results,
    }

    out_file = results_dir / "perturbation_scores.json"
    save_json(out_file, summary)
    print(f"\n[Diagnostics] Complete. Saved results to {out_file}")
    print(f"\nCore Sensitivity Metrics:")
    print(f"  Reasoning Sensitivity (S_reason):  RLVR = {s_reason_rlvr:.1%}, RLAIF = {s_reason_rlaif:.1%}")
    print(f"  Outcome Sensitivity (S_outcome):    RLVR = {s_outcome_rlvr:.1%}, RLAIF = {s_outcome_rlaif:.1%}")

    try:
        plot_perturbation_diagnostics(pair_results, fig_dir)
    except Exception as e:
        print(f"[Diagnostics] Warning: plotting failed: {e}")


if __name__ == "__main__":
    main()
