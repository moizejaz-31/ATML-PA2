"""Task 5 controlled reward-diagnostic study (20 GSM8K problems x 5 manually validated variants).

For each perturbation the clean response is the diagnostically better one:
  reasoning   : clean_correct vs corrupt_reasoning_correct_final   (same correct final)  -> S_reason
  outcome     : clean_correct vs good_reasoning_wrong_final        (reasoning ~fixed)    -> S_outcome
  filler      : clean_correct vs persuasive_filler_correct         (same correct final)
  distractor  : clean_correct vs gold_distractor_wrong_final       (gold appears, wrong final)
plus one conflict pair without a designated winner (correct final + corrupt reasoning vs sound
reasoning + wrong final) that shows which signal each mechanism follows.

Exact verifier: binary `exact_reward`; it is validated against the file's `expected_exact_reward`.
AI judge: fixed course judge (deterministic internal A/B orientation). Each pair is also judged with the
two candidates passed in the opposite order; the share of order-consistent decisions measures judge
position noise. RLAIF group reward: the normalised pairwise win rate of each of the 5 variants inside
its problem group (exactly the direct-RLAIF reward definition, with K=5).
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
from common.models import clear_gpu
from task5_feedback.judge_audit import AuditedPairwiseJudge
from task5_feedback.rlvr import exact_reward

VARIANTS = ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
            "persuasive_filler_correct", "gold_distractor_wrong_final"]
COMPARISONS = {
    "reasoning_sensitivity": ("clean_correct", "corrupt_reasoning_correct_final", "reasoning corrupted, final kept correct"),
    "outcome_sensitivity": ("clean_correct", "good_reasoning_wrong_final", "final answer changed, reasoning ~fixed"),
    "filler_susceptibility": ("clean_correct", "persuasive_filler_correct", "irrelevant persuasive filler added"),
    "distractor_robustness": ("clean_correct", "gold_distractor_wrong_final", "gold number as rejected distractor, wrong final"),
}
CONFLICT = ("corrupt_reasoning_correct_final", "good_reasoning_wrong_final")


def load_diagnostic_groups(path):
    by_problem = defaultdict(dict)
    for row in read_jsonl(path):
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, v in by_problem.items():
        missing = set(VARIANTS) - set(v)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def outcome(x: float, y: float) -> str:
    return "better" if x > y else "worse" if x < y else "tie"


def judge_outcome(pref: str) -> str:
    return {"A": "better", "B": "worse", "TIE": "tie"}[pref]


def rates(counts: dict, n: int) -> dict:
    return {"better_rate": counts["better"] / n, "tie_rate": counts["tie"] / n, "worse_rate": counts["worse"] / n,
            "counts": dict(counts)}


def plot_diagnostics(res: dict, fig_dir: Path):
    comps = list(COMPARISONS)
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.8), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    y = np.arange(len(comps))
    for j, (mech, off) in enumerate([("rlvr", -0.2), ("rlaif", 0.2)]):
        left = np.zeros(len(comps))
        for key, color in [("better_rate", "#16A34A"), ("tie_rate", "#D1D5DB"), ("worse_rate", "#DC2626")]:
            v = np.array([res["comparisons"][c][mech][key] for c in comps]) * 100
            axes[0].barh(y + off, v, 0.38, left=left, color=color, edgecolor="white",
                         label=key.replace("_rate", "") if j == 0 else None)
            left += v
    axes[0].set_yticks(np.concatenate([y - 0.2, y + 0.2]))
    axes[0].set_yticklabels([f"verifier · {c}" for c in comps] + [f"AI judge · {c}" for c in comps], fontsize=8)
    axes[0].set_xlabel("% of the 20 problems")
    axes[0].set_title("(a) Prefers diagnostically better / tie / worse")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    gr = res["group_rewards"]
    x = np.arange(len(VARIANTS))
    axes[1].bar(x - 0.2, [gr["rlvr"][v] for v in VARIANTS], 0.4, color="#2563EB", label="verifier reward")
    axes[1].bar(x + 0.2, [gr["rlaif"][v] for v in VARIANTS], 0.4, color="#9333EA", label="RLAIF group reward (K=5)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["clean", "corrupt\nreasoning", "wrong\nfinal", "persuasive\nfiller", "gold\ndistractor"], fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("(b) Mean reward per variant")
    axes[1].legend(frameon=False, fontsize=8)

    cons = [res["comparisons"][c]["rlaif_order_consistency"] for c in comps]
    axes[2].bar(range(len(comps)), np.array(cons) * 100, color="#9333EA")
    axes[2].set_xticks(range(len(comps)))
    axes[2].set_xticklabels([c.split("_")[0] for c in comps])
    axes[2].set_ylim(0, 105)
    axes[2].set_title("(c) AI judge order consistency (%)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Task 5 — controlled diagnostics  S_reason: verifier {res['s_reason']['rlvr']:.2f} / judge {res['s_reason']['rlaif']:.2f}   "
                 f"S_outcome: verifier {res['s_outcome']['rlvr']:.2f} / judge {res['s_outcome']['rlaif']:.2f}")
    fig.tight_layout()
    fig.savefig(fig_dir / "task5_perturbation_diagnostics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {fig_dir / 'task5_perturbation_diagnostics.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"]) / "task5_feedback"
    results_dir.mkdir(parents=True, exist_ok=True)
    diag_path = repo_path(cfg["paths"]["task5_diagnostics"])
    if not diag_path.exists():
        raise SystemExit(f"Diagnostic set not found at {diag_path}. Run python -m scripts.download_assets first.")
    groups = load_diagnostic_groups(diag_path)
    n = len(groups)

    # Verifier validation against the manually validated expectations.
    mismatches = [{"problem_id": pid, "variant": v, "expected": int(row["expected_exact_reward"]),
                   "got": exact_reward(row["response"], row["gold_final"])}
                  for pid, vs in groups.items() for v, row in vs.items()
                  if int(exact_reward(row["response"], row["gold_final"])) != int(row["expected_exact_reward"])]
    print(f"Verifier check: {5 * n - len(mismatches)}/{5 * n} rows match expected_exact_reward")

    judge = AuditedPairwiseJudge(cfg, results_dir / "judge_cache_diagnostics.json")
    comparisons, items = {}, []
    for name, (vb, vw, desc) in COMPARISONS.items():
        c_v = {"better": 0, "tie": 0, "worse": 0}
        c_j = {"better": 0, "tie": 0, "worse": 0}
        consistent, unparsed = 0, 0
        for pid, vs in groups.items():
            q, gold = vs[vb]["question"], vs[vb]["gold_final"]
            rb, rw = vs[vb]["response"], vs[vw]["response"]
            ov = outcome(exact_reward(rb, gold), exact_reward(rw, gold))
            oj = judge_outcome(judge.compare(q, rb, rw))
            oj_rev = {"A": "worse", "B": "better", "TIE": "tie"}[judge.compare(q, rw, rb)]
            consistent += oj == oj_rev
            unparsed += bool(judge.is_unparsed(q, rb, rw))
            c_v[ov] += 1
            c_j[oj] += 1
            items.append({"comparison": name, "problem_id": pid, "verifier": ov, "judge": oj, "judge_reversed": oj_rev})
        comparisons[name] = {"description": desc, "better_variant": vb, "worse_variant": vw,
                             "rlvr": rates(c_v, n), "rlaif": rates(c_j, n),
                             "rlaif_order_consistency": consistent / n, "rlaif_unparsed": unparsed}
        print(f"  {name:22s} verifier B/T/W={c_v['better']}/{c_v['tie']}/{c_v['worse']}  "
              f"judge B/T/W={c_j['better']}/{c_j['tie']}/{c_j['worse']}  order-consistent={consistent}/{n}")

    a, b = CONFLICT
    conflict = {"rlvr": {"first": 0, "tie": 0, "second": 0}, "rlaif": {"first": 0, "tie": 0, "second": 0}}
    for pid, vs in groups.items():
        q, gold = vs[a]["question"], vs[a]["gold_final"]
        ov = outcome(exact_reward(vs[a]["response"], gold), exact_reward(vs[b]["response"], gold))
        oj = judge_outcome(judge.compare(q, vs[a]["response"], vs[b]["response"]))
        m = {"better": "first", "worse": "second", "tie": "tie"}
        conflict["rlvr"][m[ov]] += 1
        conflict["rlaif"][m[oj]] += 1

    group_r = {"rlvr": defaultdict(list), "rlaif": defaultdict(list)}
    for pid, vs in groups.items():
        resp = [vs[v]["response"] for v in VARIANTS]
        rew = judge.group_rewards(vs[VARIANTS[0]]["question"], resp)
        for v, r_ai in zip(VARIANTS, rew):
            group_r["rlaif"][v].append(r_ai)
            group_r["rlvr"][v].append(exact_reward(vs[v]["response"], vs[v]["gold_final"]))
    judge.flush()
    cost = judge.cost()
    clear_gpu(judge.model)

    pooled_outcome = {
        mech: (comparisons["outcome_sensitivity"][mech]["counts"]["better"] + comparisons["distractor_robustness"][mech]["counts"]["better"]) / (2 * n)
        for mech in ("rlvr", "rlaif")}
    res = {
        "num_problems": n,
        "verifier_validation": {"rows": 5 * n, "mismatches": mismatches},
        "s_reason": {m: comparisons["reasoning_sensitivity"][m]["better_rate"] for m in ("rlvr", "rlaif")},
        "s_outcome": {m: comparisons["outcome_sensitivity"][m]["better_rate"] for m in ("rlvr", "rlaif")},
        "s_outcome_pooled_with_distractor": pooled_outcome,
        "comparisons": comparisons,
        "conflict_pair": {"first": a, "second": b, **conflict},
        "group_rewards": {m: {v: float(np.mean(x)) for v, x in d.items()} for m, d in group_r.items()},
        "judge_cost": cost,
    }
    save_json(results_dir / "perturbation_scores.json", res)
    save_json(results_dir / "perturbation_items.json", items)
    plot_diagnostics(res, repo_path("report/figures"))
    print(f"\nS_reason : verifier={res['s_reason']['rlvr']:.2f}  judge={res['s_reason']['rlaif']:.2f}")
    print(f"S_outcome: verifier={res['s_outcome']['rlvr']:.2f}  judge={res['s_outcome']['rlaif']:.2f}")
    print(f"Conflict ({a} vs {b}): verifier {conflict['rlvr']}  judge {conflict['rlaif']}")


if __name__ == "__main__":
    main()
