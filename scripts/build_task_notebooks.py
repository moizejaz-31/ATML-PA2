"""Generate the per-step inspection notebooks (task*/notebooks/*.ipynb).

The experiment pipeline is the Python scripts; each notebook only (optionally) runs the scripts of one
step and then displays that step's saved tables (report/tables, built by common.visualize_all) and
figures (report/figures). Notebooks are written without outputs.
Run: python -m scripts.build_task_notebooks
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUP = """import os, sys, glob
from pathlib import Path
ROOT = Path.cwd()
while not (ROOT / "configs").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import Image, Markdown, display
pd.set_option("display.max_colwidth", 80)

def show_tables(*names):
    for n in names:
        p = ROOT / "report" / "tables" / f"{n}.csv"
        if p.exists():
            display(Markdown(f"**{n}**")); display(pd.read_csv(p))
        else:
            print(f"[missing] {p} - run the step, then `python -m common.visualize_all`")

def show_figures(*names):
    for n in names:
        hits = sorted(glob.glob(str(ROOT / "report" / "figures" / n)))
        if not hits:
            print(f"[missing] report/figures/{n}")
        for h in hits:
            display(Markdown(f"`{Path(h).name}`")); display(Image(filename=h, width=1100))
print("repo root:", ROOT)"""

NOTEBOOKS = {
    "task1_dpo/notebooks/1_standard_dpo.ipynb": dict(
        title="Task 1 · Step 1 — Standard DPO (one epoch)",
        about="Truncation policy report, SFT reference evaluation, one-epoch DPO, held-out evaluation "
              "(held-out DPO loss / preference accuracy on the filtered pairs; RM score, sampled KL, entropy and length on 200 held-out prompts).",
        rq=["Is stronger preference fitting accompanied by useful behaviour, or by disproportionate drift / a trivial length shift?"],
        run=["python -m task1_dpo.preprocess --config configs/dpo.yaml",
             "python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft_reference",
             "python -m task1_dpo.train --config configs/dpo.yaml --run-name standard",
             "python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard"],
        tables=["t1_truncation_policy", "t1_dpo_summary"],
        figures=["task1_dpo_truncation_policy.png", "task1_dpo_training_curves_standard.png", "task1_dpo_implicit_reward_standard.png",
                 "task1_dpo_eval_sft_reference.png", "task1_dpo_eval_standard.png"]),
    "task1_dpo/notebooks/2_beta_ablation.ipynb": dict(
        title="Task 1 · Step 2 — Regularisation strength (β ∈ {0.03, 0.10, 0.30})",
        about="Short forks (first 600 kept training pairs) from the same initialisation; only β changes. The standard run uses a different budget.",
        rq=["How does β change preference fitting, KL and RM score? Is the relationship monotonic over the tested range?"],
        run=["python -m task1_dpo.ablate_beta --config configs/dpo.yaml --skip-train"],
        tables=["t1_dpo_summary"],
        figures=["task1_dpo_beta_ablation.png", "task1_dpo_reward_vs_kl_frontier.png", "task1_dpo_beta_training_curves.png"]),
    "task1_dpo/notebooks/3_length_confounding.ipynb": dict(
        title="Task 1 · Step 3 — Length confounding",
        about="Length-balanced DPO vs standard DPO on the length-stratified held-out pairs; word-limit compliance (greedy + 8 samples/prompt) for SFT, standard and length-balanced DPO.",
        rq=["How much of the length behaviour is explained by the preference data? Does balancing the strata change which pairs are learned and how long the model answers?",
            "Cases where the aligned policy has a stronger preference/reward signal but a worse response (see report/qualitative_candidates.md)."],
        run=["python -m task1_dpo.analyze_length --config configs/dpo.yaml --skip-train"],
        tables=["t1_length_strata", "t1_word_limit"],
        figures=["task1_dpo_length_confounding.png", "task1_dpo_training_curves_length_balanced.png", "task1_dpo_eval_length_balanced.png"]),
    "task2_ppo/notebooks/1_standard_ppo.ipynb": dict(
        title="Task 2 · Step 1 — Standard PPO continuation (20 updates)",
        about="Continuation from the supplied policy + critic midpoint; held-out evaluation of SFT, midpoint and the continued policy on the same 64 prompts (cap 768).",
        rq=["Do reward, drift and critic behaviour evolve consistently over the continuation?"],
        run=["python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint",
             "python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter none --name sft_reference",
             "python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard",
             "python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard"],
        tables=["t2_ppo_heldout", "t_compute"],
        figures=["task2_ppo_continuation_standard.png", "task2_ppo_eval_midpoint.png", "task2_ppo_eval_standard.png"]),
    "task2_ppo/notebooks/2_clipping_study.ipynb": dict(
        title="Task 2 · Step 2 — Clipping study (ε ∈ {0.05, 0.20, 0.50})",
        about="Cached-batch geometry (midpoint vs cached old policy) + an update probe on the fixed cached batch, then matched 8-update forks.",
        rq=["How does ε change the fraction of updates constrained by clipping and the stability of the short continuation?"],
        run=["python -m task2_ppo.analyze_clipping --config configs/ppo.yaml"],
        tables=["t2_ppo_cached_clipping", "t2_ppo_heldout"],
        figures=["task2_ppo_clipping_study.png", "task2_ppo_continuation_fork_eps*.png"]),
    "task2_ppo/notebooks/3_kl_pressure_study.ipynb": dict(
        title="Task 2 · Step 3 — KL pressure (β_KL ∈ {0, 0.10, 0.20})",
        about="Matched 8-update forks at ε = 0.20 (the ε=0.20 / β_KL=0.10 fork is shared with the clipping study).",
        rq=["When KL pressure is weakened, which observable changes first?", "How informative is the learned reward as the policy moves away from the reference?"],
        run=["python -m task2_ppo.ablate_kl --config configs/ppo.yaml"],
        tables=["t2_ppo_heldout"],
        figures=["task2_ppo_kl_ablation.png", "task2_ppo_reward_vs_drift_tradeoff.png"]),
    "task3_grpo/notebooks/1_standard_grpo.ipynb": dict(
        title="Task 3 · Step 1 — Standard GRPO continuation (20 updates, K = 4)",
        about="Critic-free continuation from the supplied midpoint; completions that hit the 512-token cap are masked from the loss.",
        rq=["Is useful learning signal consistently available across prompt groups?", "After removing the critic, what dominates instability / sample inefficiency?"],
        run=["python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint",
             "python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard",
             "python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard"],
        tables=["t3_grpo_heldout", "t_compute"],
        figures=["task3_grpo_continuation_standard.png", "task3_grpo_eval_standard.png", "task6_standard_rl_trajectories.png"]),
    "task3_grpo/notebooks/2_group_size_study.ipynb": dict(
        title="Task 3 · Step 2 — Equal-generation group-size study (K ∈ {2, 4, 8})",
        about="192 cached generations regrouped into disjoint groups; difficulty = tertiles of the 8-sample mean RM reward.",
        rq=["How does K change the probability of a useful relative signal, and which difficulty regimes benefit most?"],
        run=["python -m task3_grpo.analyze_group_size --config configs/grpo.yaml"],
        tables=["t3_group_size"],
        figures=["task3_grpo_group_size_study.png"]),
    "task3_grpo/notebooks/3_normalization_study.ipynb": dict(
        title="Task 3 · Step 3 — Canonical GRPO vs Dr. GRPO normalisation",
        about="Matched 8-update forks; per-completion gradient norm vs length is logged during training.",
        rq=["Does the normalisation change response length or the allocation of gradient across short and long completions? Is that associated with quality?"],
        run=["python -m task3_grpo.compare_normalization --config configs/grpo.yaml --skip-train"],
        tables=["t3_normalization", "t3_grpo_heldout"],
        figures=["task3_grpo_normalization_study.png", "task3_grpo_continuation_fork_norm_*.png"]),
    "task4_safety/notebooks/1_generation_and_judging.ipynb": dict(
        title="Task 4 · Steps 1–2 — XSTest generation and AI judging",
        about="Greedy responses (cap 256) from SFT / standard DPO / standard PPO / standard GRPO, categorical AI judge.",
        rq=["Do higher preference/reward scores correspond to better safety calibration?", "Which policies confuse sensitive wording with harmful intent?"],
        run=["python -m task4_safety.generate_responses --config configs/feedback.yaml",
             "python -m task4_safety.judge_responses --config configs/feedback.yaml",
             "python -m task4_safety.evaluate_safety --config configs/feedback.yaml"],
        tables=["t4_safety"],
        figures=["task4_safety_calibration_dashboard.png"]),
    "task4_safety/notebooks/2_audit_and_calibration.ipynb": dict(
        title="Task 4 · Step 3 — Manual audit",
        about="Fill results/task4_safety/manual_audit_sheet.csv (blind: no policy, no AI label; 60 prompts x 4 policies) without opening manual_audit_key.csv, then re-run the aggregation.",
        rq=["What kinds of responses does the AI judge misclassify, and how much do those errors affect the policy comparison?"],
        run=["python -m task4_safety.make_audit_sheet --config configs/feedback.yaml",
             "python -m task4_safety.evaluate_safety --config configs/feedback.yaml"],
        tables=["t4_manual_audit", "t4_safety"],
        figures=["task4_manual_audit_confusion.png"]),
    "task5_feedback/notebooks/1_indomain_gsm8k.ipynb": dict(
        title="Task 5 · Step 1 — In-domain GSM8K comparison",
        about="Greedy generations (cap 512) from SFT / RLVR / RLAIF; exact verifier; fixed pairwise judge; verifier-judge agreement.",
        rq=["Where does AI preference feedback distinguish responses the binary verifier treats identically?"],
        run=["python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm"],
        tables=["t5_math"],
        figures=["task5_math_eval_gsm.png"]),
    "task5_feedback/notebooks/2_controlled_diagnostics.ipynb": dict(
        title="Task 5 · Step 2 — Controlled reward diagnostics",
        about="100 manually validated responses (20 problems x 5 variants); verifier vs judge per perturbation, judge order consistency, RLAIF group reward per variant.",
        rq=["When does the judge make an incorrect distinction because of style, verbosity or persuasive framing?",
            "Which reward source is more vulnerable to each diagnostic category?"],
        run=["python -m task5_feedback.score_perturbations --config configs/feedback.yaml"],
        tables=["t5_diagnostics"],
        figures=["task5_perturbation_diagnostics.png"]),
    "task5_feedback/notebooks/3_transfer_and_synthesis.ipynb": dict(
        title="Task 5 · Step 3 — SVAMP transfer and feedback-source synthesis",
        about="Same three policies on the fixed 100-example SVAMP subset; drops from in-domain; coverage / noise / exploitability / cost table.",
        rq=["Which behaviours transfer out of domain, and what supports or weakens the claim that either feedback source promotes more general reasoning?"],
        run=["python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer",
             "python -m task5_feedback.compare_feedback --config configs/feedback.yaml"],
        tables=["t5_math"],
        figures=["task5_math_eval_transfer.png", "task5_feedback_synthesis_dashboard.png"]),
    "task6_synthesis/notebooks/1_cross_task_synthesis.ipynb": dict(
        title="Task 6 — Cross-task synthesis (no new training)",
        about="Builds every report table and the cross-task figures from saved results, plus auto-selected qualitative candidates.",
        rq=["Preference strength vs drift; offline vs online feedback; optimisation bias; safety calibration; reward-source design."],
        run=["python -m common.visualize_all", "python -m scripts.extract_qualitative"],
        tables=["t1_dpo_summary", "t2_ppo_heldout", "t3_grpo_heldout", "t4_safety", "t5_math", "t_compute"],
        figures=["task6_cross_task_synthesis.png", "task6_standard_rl_trajectories.png"]),
}


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def build(spec):
    cells = [cell("markdown", f"# {spec['title']}\n\n{spec['about']}\n\n**Research questions (manual)**\n" +
                  "\n".join(f"- {q}" for q in spec["rq"]) +
                  "\n\nThis notebook only runs the step's scripts and shows their saved outputs; interpretation is left to the report.")]
    cells.append(cell("code", SETUP))
    run = "\n".join(f"    !{c}" for c in spec["run"])
    cells.append(cell("markdown", "## Run (optional; the Kaggle runner already executes these)"))
    cells.append(cell("code", f"RUN = False   # set True to (re)run this step here\nif RUN:\n{run}"))
    cells.append(cell("markdown", "## Tables"))
    cells.append(cell("code", "import subprocess\n"
                              "subprocess.run([sys.executable, '-m', 'common.visualize_all'], capture_output=True)  # rebuild report/tables\n"
                              "show_tables(" + ", ".join(repr(t) for t in spec["tables"]) + ")"))
    cells.append(cell("markdown", "## Figures"))
    cells.append(cell("code", "show_figures(" + ", ".join(repr(f) for f in spec["figures"]) + ")"))
    if "task6" in spec["title"].lower() or spec["title"].startswith("Task 6"):
        cells.append(cell("code", "display(Markdown((ROOT / 'report' / 'qualitative_candidates.md').read_text(encoding='utf-8')))"))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    for rel, spec in NOTEBOOKS.items():
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(build(spec), indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", rel)


if __name__ == "__main__":
    main()
