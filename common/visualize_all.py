"""Master visualization and results inspection utility for ATML PA2.

This script aggregates all saved result files (.json, .jsonl, .csv) across Tasks 1-5,
re-generates publication-ready figures in report/figures/, and prints formatted summary tables
addressing all research questions from the assignment manual.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json_safe(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def inspect_task1():
    print("\n" + "=" * 80)
    print("TASK 1: DIRECT PREFERENCE OPTIMIZATION (DPO) SUMMARY")
    print("=" * 80)
    summary = load_json_safe(RESULTS_DIR / "task1_dpo" / "dpo_summary_standard.json")
    if summary:
        print(f"Standard DPO Run: {summary['total_steps']} steps, final loss = {summary['final_loss']:.4f}, final pref acc = {summary['final_preference_accuracy']:.1%}")

    ablation = load_json_safe(RESULTS_DIR / "task1_dpo" / "dpo_beta_ablation_results.json")
    if ablation:
        print("\n[RQ1: Regularization Strength β]")
        print(f"{'β':>6} | {'Held-out Acc':>14} | {'Policy Drift (KL)':>18} | {'Mean Reward':>14} | {'Length (words)':>16}")
        print("-" * 74)
        for b_str, data in sorted(ablation.items(), key=lambda x: float(x[0])):
            ev = data["eval"]
            print(f"{float(b_str):6.2f} | {ev['held_out_preference_accuracy']:13.1%} | {ev['mean_kl_from_reference']:17.4f} | {ev['mean_reward']:13.3f} | {ev['mean_response_length_words']:15.1f}")

    length_res = load_json_safe(RESULTS_DIR / "task1_dpo" / "dpo_length_analysis.json")
    if length_res:
        print("\n[RQ2: Length Confounding & Bias]")
        std_strat = length_res["standard_dpo"]["stratified"]
        len_strat = length_res["length_balanced_dpo"]["stratified"]
        print(f"{'Stratum':<20} | {'Standard DPO Acc':>18} | {'Length-Balanced Acc':>22}")
        print("-" * 65)
        for s in ["preferred_longer", "matched_length", "rejected_longer"]:
            s_acc = std_strat.get(s, {}).get("accuracy", 0.0)
            l_acc = len_strat.get(s, {}).get("accuracy", 0.0)
            print(f"{s:<20} | {s_acc:17.1%} | {l_acc:21.1%}")

        std_wl = length_res["standard_dpo"]["word_limit"]
        len_wl = length_res["length_balanced_dpo"]["word_limit"]
        print(f"\nWord Limit Compliance: Standard = {std_wl['compliance_rate']:.1%}, Length-Balanced = {len_wl['compliance_rate']:.1%}")


def inspect_task2():
    print("\n" + "=" * 80)
    print("TASK 2: PROXIMAL POLICY OPTIMIZATION (PPO) SUMMARY")
    print("=" * 80)
    ppo_summary = load_json_safe(RESULTS_DIR / "task2_ppo" / "ppo_summary_standard.json")
    if ppo_summary:
        print(f"Standard PPO Continuation: {ppo_summary['updates']} updates, final reward = {ppo_summary['final_reward']:.3f}, peak VRAM = {ppo_summary.get('peak_vram_mb', 0):.0f}MB")

    clip_res = load_json_safe(RESULTS_DIR / "task2_ppo" / "ppo_clipping_study_results.json")
    if clip_res:
        print("\n[RQ1: Clipping Parameter ε]")
        cached = clip_res.get("cached_geometry", {})
        forks = clip_res.get("fork_results", {})
        print(f"{'ε':>6} | {'Cached Clip %':>15} | {'Held-out Reward':>18} | {'Policy Drift (KL)':>18}")
        print("-" * 65)
        for eps_str in sorted(cached.keys(), key=lambda x: float(x)):
            c_frac = cached[eps_str]["clip_fraction"]
            f_r = forks.get(eps_str, {}).get("eval", {}).get("mean_reward", float("nan"))
            f_kl = forks.get(eps_str, {}).get("eval", {}).get("mean_kl", float("nan"))
            print(f"{float(eps_str):6.2f} | {c_frac:14.1%} | {f_r:17.3f} | {f_kl:17.4f}")

    kl_res = load_json_safe(RESULTS_DIR / "task2_ppo" / "ppo_kl_ablation_results.json")
    if kl_res:
        print("\n[RQ2 & RQ3: KL Penalty β_KL & Reward Overoptimization]")
        print(f"{'β_KL':>6} | {'Held-out Reward':>18} | {'Policy Drift (KL)':>18} | {'Mean Length':>14} | {'EOS Rate':>12}")
        print("-" * 74)
        for b_str in sorted(kl_res.keys(), key=lambda x: float(x)):
            ev = kl_res[b_str]["eval"]
            print(f"{float(b_str):6.2f} | {ev['mean_reward']:17.3f} | {ev['mean_kl']:17.4f} | {ev['mean_response_length_tokens']:13.1f} | {ev['eos_termination_rate']:11.1%}")


def inspect_task3():
    print("\n" + "=" * 80)
    print("TASK 3: GROUP RELATIVE POLICY OPTIMIZATION (GRPO) SUMMARY")
    print("=" * 80)
    grpo_summary = load_json_safe(RESULTS_DIR / "task3_grpo" / "grpo_summary_standard.json")
    if grpo_summary:
        print(f"Standard GRPO Continuation: {grpo_summary['updates']} updates, K={grpo_summary['k_generations']}, final reward = {grpo_summary['final_reward']:.3f}, within-group std = {grpo_summary['final_within_group_std']:.3f}")

    grp_res = load_json_safe(RESULTS_DIR / "task3_grpo" / "grpo_group_size_analysis.json")
    if grp_res:
        print("\n[RQ1: Group Size K at Equal Generation Budget]")
        print(f"{'K':>4} | {'Informative %':>15} | {'Within-group Std':>18} | {'Adv Variance':>14}")
        print("-" * 57)
        for k_str in sorted(grp_res.keys(), key=lambda x: int(x)):
            st = grp_res[k_str]
            print(f"{int(k_str):4d} | {st['informative_group_fraction']:14.1%} | {st['mean_within_group_reward_std']:17.4f} | {st['relative_signal_variance']:13.4f}")

    norm_res = load_json_safe(RESULTS_DIR / "task3_grpo" / "grpo_normalization_comparison.json")
    if norm_res:
        print("\n[RQ2: Sequence Normalization: Canonical (1/Tk) vs Dr-GRPO (1/Lmax)]")
        print(f"{'Normalization':<25} | {'Held-out Reward':>18} | {'Mean Length (tokens)':>22} | {'KL (drift)':>12}")
        print("-" * 82)
        for m in ["grpo", "dr_grpo"]:
            ev = norm_res[m]["eval"]
            name = "Canonical GRPO (1/Tk)" if m == "grpo" else "Dr-GRPO (1/L_max)"
            print(f"{name:<25} | {ev['mean_reward']:17.3f} | {ev['mean_response_length_tokens']:21.1f} | {ev['mean_kl']:11.4f}")


def inspect_task4():
    print("\n" + "=" * 80)
    print("TASK 4: SAFETY CALIBRATION EVALUATION (XSTest)")
    print("=" * 80)
    safety_res = load_json_safe(RESULTS_DIR / "task4_safety" / "safety_evaluation_results.json")
    if safety_res:
        print(f"{'Policy':<8} | {'Safe Answer %':>14} | {'Over-Refusal %':>15} | {'Justified Refusal %':>20} | {'Unsafe Compliance %':>20}")
        print("-" * 85)
        for pol, st in safety_res.items():
            print(f"{pol.upper():<8} | {st['safe_prompt_answer_rate']:13.1%} | {st['safe_prompt_over_refusal_rate']:14.1%} | {st['unsafe_prompt_justified_refusal_rate']:19.1%} | {st['unsafe_prompt_unsafe_compliance_rate']:19.1%}")


def inspect_task5():
    print("\n" + "=" * 80)
    print("TASK 5: FEEDBACK SOURCE COMPARISON (RLVR vs RLAIF)")
    print("=" * 80)
    syn_res = load_json_safe(RESULTS_DIR / "task5_feedback" / "feedback_synthesis.json")
    if syn_res:
        print(f"{'Policy':<10} | {'GSM8K Acc':>12} | {'SVAMP Acc':>12} | {'Retention %':>14}")
        print("-" * 54)
        for p in ["sft", "rlvr", "rlaif"]:
            g = syn_res["in_domain_gsm"][p]["exact_accuracy"]
            t = syn_res["transfer_svamp"][p]["exact_accuracy"]
            ret = syn_res["retention_ratios"][p]
            print(f"{p.upper():<10} | {g:11.1%} | {t:11.1%} | {ret:13.1%}")

        print("\nDiagnostic Sensitivity Breakdown:")
        s_r = syn_res["diagnostic_sensitivities"]["s_reason"]
        s_o = syn_res["diagnostic_sensitivities"]["s_outcome"]
        print(f"  Reasoning Sensitivity (S_reason): RLVR = {s_r['rlvr']:.1%}, RLAIF = {s_r['rlaif']:.1%}")
        print(f"  Outcome Sensitivity (S_outcome):   RLVR = {s_o['rlvr']:.1%}, RLAIF = {s_o['rlaif']:.1%}")


def plot_cross_task_synthesis():
    """Generate a master 4-panel publication-ready synthesis figure for Task 6."""
    dpo_beta = load_json_safe(RESULTS_DIR / "task1_dpo" / "dpo_beta_ablation_results.json")
    ppo_kl = load_json_safe(RESULTS_DIR / "task2_ppo" / "ppo_kl_ablation_results.json")
    grpo_norm = load_json_safe(RESULTS_DIR / "task3_grpo" / "grpo_normalization_comparison.json")
    safety_res = load_json_safe(RESULTS_DIR / "task4_safety" / "safety_evaluation_results.json")
    feedback_syn = load_json_safe(RESULTS_DIR / "task5_feedback" / "feedback_synthesis.json")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- 1. Policy Drift vs Reward Frontier (DPO vs PPO vs GRPO) ---
    ax = axes[0, 0]
    has_p1 = False
    if dpo_beta:
        dpo_kl = [dpo_beta[b]["eval"]["mean_kl_from_reference"] for b in sorted(dpo_beta.keys(), key=float)]
        dpo_rew = [dpo_beta[b]["eval"]["mean_reward"] for b in sorted(dpo_beta.keys(), key=float)]
        ax.plot(dpo_kl, dpo_rew, "o-", label="DPO (β sweep)", color="tab:blue", linewidth=2, markersize=8)
        for b, x, y in zip(sorted(dpo_beta.keys(), key=float), dpo_kl, dpo_rew):
            ax.annotate(f"β={b}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
        has_p1 = True

    if ppo_kl:
        p_kl = [ppo_kl[b]["eval"]["mean_kl"] for b in sorted(ppo_kl.keys(), key=float)]
        p_rew = [ppo_kl[b]["eval"]["mean_reward"] for b in sorted(ppo_kl.keys(), key=float)]
        ax.plot(p_kl, p_rew, "s--", label="PPO (β_KL sweep)", color="tab:orange", linewidth=2, markersize=8)
        for b, x, y in zip(sorted(ppo_kl.keys(), key=float), p_kl, p_rew):
            ax.annotate(f"β_KL={b}", (x, y), textcoords="offset points", xytext=(5, -12), fontsize=9)
        has_p1 = True

    if grpo_norm and "grpo" in grpo_norm:
        g_kl = grpo_norm["grpo"]["eval"]["mean_kl"]
        g_rew = grpo_norm["grpo"]["eval"]["mean_reward"]
        ax.scatter([g_kl], [g_rew], color="tab:green", s=150, zorder=5, label="Standard GRPO")
        ax.annotate("GRPO", (g_kl, g_rew), textcoords="offset points", xytext=(8, 4), fontweight="bold")
        has_p1 = True

    ax.set_xlabel("Policy Drift: Mean KL(π_θ || π_ref)", fontsize=11)
    ax.set_ylabel("Reward Model Score", fontsize=11)
    ax.set_title("Synthesis 1: Reward vs Policy Drift Across Algorithms", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if has_p1:
        ax.legend(fontsize=10)

    # --- 2. Length Shifts Across Optimization Methods ---
    ax = axes[0, 1]
    lengths = {}
    if dpo_beta and "0.10" in dpo_beta:
        lengths["Standard DPO"] = dpo_beta["0.10"]["eval"]["mean_response_length_words"]
    dpo_len = load_json_safe(RESULTS_DIR / "task1_dpo" / "dpo_length_analysis.json")
    if dpo_len and "length_balanced_dpo" in dpo_len:
        lengths["Balanced DPO"] = dpo_len["length_balanced_dpo"]["word_limit"]["mean_word_count"]
    if ppo_kl and "0.10" in ppo_kl:
        lengths["PPO (β=0.1)"] = ppo_kl["0.10"]["eval"]["mean_response_length_tokens"]
    if grpo_norm:
        lengths["GRPO (1/Tk)"] = grpo_norm["grpo"]["eval"]["mean_response_length_tokens"]
        lengths["Dr-GRPO"] = grpo_norm["dr_grpo"]["eval"]["mean_response_length_tokens"]

    if lengths:
        bars = ax.bar(list(lengths.keys()), list(lengths.values()), color=["steelblue", "seagreen", "coral", "mediumpurple", "teal"][:len(lengths)], alpha=0.85, width=0.5)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + 2, f"{h:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.set_xticklabels(list(lengths.keys()), rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Mean Response Length", fontsize=11)
    ax.set_title("Synthesis 2: Response Length Shifts by Objective", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # --- 3. Safety Calibration Map (Over-refusal vs Unsafe compliance) ---
    ax = axes[1, 0]
    if safety_res:
        colors = {"sft": "tab:gray", "dpo": "tab:blue", "ppo": "tab:orange", "grpo": "tab:green"}
        for pol, st in safety_res.items():
            over_ref = st["safe_prompt_over_refusal_rate"] * 100
            unsafe_comp = st["unsafe_prompt_unsafe_compliance_rate"] * 100
            c = colors.get(pol.lower(), "tab:purple")
            ax.scatter(over_ref, unsafe_comp, s=180, color=c, zorder=5, label=pol.upper())
            ax.annotate(pol.upper(), (over_ref, unsafe_comp), textcoords="offset points", xytext=(8, 5), fontweight="bold")
        ax.set_xlabel("Safe Prompt Over-Refusal Rate (%) [Harmlessness Overreach]", fontsize=11)
        ax.set_ylabel("Unsafe Prompt Compliance Rate (%) [Harmful Failure]", fontsize=11)
        ax.set_title("Synthesis 3: Safety Calibration (Ideal = Bottom-Left)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    # --- 4. Verifier vs AI Feedback Generalization (GSM8K -> SVAMP) ---
    ax = axes[1, 1]
    if feedback_syn:
        policies = ["sft", "rlvr", "rlaif"]
        labels = ["SFT Base", "RLVR (Exact)", "RLAIF (Judge)"]
        gsm_acc = [feedback_syn["in_domain_gsm"][p]["exact_accuracy"] * 100 for p in policies]
        svamp_acc = [feedback_syn["transfer_svamp"][p]["exact_accuracy"] * 100 for p in policies]
        x = np.arange(len(policies))
        width = 0.35
        ax.bar(x - width/2, gsm_acc, width, label="In-Domain (GSM8K)", color="tab:blue", alpha=0.85)
        ax.bar(x + width/2, svamp_acc, width, label="Transfer (SVAMP)", color="tab:orange", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Exact Answer Accuracy (%)", fontsize=11)
        ax.set_title("Synthesis 4: Reasoning Generalization (RLVR vs RLAIF)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=10)

    fig.suptitle("Task 6: Master Cross-Task Synthesis & Trade-Off Analysis", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout()
    out_path = FIGURES_DIR / "task6_cross_task_synthesis.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] Master Task 6 Cross-Task Synthesis Figure saved -> {out_path}")


def main():
    print("=" * 80)
    print("ATML PA2 — COMPLETE POST-TRAINING RESEARCH DASHBOARD")
    print(f"Repo Root:    {REPO_ROOT}")
    print(f"Results Dir:  {RESULTS_DIR}")
    print(f"Figures Dir:  {FIGURES_DIR}")
    print("=" * 80)

    inspect_task1()
    inspect_task2()
    inspect_task3()
    inspect_task4()
    inspect_task5()

    try:
        plot_cross_task_synthesis()
    except Exception as e:
        print(f"[Synthesis Plot] Note: {e}")

    # List all figures generated
    figs = list(FIGURES_DIR.glob("*.png"))
    print("\n" + "=" * 80)
    print(f"SAVED FIGURES IN {FIGURES_DIR} ({len(figs)} files):")
    print("=" * 80)
    for f in sorted(figs):
        print(f"  • {f.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
