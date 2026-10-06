"""Report builder for ATML PA2: summary tables (CSV + LaTeX) and cross-task figures.

Reads only saved files under results/; any section whose inputs are missing is skipped with a note.
Run: python -m common.visualize_all
Outputs: report/tables/*.csv|*.tex and report/figures/task6_*.png (+ reprints of key tables).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "report" / "figures"
TABLES_DIR = REPO_ROOT / "report" / "tables"


def load(rel: str):
    p = RESULTS_DIR / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def jsonl(rel: str):
    p = RESULTS_DIR / rel
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def write_table(df: pd.DataFrame, name: str, caption: str):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    try:
        tex = df.to_latex(index=False, escape=True, na_rep="--", caption=caption, label=f"tab:{name}")
    except Exception:
        tex = df.to_string(index=False)
    (TABLES_DIR / f"{name}.tex").write_text(tex, encoding="utf-8")
    print(f"\n### {caption}  [{name}]")
    print(df.to_string(index=False))


def f(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def pm(mean, sd, nd=1):
    return f"{mean:.{nd}f} ± {sd:.{nd}f}"


# --------------------------------------------------------------------------- Task 1
def task1():
    rows = []
    specs = [("SFT (no adapter)", "dpo_eval_sft_reference.json", "—", "—"),
             ("Standard DPO", "dpo_eval_standard.json", "1 epoch, all kept train pairs", "0.10")]
    abl = load("task1_dpo/dpo_beta_ablation_results.json") or {}
    for b in sorted(abl, key=float):
        specs.append((f"Short fork β={float(b):g}", f"dpo_eval_ablation_beta_{float(b):.2f}".replace(".", "_") + ".json",
                      abl[b].get("budget", "short"), f"{float(b):g}"))
    for name, fn, budget, beta in specs:
        e = load(f"task1_dpo/{fn}")
        if e is None:
            continue
        sft = name.startswith("SFT")
        rows.append({"condition": name, "β": beta, "budget": budget,
                     "held-out DPO loss": None if sft else f(e["heldout_dpo_loss"]),
                     "pref. acc.": None if sft else f(e["heldout_preference_accuracy"]),
                     "KL tok": f(e["kl_token_mean"], 4), "KL seq": f(e["kl_sequence_mean"], 2),
                     "RM": f"{e['mean_reward']:.3f} ± {e['sem_reward']:.3f}",
                     "len (tok)": pm(e["length_tokens"]["mean"], e["length_tokens"]["std"]),
                     "IQR": f(e["length_tokens"]["iqr"], 0), "EOS": f(e["eos_rate"], 2)})
    if rows:
        write_table(pd.DataFrame(rows), "t1_dpo_summary",
                    "DPO summary (standard one-epoch run and short β forks use different budgets; RM ± s.e.m.; length mean ± std)")
    la = load("task1_dpo/dpo_length_analysis.json")
    if la and "standard" in la:
        rows = []
        for s in ["preferred_longer", "length_matched", "rejected_longer"]:
            a, b = la["standard"]["stratified"].get(s, {}), la["length_balanced"]["stratified"].get(s, {})
            rows.append({"stratum": s, "n": a.get("total_pairs"), "std acc": f(a.get("accuracy")), "LB acc": f(b.get("accuracy")),
                         "std margin": f(a.get("mean_margin"), 2), "LB margin": f(b.get("mean_margin"), 2),
                         "pairs w/ truncated resp.": a.get("pairs_with_truncated_response")})
        write_table(pd.DataFrame(rows), "t1_length_strata", "Held-out preference accuracy by length stratum")
        rows = []
        for key, lab, ev in [("sft", "SFT", "dpo_eval_sft_reference.json"), ("standard", "standard DPO", "dpo_eval_standard.json"),
                             ("length_balanced", "length-balanced DPO", "dpo_eval_length_balanced.json")]:
            wl = la[key]["word_limit"]
            e = load(f"task1_dpo/{ev}")
            rows.append({"policy": lab, "word-limit compliance (greedy)": f(wl["greedy_compliance_rate"], 2),
                         "compliance (8 samples/prompt)": f(wl.get("sampled_compliance_rate"), 2),
                         "words (greedy)": pm(wl["greedy_word_count"]["mean"], wl["greedy_word_count"]["std"]),
                         "words / limit": f(wl["mean_excess_ratio"], 2),
                         "held-out len (tok)": pm(e["length_tokens"]["mean"], e["length_tokens"]["std"]) if e else None})
        write_table(pd.DataFrame(rows), "t1_word_limit", "Word-limit compliance and generated length")
    pre = load("task1_dpo/dpo_preprocessing_report.json")
    if pre:
        rows = []
        for k, v in pre["files"].items():
            t = v["total"]
            rows.append({"file": k, "pairs": t.get("pairs"), "overlength": t.get("overlength_pairs", 0),
                         "dropped (prompt > limit)": t.get("dropped_here", 0),
                         "response truncated": t.get("either_truncated_here", 0),
                         "released rule: prompt cut": t.get("released_prompt_cut", 0),
                         "frac chosen longer (full)": f(v["chosen_minus_rejected_tokens_full"]["frac_chosen_longer"], 2),
                         "frac chosen longer (after trunc.)": f(v["chosen_minus_rejected_tokens_after_truncation"]["frac_chosen_longer"], 2)})
        write_table(pd.DataFrame(rows), "t1_truncation_policy", "Effect of the 768-token limit under the prompt-preserving policy")


# --------------------------------------------------------------------------- Task 2
def task2():
    rows = []
    for name, fn, cond in [("SFT (no adapter)", "ppo_eval_sft_reference.json", "—"),
                           ("PPO midpoint (start)", "ppo_eval_midpoint.json", "—"),
                           ("Standard PPO (20 updates)", "ppo_eval_standard.json", "ε=0.20, β_KL=0.10")]:
        e = load(f"task2_ppo/{fn}")
        if e:
            rows.append({"policy": name, "setting": cond, "RM": f"{e['mean_reward']:.3f} ± {e['sem_reward']:.3f}",
                         "KL tok": f(e["kl_token_mean"], 4), "entropy": f(e["entropy_exact"], 3),
                         "len": pm(e["length_tokens"]["mean"], e["length_tokens"]["std"]), "EOS": f(e["eos_rate"], 2)})
    clip = load("task2_ppo/ppo_clipping_study_results.json")
    kl = load("task2_ppo/ppo_kl_ablation_results.json")
    for study, data, key in [("clip", clip, "eps_values"), ("kl", kl, "kl_values")]:
        if not data:
            continue
        for v in data[key]:
            fk = data["forks"].get(str(v))
            if not fk:
                continue
            e, t = fk["eval"], fk["train"]
            rows.append({"policy": f"fork ({study} study, 8 updates)", "setting": f"ε={fk['clip_epsilon']}, β_KL={fk['kl_beta']}",
                         "RM": f"{e['mean_reward']:.3f} ± {e['sem_reward']:.3f}", "KL tok": f(e["kl_token_mean"], 4),
                         "entropy": f(e["entropy_exact"], 3), "len": pm(e["length_tokens"]["mean"], e["length_tokens"]["std"]),
                         "EOS": f(e["eos_rate"], 2), "max KL(old,new)": f(t.get("stability_max_approx_kl_old_new"), 5),
                         "grad-norm CV": f(t.get("stability_policy_grad_norm_cv"), 2)})
    if rows:
        write_table(pd.DataFrame(rows), "t2_ppo_heldout", "PPO held-out evaluation (64 fixed prompts, cap 768, RM ± s.e.m.)")
    if clip:
        st = clip["cached_static"]
        pr = clip.get("cached_probe", {})
        rows = [{"ε": e, "static clip frac": f(st[str(e)]["clip_fraction"], 4), "static affected frac": f(st[str(e)]["affected_token_fraction"], 4),
                 "probe clip frac": f(pr.get(str(e), {}).get("geometry", {}).get(str(e), {}).get("clip_fraction"), 4),
                 "probe affected frac": f(pr.get(str(e), {}).get("geometry", {}).get(str(e), {}).get("affected_token_fraction"), 4),
                 "clipped surrogate": f(st[str(e)]["clipped_surrogate"], 4), "unclipped surrogate": f(st[str(e)]["unclipped_surrogate"], 4)}
                for e in clip["eps_values"]]
        write_table(pd.DataFrame(rows), "t2_ppo_cached_clipping", f"Cached-rollout clipping geometry ({clip['cache']['rebuilt']} rollouts)")
    s = load("task2_ppo/ppo_summary_standard.json")
    g = load("task3_grpo/grpo_summary_standard.json")
    d = load("task1_dpo/dpo_summary_standard.json")
    rows = []
    for name, x in [("DPO standard (1 epoch)", d), ("PPO standard (20 updates)", s), ("GRPO standard (20 updates)", g)]:
        if x:
            rows.append({"run": name, "wall-clock (min)": f(x["wall_time_seconds"] / 60, 1), "peak VRAM (GB)": f(x.get("peak_vram_mb", 0) / 1024, 2),
                         "generated tokens": x.get("generated_tokens", "— (offline)")})
    if rows:
        write_table(pd.DataFrame(rows), "t_compute", "Compute for the standard runs")


# --------------------------------------------------------------------------- Task 3
def task3():
    rows = []
    for name, fn in [("SFT (no adapter)", "grpo_eval_sft_reference.json"), ("GRPO midpoint (start)", "grpo_eval_midpoint.json"),
                     ("Standard GRPO (20 updates)", "grpo_eval_standard.json"),
                     ("fork canonical GRPO (8)", "grpo_eval_fork_norm_grpo.json"), ("fork Dr. GRPO (8)", "grpo_eval_fork_norm_dr_grpo.json")]:
        e = load(f"task3_grpo/{fn}")
        if e:
            rows.append({"policy": name, "RM": f"{e['mean_reward']:.3f} ± {e['sem_reward']:.3f}", "KL tok": f(e["kl_token_mean"], 4),
                         "entropy": f(e["entropy_exact"], 3), "len": pm(e["length_tokens"]["mean"], e["length_tokens"]["std"]),
                         "hit cap": f(e["hit_max_tokens_rate"], 2)})
    if rows:
        write_table(pd.DataFrame(rows), "t3_grpo_heldout", "GRPO held-out evaluation (64 fixed prompts, cap 512)")
    gs = load("task3_grpo/grpo_group_size_analysis.json")
    if gs:
        rows = []
        for k in sorted([k for k in gs if not k.startswith("_")], key=int):
            s = gs[k]
            row = {"K": int(k), "groups": s["groups"], "informative": f(s["informative_group_fraction"]),
                   "std>0.1": f(s["meaningful_group_fraction"]), "mean σ_g": f(s["mean_within_group_reward_std"]),
                   "var(adv)": f(s["relative_signal_variance"]), "sign agree": f(s["advantage_sign_agreement"]),
                   "baseline err": f(s["mean_baseline_error"])}
            for dname in ["hard", "medium", "easy"]:
                row[f"sign agree ({dname})"] = f(s["difficulty_breakdown"][dname].get("advantage_sign_agreement"))
            rows.append(row)
        write_table(pd.DataFrame(rows), "t3_group_size", "Equal-generation group-size study (192 cached generations)")
    nm = load("task3_grpo/grpo_normalization_comparison.json")
    if nm:
        rows = []
        for lt in ["grpo", "dr_grpo"]:
            if lt in nm:
                lc, e = nm[lt]["length_conditioned"], nm[lt]["eval"]
                rows.append({"condition": nm[lt]["label"], "RM": f"{e['mean_reward']:.3f} ± {e['sem_reward']:.3f}",
                             "KL tok": f(e["kl_token_mean"], 4), "len": pm(e["length_tokens"]["mean"], e["length_tokens"]["std"]),
                             "ρ_s(len, ‖g‖/|A|)": f(lc["spearman_length_vs_gradnorm_per_unit_adv"]),
                             "grad share long third": f(lc["grad_mass_share_longest_third"]),
                             "grad share short third": f(lc["grad_mass_share_shortest_third"])})
        write_table(pd.DataFrame(rows), "t3_normalization", "Canonical vs Dr. GRPO normalisation")


# --------------------------------------------------------------------------- Task 4
def task4():
    s = load("task4_safety/safety_evaluation_results.json")
    if not s:
        return
    rows = []
    for p in ["sft", "dpo", "ppo", "grpo"]:
        if p not in s:
            continue
        m = s[p]["metrics"]
        cell = lambda k: f"{m[k]['rate'] * 100:.1f} [{m[k]['ci95'][0] * 100:.0f},{m[k]['ci95'][1] * 100:.0f}]"
        rows.append({"policy": p.upper(), "safe answer %": cell("safe_answer"), "over-refusal %": cell("over_refusal"),
                     "any refusal on SAFE %*": cell("safe_any_refusal"), "justified refusal %": cell("justified_refusal"),
                     "unsafe compliance %": cell("unsafe_compliance"), "ambiguous %": cell("ambiguous"),
                     "len (tok)": pm(s[p]["length_tokens"]["mean"], s[p]["length_tokens"]["std"])})
    write_table(pd.DataFrame(rows), "t4_safety", "XSTest safety calibration (AI judge; 95% Wilson CI; *supplementary)")
    a = s.get("_manual_audit")
    if a and a.get("pooled"):
        rows = [{"policy": p.upper(), "n": v["n"], "agreement": f(v["agreement"]), "κ": f(v["cohen_kappa"]),
                 "AI ambiguous": f(v["ai_ambiguous_rate"]), "manual ambiguous": f(v["manual_ambiguous_rate"])}
                for p, v in a["per_policy"].items() if v]
        rows.append({"policy": "pooled", "n": a["pooled"]["n"], "agreement": f(a["pooled"]["agreement"]), "κ": f(a["pooled"]["cohen_kappa"]),
                     "AI ambiguous": f(a["pooled"]["ai_ambiguous_rate"]), "manual ambiguous": f(a["pooled"]["manual_ambiguous_rate"])})
        write_table(pd.DataFrame(rows), "t4_manual_audit", "Manual audit vs AI judge (60 fixed prompts per policy)")


# --------------------------------------------------------------------------- Task 5
def task5():
    g, t, d = load("task5_feedback/math_eval_gsm.json"), load("task5_feedback/math_eval_transfer.json"), load("task5_feedback/perturbation_scores.json")
    if g and t:
        rows = []
        for p in ["sft", "rlvr", "rlaif"]:
            a, b = g["per_policy"][p], t["per_policy"][p]
            rows.append({"policy": p.upper(), "GSM acc": f(a["exact_accuracy"]), "GSM format": f(a["format_compliance_rate"]),
                         "GSM len": pm(a["length_tokens"]["mean"], a["length_tokens"]["std"], 0),
                         "SVAMP acc": f(b["exact_accuracy"]), "SVAMP len": pm(b["length_tokens"]["mean"], b["length_tokens"]["std"], 0),
                         "acc drop": f(a["exact_accuracy"] - b["exact_accuracy"])})
        for k in ["rlvr_vs_sft", "rlaif_vs_sft", "rlvr_vs_rlaif"]:
            pa, pb = g["pairwise_comparisons"][k], t["pairwise_comparisons"][k]
            rows.append({"policy": k.replace("_vs_", " vs ").upper(), "GSM acc": f"win {pa['win_rate_a_ties_half']:.3f}",
                         "GSM format": f"ties {pa['explicit_ties']}/unparsed {pa['unparsed_judge_outputs']}",
                         "GSM len": f"agree {pa['agreement_on_verifier_decisive']:.2f}" if pa["agreement_on_verifier_decisive"] is not None else "",
                         "SVAMP acc": f"win {pb['win_rate_a_ties_half']:.3f}", "SVAMP len": "",
                         "acc drop": f(pa["win_rate_a_ties_half"] - pb["win_rate_a_ties_half"])})
        write_table(pd.DataFrame(rows), "t5_math", "RLVR vs RLAIF: exact accuracy, AI-judge win rate (tie=0.5), verifier agreement")
    amb = load("task5_feedback/judge_ambiguity.json")
    if amb:
        rows = []
        for ds in ["gsm", "transfer", "diagnostics"]:
            for k, v in (amb.get(ds) or {}).items():
                if isinstance(v, dict) and "n" in v:
                    rows.append({"set": ds, "comparison": k, "n": v["n"], "decisive A/B": v["decisive"],
                                 "explicit TIE": v["explicit_tie"], "ambiguous (echo/unparsed)": v["ambiguous"]})
        write_table(pd.DataFrame(rows), "t5_judge_outputs",
                    "Raw pairwise-judge outputs: ambiguous = several or no labels (released parser keeps the first)")
    if d:
        rows = []
        for c, v in d["comparisons"].items():
            rows.append({"perturbation": c, "verifier better/tie/worse": f"{v['rlvr']['better_rate']:.2f}/{v['rlvr']['tie_rate']:.2f}/{v['rlvr']['worse_rate']:.2f}",
                         "judge better/tie/worse": f"{v['rlaif']['better_rate']:.2f}/{v['rlaif']['tie_rate']:.2f}/{v['rlaif']['worse_rate']:.2f}",
                         "judge order consistency": f(v["rlaif_order_consistency"], 2)})
        write_table(pd.DataFrame(rows), "t5_diagnostics",
                    f"Controlled diagnostics (S_reason verifier {d['s_reason']['rlvr']:.2f} / judge {d['s_reason']['rlaif']:.2f}; "
                    f"S_outcome verifier {d['s_outcome']['rlvr']:.2f} / judge {d['s_outcome']['rlaif']:.2f})")


# --------------------------------------------------------------------------- Task 6
def plot_cross_task_synthesis():
    pts = []  # (method, label, kl, d_reward, d_len, eval set)
    d_ref = load("task1_dpo/dpo_eval_sft_reference.json")
    if d_ref:
        for lab, fn in [("DPO std", "dpo_eval_standard.json"), ("DPO β=.03", "dpo_eval_ablation_beta_0_03.json"),
                        ("DPO β=.10", "dpo_eval_ablation_beta_0_10.json"), ("DPO β=.30", "dpo_eval_ablation_beta_0_30.json"),
                        ("DPO len-bal", "dpo_eval_length_balanced.json")]:
            e = load(f"task1_dpo/{fn}")
            if e:
                pts.append(("DPO", lab, e["kl_token_mean"], e["mean_reward"] - d_ref["mean_reward"],
                            e["length_tokens"]["mean"] - d_ref["length_tokens"]["mean"]))
    for method, task, start, files in [
        ("PPO", "task2_ppo", "ppo_eval_midpoint.json", [("PPO std", "ppo_eval_standard.json")] +
         [(f"PPO {n}", f"ppo_eval_{n}.json") for n in ["fork_eps0_05_kl0_10", "fork_eps0_20_kl0_10", "fork_eps0_50_kl0_10",
                                                       "fork_eps0_20_kl0_00", "fork_eps0_20_kl0_20"]]),
        ("GRPO", "task3_grpo", "grpo_eval_midpoint.json", [("GRPO std", "grpo_eval_standard.json"),
                                                           ("GRPO canon", "grpo_eval_fork_norm_grpo.json"),
                                                           ("Dr.GRPO", "grpo_eval_fork_norm_dr_grpo.json")])]:
        s0 = load(f"{task}/{start}")
        if not s0:
            continue
        pts.append((method, f"{method} midpoint", s0["kl_token_mean"], 0.0, 0.0))
        for lab, fn in files:
            e = load(f"{task}/{fn}")
            if e:
                pts.append((method, lab.replace("fork_", "").replace("_kl", " kl").replace("eps", "ε"), e["kl_token_mean"],
                            e["mean_reward"] - s0["mean_reward"], e["length_tokens"]["mean"] - s0["length_tokens"]["mean"]))
    safety = load("task4_safety/safety_evaluation_results.json")
    if not pts and not safety:
        print("[Task 6] nothing to plot yet")
        return
    cols = {"DPO": "#2563EB", "PPO": "#DC2626", "GRPO": "#16A34A"}
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    for m, lab, kl, dr, dl in pts:
        axes[0].scatter(kl, dr, color=cols[m], s=45)
        axes[0].annotate(lab, (kl, dr), fontsize=7, xytext=(3, 3), textcoords="offset points")
        axes[1].scatter(kl, dl, color=cols[m], s=45)
        axes[1].annotate(lab, (kl, dl), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for ax, yl, t in [(axes[0], "Δ RM score vs own start", "(a) Reward change vs drift"),
                      (axes[1], "Δ generated tokens vs own start", "(b) Length change vs drift")]:
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xlabel("held-out sampled KL to the frozen reference (token mean)")
        ax.set_ylabel(yl)
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
    for m, c in cols.items():
        axes[0].scatter([], [], color=c, label=m)
    axes[0].legend(frameon=False)
    if safety:
        for p, c in zip(["sft", "dpo", "ppo", "grpo"], ["#6B7280", "#2563EB", "#DC2626", "#16A34A"]):
            if p in safety:
                m = safety[p]["metrics"]
                x, y = m["safe_any_refusal"]["rate"] * 100, m["unsafe_compliance"]["rate"] * 100
                axes[2].errorbar(x, y, xerr=[[x - m["safe_any_refusal"]["ci95"][0] * 100], [m["safe_any_refusal"]["ci95"][1] * 100 - x]],
                                 yerr=[[y - m["unsafe_compliance"]["ci95"][0] * 100], [m["unsafe_compliance"]["ci95"][1] * 100 - y]],
                                 fmt="o", color=c, capsize=3, label=p.upper())
        axes[2].set_xlabel("refusal-type labels on SAFE prompts (%)")
        axes[2].set_ylabel("unsafe compliance on UNSAFE prompts (%)")
        axes[2].set_title("(c) Safety calibration (lower-left is better)", fontsize=10)
        axes[2].legend(frameon=False)
        axes[2].grid(alpha=0.3)
    fig.suptitle("Task 6 — cross-task synthesis (DPO measured on UltraFeedback held-out pairs' prompts; PPO/GRPO on the RL held-out pool; "
                 "Δ relative to each method's own starting policy)", fontsize=10)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "task6_cross_task_synthesis.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {FIGURES_DIR / 'task6_cross_task_synthesis.png'}")


def plot_standard_trajectories():
    """One compact figure with the standard PPO and GRPO continuation trajectories side by side."""
    ppo, grpo = jsonl("task2_ppo/ppo_train_standard.jsonl"), jsonl("task3_grpo/grpo_train_standard.jsonl")
    if not ppo and not grpo:
        return
    keys = [("reward_rm", "reward_mean", "reward"), ("kl_token_mean", "kl_token_mean", "sampled KL"),
            ("entropy_exact", "entropy_exact", "entropy"), ("response_length", "response_length", "length (tokens)")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.8))
    for ax, (kp, kg, t) in zip(axes, keys):
        if ppo:
            ax.plot([r["update"] for r in ppo], [r[kp] for r in ppo], "-o", ms=3, color="#DC2626", label="PPO")
        if grpo:
            ax.plot([r["update"] for r in grpo], [r[kg] for r in grpo], "-s", ms=3, color="#16A34A", label="GRPO")
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("update")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    fig.suptitle("Standard continuations from the supplied midpoints (rollout statistics, 1 prompt per update)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "task6_standard_rl_trajectories.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    for fn in (task1, task2, task3, task4, task5):
        try:
            fn()
        except Exception as e:
            print(f"[warn] {fn.__name__}: {e}")
    for fn in (plot_cross_task_synthesis, plot_standard_trajectories):
        try:
            fn()
        except Exception as e:
            print(f"[warn] {fn.__name__}: {e}")
    print(f"\nTables in {TABLES_DIR}, figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
