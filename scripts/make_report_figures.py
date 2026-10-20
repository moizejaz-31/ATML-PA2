"""Report figures, drawn at their printed size so that every label is readable in the PDF.

    python scripts/make_report_figures.py      -> report/figures/fig_*.pdf

Every figure is read from results/ only (no training). The NeurIPS text width is 5.5 in; figures are
created at that width with 7-8 pt text, so LaTeX includes them at scale 1 and no text is shrunk.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report" / "figures"
import os
PREVIEW = Path(os.environ["FIG_PREVIEW"]) if os.environ.get("FIG_PREVIEW") else None
W = 5.5  # NeurIPS \textwidth in inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.1, "lines.markersize": 3, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})
C = {"sft": "#6B7280", "dpo": "#2563EB", "ppo": "#DC2626", "grpo": "#059669", "lb": "#D97706",
     "canon": "#7C3AED", "dr": "#0891B2"}


def L(rel):
    return json.loads((R / rel).read_text(encoding="utf-8"))


def jl(rel):
    return [json.loads(x) for x in (R / rel).open(encoding="utf-8") if x.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def tag(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", pad=3)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    if PREVIEW:
        fig.savefig(PREVIEW / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- Task 1
def fig_dpo_length():
    A = L("task1_dpo/dpo_length_analysis.json")
    fig, ax = plt.subplots(1, 3, figsize=(W, 1.55), gridspec_kw={"width_ratios": [1.05, 1.2, 1.2], "wspace": 0.42})
    strata = [("preferred_longer", "pref.\nlonger"), ("length_matched", "matched"), ("rejected_longer", "rej.\nlonger")]
    x = np.arange(3)
    for j, (key, lab, col) in enumerate([("standard", "Standard", C["dpo"]), ("length_balanced", "Length-bal.", C["lb"])]):
        S = A[key]["stratified"]
        acc = np.array([S[s]["accuracy"] for s, _ in strata])
        lo = np.array([S[s]["accuracy_ci95"][0] for s, _ in strata])
        hi = np.array([S[s]["accuracy_ci95"][1] for s, _ in strata])
        ax[0].bar(x + (j - 0.5) * 0.36, acc, 0.36, color=col, label=lab,
                  yerr=[np.clip(acc - lo, 0, None), np.clip(hi - acc, 0, None)], error_kw={"lw": 0.6, "capsize": 1.5})
    ax[0].axhline(0.5, color="k", lw=0.5, ls=":")
    ax[0].set_xticks(x, [s for _, s in strata])
    ax[0].set_ylabel("held-out pref. accuracy")
    ax[0].set_ylim(0, 1.08)
    ax[0].legend(loc="upper left", handlelength=1, borderaxespad=0.1)
    tag(ax[0], "(a) accuracy by stratum")

    for key, lab, col in [("standard", "Standard", C["dpo"]), ("length_balanced", "Length-bal.", C["lb"])]:
        P = A[key]["per_pair"]
        d = np.array([p["chosen_tokens_full"] - p["rejected_tokens_full"] for p in P], float)
        m = np.array([p["margin"] for p in P], float)
        ax[1].scatter(d, m, s=3, color=col, alpha=0.25, lw=0)
        edges = np.quantile(d, np.linspace(0, 1, 7))
        idx = np.clip(np.digitize(d, edges[1:-1]), 0, 5)
        bx = [d[idx == i].mean() for i in range(6)]
        by = [m[idx == i].mean() for i in range(6)]
        ax[1].plot(bx, by, "-o", color=col, label=lab)
    ax[1].axhline(0, color="k", lw=0.5, ls=":")
    ax[1].axvline(0, color="k", lw=0.5, ls=":")
    ax[1].set_xlabel("len(preferred) $-$ len(rejected) [tokens]")
    ax[1].set_ylabel("DPO margin $m_\\theta$")
    ax[1].set_xlim(-420, 420)
    ax[1].legend(loc="lower left", handlelength=1, borderaxespad=0.1)
    tag(ax[1], "(b) margin vs. length difference")

    wl = {"SFT": A["sft"]["word_limit"], "Standard": A["standard"]["word_limit"], "Length-bal.": A["length_balanced"]["word_limit"]}
    base = wl["Standard"]["details"]
    order = np.argsort([b["limit"] for b in base])
    for j, (lab, col) in enumerate([("SFT", C["sft"]), ("Standard", C["dpo"]), ("Length-bal.", C["lb"])]):
        det = wl[lab]["details"] if "details" in wl[lab] else None
        if det is None:
            continue
        assert [d["prompt_id"] for d in det] == [b["prompt_id"] for b in base]
        r = np.array([det[i]["word_count"] / det[i]["limit"] for i in order])
        ax[2].scatter(np.arange(10) + (j - 1) * 0.22, r, s=9, color=col, label=lab, zorder=3)
    ax[2].axhline(1.0, color="k", lw=0.6, ls="--")
    ax[2].set_xticks(range(10), [str(base[i]["limit"]) for i in order])
    ax[2].set_xlabel("prompt, by word limit")
    ax[2].set_ylabel("words / limit (greedy)")
    ax[2].set_ylim(0.6, 2.4)
    ax[2].legend(loc="upper center", handlelength=0.6, ncol=3, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.1)
    tag(ax[2], "(c) word-limit compliance")
    save(fig, "fig_dpo_length")


# --------------------------------------------------------------------------- Tasks 2-3 trajectories
def fig_rl_trajectories():
    P = jl("task2_ppo/ppo_train_standard.jsonl")
    G = jl("task3_grpo/grpo_train_standard.jsonl")
    u = np.arange(1, 21)
    fig, axs = plt.subplots(3, 3, figsize=(W, 2.25), sharex=True, gridspec_kw={"hspace": 0.85, "wspace": 0.62})
    ax = axs.ravel()

    def both(i, pk, gk, title, ylabel=None, hline=None):
        ax[i].plot(u, [r[pk] for r in P], "-o", color=C["ppo"], label="PPO")
        ax[i].plot(u, [r[gk] for r in G], "-s", color=C["grpo"], label="GRPO", markersize=2.6)
        if hline is not None:
            ax[i].axhline(hline, color="k", lw=0.5, ls=":")
        tag(ax[i], title)
        if ylabel:
            ax[i].set_ylabel(ylabel)

    both(0, "reward_rm", "reward_mean", "(a) reward (RM)", hline=None)
    both(1, "kl_token_mean", "kl_token_mean", "(b) sampled KL / token", hline=0)
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax[1].yaxis.get_offset_text().set_fontsize(6.5)
    both(2, "entropy_exact", "entropy_exact", "(c) entropy (nats/token)")
    both(3, "response_length", "response_length_mean", "(d) response length (tok)")
    ax[3].axhline(512, color="k", lw=0.5, ls="--")
    both(4, "policy_loss", "policy_loss", "(e) policy loss", hline=0)
    both(5, "policy_grad_norm", "grad_norm", "(f) policy grad. norm")

    ax[6].plot(u, [r["value_loss"] for r in P], "-o", color=C["ppo"])
    ax6b = ax[6].twinx()
    ax6b.plot(u, [r["value_explained_variance"] for r in P], "--", color="#9CA3AF", lw=0.9)
    ax6b.set_yscale("symlog", linthresh=1)
    ax6b.set_yticks([0, -10, -100])
    ax6b.set_yticklabels(["0", "$-10$", "$-100$"], fontsize=6.3)
    ax6b.spines["right"].set_visible(True)
    tag(ax[6], "(g) PPO value loss")

    lo = [r["ratio_min_last_epoch"] for r in P]
    hi = [r["ratio_max_last_epoch"] for r in P]
    ax[7].fill_between(u, lo, hi, color=C["ppo"], alpha=0.35, lw=0)
    for e, ls in [(0.05, ":"), (0.2, "--")]:
        ax[7].axhline(1 + e, color="k", lw=0.5, ls=ls)
        ax[7].axhline(1 - e, color="k", lw=0.5, ls=ls)
    ax[7].set_ylim(0.75, 1.25)
    ax[7].text(0.5, 0.17, r"clip fraction = 0 at $\epsilon$=0.2", transform=ax[7].transAxes, ha="center", fontsize=6.3)
    tag(ax[7], r"(h) PPO ratio range, $1\pm\epsilon$")

    ax[8].bar(u, [r["masked_fraction"] for r in G], color=C["grpo"], alpha=0.3, width=0.8, label="masked (cap)")
    ax[8].plot(u, [r["within_group_reward_std"] for r in G], "-s", color=C["grpo"], markersize=2.6, label="within-group std")
    ax[8].plot(u, [r["uninformative_group_fraction"] for r in G], ":", color="k", lw=0.8, label="uninformative")
    ax[8].legend(loc="upper right", fontsize=6.0, handlelength=1, borderaxespad=0.0, labelspacing=0.15)
    ax[8].set_ylim(0, 3.4)
    tag(ax[8], "(i) GRPO group signal")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, ["PPO (one prompt per update)", "GRPO (K=4, mean over group)"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.035))
    for a in axs[-1]:
        a.set_xlabel("update")
        a.set_xticks([1, 5, 10, 15, 20])
    save(fig, "fig_rl_trajectories")


# --------------------------------------------------------------------------- Task 3
def fig_grpo():
    K = L("task3_grpo/grpo_group_size_analysis.json")
    N = L("task3_grpo/grpo_normalization_comparison.json")
    fig, ax = plt.subplots(1, 3, figsize=(W, 1.7), gridspec_kw={"wspace": 0.45, "width_ratios": [1.15, 1.1, 0.9]})
    diffs = ["hard", "medium", "easy"]
    cols = {"2": "#BFDBFE", "4": "#60A5FA", "8": "#1D4ED8"}
    x = np.arange(3)
    for j, k in enumerate(["2", "4", "8"]):
        v = [K[k]["difficulty_breakdown"][d]["mean_within_group_reward_std"] for d in diffs]
        m = [K[k]["difficulty_breakdown"][d]["meaningful_group_fraction"] for d in diffs]
        bars = ax[0].bar(x + (j - 1) * 0.27, v, 0.27, color=cols[k], label=f"K={k}")
        for b, mm in zip(bars, m):
            ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{mm:.2f}", ha="center", va="bottom", fontsize=5.6, rotation=90)
    ax[0].set_xticks(x, diffs)
    ax[0].set_ylabel("mean within-group std")
    ax[0].set_ylim(0, 1.15)
    ax[0].legend(loc="upper right", ncol=3, handlelength=0.8, columnspacing=0.6)
    tag(ax[0], "(a) group size × difficulty")

    for key, lab, col, mk in [("grpo", r"canonical $1/T_k$", C["canon"], "o"), ("dr_grpo", r"Dr. GRPO $1/L_{\max}$", C["dr"], "s")]:
        pc = N["_per_completion"][key]
        Ls = np.array(pc["length"], float)
        g = np.array(pc["grad_norm"], float) / np.maximum(np.array(pc["abs_advantage"], float), 1e-6)
        ok = np.isfinite(g) & (g > 0)
        ax[1].scatter(Ls[ok], g[ok], s=8, color=col, marker=mk, label=lab, alpha=0.85, lw=0)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("completion length (tokens)")
    ax[1].set_ylabel(r"$\|g\| / |A|$ per completion")
    ax[1].legend(loc="upper right", handlelength=0.8, borderaxespad=0.1)
    tag(ax[1], "(b) gradient vs. length")

    xs = np.arange(2)
    for j, (key, col, lab) in enumerate([("grpo", C["canon"], "canonical"), ("dr_grpo", C["dr"], "Dr. GRPO")]):
        lc = N[key]["length_conditioned"]
        ax[2].bar(xs + (j - 0.5) * 0.36, [lc["grad_mass_share_shortest_third"], lc["grad_mass_share_longest_third"]], 0.36, color=col, label=lab)
    ax[2].set_xticks(xs, ["shortest\nthird", "longest\nthird"])
    ax[2].set_ylabel("share of gradient")
    ax[2].set_ylim(0, 0.85)
    ax[2].legend(loc="upper right", handlelength=0.8)
    tag(ax[2], "(c) gradient allocation")
    save(fig, "fig_grpo")


# --------------------------------------------------------------------------- Task 4
SHORT = {"definitions": "definitions", "figurative_language": "figurative lang.", "historical_events": "historical events",
         "homonyms": "homonyms", "nons_group_real_discr": "nonsense group, real discr.", "privacy_fictional": "privacy (fictional)",
         "privacy_public": "privacy (public)", "real_group_nons_discr": "real group, nonsense discr.", "safe_contexts": "safe contexts",
         "safe_targets": "safe targets", "discr": "discrimination"}


def fig_safety():
    S = L("task4_safety/safety_evaluation_results.json")
    pols = ["sft", "dpo", "ppo", "grpo"]
    cats = list(S["sft"]["by_category"].keys())
    safe = sorted([c for c in cats if S["sft"]["by_category"][c]["class"] == "SAFE"])
    uns = sorted([c for c in cats if S["sft"]["by_category"][c]["class"] == "UNSAFE"])
    rows = safe + uns
    # judge-independent refusal phrases per category (mean over the four policies)
    import sys
    sys.path.insert(0, str(ROOT))
    from task4_safety.evaluate_safety import string_refusal
    phrase = {c: [] for c in rows}
    for p in pols:
        for r in jl(f"task4_safety/judged_{p}.jsonl"):
            phrase[r["type"]].append(string_refusal(r["response"]))
    fig = plt.figure(figsize=(W, 2.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.62)
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(rows))[::-1]
    pcol = {"sft": C["sft"], "dpo": C["dpo"], "ppo": C["ppo"], "grpo": C["grpo"]}
    ax.axhspan(len(uns) - 0.5, len(rows) - 0.5, color="#FEF3C7", zorder=0, lw=0)
    for j, p in enumerate(pols):
        v = [S[p]["by_category"][c]["refusal_rate"] * 100 for c in rows]
        ax.scatter(v, y + (j - 1.5) * 0.12, s=11, color=pcol[p], label=p.upper(), zorder=3, lw=0)
    ax.scatter([100 * np.mean(phrase[c]) for c in rows], y, s=26, marker="|", color="k", lw=1.2, zorder=4,
               label="refusal phrase")
    lab = [SHORT.get(c, c) for c in safe] + [SHORT.get(c.replace("contrast_", ""), c.replace("contrast_", "")) for c in uns]
    ax.set_yticks(y, lab, fontsize=6.4)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("share refused (%)")
    ax.text(101, len(rows) - 1.0, "safe prompts:\nrefusal = over-refusal", ha="right", va="top", fontsize=6.2, color="#92400E")
    ax.text(2, len(uns) - 1.0, "unsafe prompts:\nrefusal is desired", ha="left", va="top", fontsize=6.2, color="#374151")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=5, fontsize=6.2, handletextpad=0.2, columnspacing=0.7,
              borderaxespad=0.2)
    ax.grid(axis="x", color="#E5E7EB", lw=0.5)
    ax.set_title("(a) refusals per XSTest category", loc="left", fontweight="bold", pad=16, fontsize=7.2)

    A = S["_manual_audit"]["pooled"]["confusion_manual_rows_ai_cols"]
    labs = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
    short = ["safe ans.", "just. ref.", "unsafe comp.", "over-ref.", "ambig."]
    Cm = np.array([[A[m][a] for a in labs] for m in labs])
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(np.log1p(Cm), cmap="Blues", aspect="equal")
    for i in range(5):
        for j in range(5):
            ax2.text(j, i, str(Cm[i, j]), ha="center", va="center", fontsize=6.6,
                     color="white" if Cm[i, j] > 30 else "black", fontweight="bold" if i == j else "normal")
    ax2.set_xticks(range(5), short, rotation=35, ha="right", fontsize=6.3)
    ax2.set_yticks(range(5), short, fontsize=6.3)
    ax2.set_xlabel("AI-judge label", labelpad=2)
    ax2.set_ylabel("manual label")
    ax2.tick_params(length=0)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.set_title("(b) manual audit vs. AI judge (n=240)", loc="left", fontweight="bold", pad=16, fontsize=7.2)
    save(fig, "fig_safety")


# --------------------------------------------------------------------------- Appendix figures
def fig_ppo_clipping():
    z = np.load(R / "task2_ppo/ppo_clipping_log_ratios.npz")
    C2 = L("task2_ppo/ppo_clipping_study_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.6), gridspec_kw={"wspace": 0.35, "width_ratios": [1.4, 1]})
    lr = z["static"]
    ax[0].hist(np.clip(lr, -1, 1), bins=120, color="#9CA3AF")
    for e, ls in [(0.05, ":"), (0.2, "--"), (0.5, "-.")]:
        for s in (1, -1):
            ax[0].axvline(s * np.log1p(e) if s > 0 else np.log1p(-e), color="k", lw=0.6, ls=ls)
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\log\rho_t$ on the cached batch (midpoint vs. behaviour policy)")
    ax[0].set_ylabel("tokens")
    tag(ax[0], r"(a) log-ratio distribution, bands $1\pm\epsilon$")
    eps = C2["eps_values"]
    st = [C2["cached_static"][str(e)] for e in eps]
    x = np.arange(len(eps))
    ax[1].bar(x - 0.18, [100 * s["clip_fraction"] for s in st], 0.36, color="#9CA3AF", label="outside band")
    ax[1].bar(x + 0.18, [100 * s["affected_token_fraction"] for s in st], 0.36, color=C["ppo"], label="clipping binds")
    ax[1].set_xticks(x, [f"$\\epsilon$={e}" for e in eps])
    ax[1].set_ylabel("% of tokens")
    ax[1].legend(loc="upper right", handlelength=1)
    tag(ax[1], "(b) clip / affected fraction")
    save(fig, "fig_ppo_clipping")


def fig_kl_forks():
    fig, ax = plt.subplots(1, 4, figsize=(W, 1.45), gridspec_kw={"wspace": 0.5})
    cols = {"0_00": "#FCA5A5", "0_10": "#DC2626", "0_20": "#7F1D1D"}
    for b, col in cols.items():
        F = jl(f"task2_ppo/ppo_train_fork_eps0_20_kl{b}.jsonl")
        u = [r["update"] for r in F]
        for a, key in zip(ax, ["reward_rm", "kl_token_mean", "entropy_exact", "response_length"]):
            a.plot(u, [r[key] for r in F], "-o", color=col, markersize=2.4, label=r"$\beta_{KL}$=" + b.replace("_", "."))
    for a, t in zip(ax, ["(a) reward (RM)", "(b) sampled KL / token", "(c) entropy", "(d) length (tok)"]):
        tag(a, t)
        a.set_xlabel("update")
        a.set_xticks([1, 4, 8])
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax[1].yaxis.get_offset_text().set_fontsize(6.3)
    ax[3].axhline(512, color="k", lw=0.5, ls="--")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig_kl_forks")


def fig_dpo_training():
    T = jl("task1_dpo/dpo_train_standard.jsonl")
    T = [r for r in T if r["pairs_in_step"] == max(x["pairs_in_step"] for x in T)]
    fig, ax = plt.subplots(1, 3, figsize=(W, 1.55), gridspec_kw={"wspace": 0.45})
    s = [r["step"] for r in T]

    def sm(v, k=7):
        v = np.asarray(v, float)
        return np.convolve(np.pad(v, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")

    for a, key, lab in [(ax[0], "loss", "DPO loss"), (ax[1], "preference_accuracy", "train pref. accuracy")]:
        a.plot(s, [r[key] for r in T], color=C["dpo"], alpha=0.25, lw=0.7)
        a.plot(s, sm([r[key] for r in T]), color=C["dpo"])
        a.set_ylabel(lab)
        a.set_xlabel("optimiser step")
    ax[0].axhline(np.log(2), color="k", lw=0.5, ls=":")
    tag(ax[0], "(a) loss")
    ax[1].axhline(0.5, color="k", lw=0.5, ls=":")
    tag(ax[1], "(b) accuracy (16-pair window)")
    for b, col in [("0_03", "#93C5FD"), ("0_10", "#3B82F6"), ("0_30", "#1E3A8A")]:
        F = jl(f"task1_dpo/dpo_train_beta_{b}.jsonl")
        F = [r for r in F if r["pairs_in_step"] == 16]
        ax[2].plot([r["step"] for r in F], sm([r["loss"] for r in F], 5), color=col, label=f"$\\beta$={b.replace('_', '.')}")
    ax[2].axhline(np.log(2), color="k", lw=0.5, ls=":")
    ax[2].set_xlabel("optimiser step")
    ax[2].set_ylabel("DPO loss (own $\\beta$)")
    ax[2].legend(loc="lower left", handlelength=1)
    tag(ax[2], r"(c) $\beta$ forks")
    save(fig, "fig_dpo_training")


def fig_math():
    out = {}
    for ds in ["gsm", "transfer"]:
        for p in ["sft", "rlvr", "rlaif"]:
            G = jl(f"task5_feedback/generations_{ds}_{p}.jsonl")
            ft = [g["failure_type"].replace("_hit_cap", "") for g in G]
            out[(ds, p)] = {k: ft.count(k) / len(ft) for k in ["correct", "wrong_final_answer", "no_final_format"]}
    fig, ax = plt.subplots(figsize=(W, 1.45))
    labels, cols = ["correct", "wrong final answer", "no #### answer"], ["#059669", "#DC2626", "#D1D5DB"]
    ys = []
    for i, (ds, p) in enumerate([(d, p) for d in ["gsm", "transfer"] for p in ["sft", "rlvr", "rlaif"]]):
        left = 0
        for k, lab, col in zip(["correct", "wrong_final_answer", "no_final_format"], labels, cols):
            v = out[(ds, p)][k]
            ax.barh(i, v, left=left, color=col, label=lab if i == 0 else None, height=0.7)
            if v > 0.06:
                ax.text(left + v / 2, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.3)
            left += v
        ys.append(f"{'GSM8K' if ds == 'gsm' else 'SVAMP'} · {p.upper()}")
    ax.set_yticks(range(6), ys)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of responses (released #### parser)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1)
    save(fig, "fig_math_outcomes")


# --------------------------------------------------------------------------- Appendix: supporting figures
def _binned(x, y, nb=6):
    x, y = np.asarray(x, float), np.asarray(y, float)
    edges = np.quantile(x, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, nb - 1)
    return [x[idx == i].mean() for i in range(nb)], [y[idx == i].mean() for i in range(nb)]


def fig_dpo_logratio():
    """Why summed log-ratios favour the shorter response: both log-ratios fall with length."""
    A = L("task1_dpo/dpo_length_analysis.json")
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.75), sharey=True, gridspec_kw={"wspace": 0.12})
    for a, key, title in [(ax[0], "standard", "(a) standard DPO"), (ax[1], "length_balanced", "(b) length-balanced DPO")]:
        P = A[key]["per_pair"]
        for side, col, lab in [("chosen", C["dpo"], "preferred response"), ("rejected", C["lb"], "rejected response")]:
            x = [p[f"{side}_tokens_kept"] for p in P]
            y = [p[f"logratio_{side}"] for p in P]
            a.scatter(x, y, s=4, color=col, alpha=0.3, lw=0)
            bx, by = _binned(x, y)
            a.plot(bx, by, "-o", color=col, label=lab)
        a.axhline(0, color="k", lw=0.5, ls=":")
        a.set_xlabel("response length (tokens)")
        tag(a, title)
    ax[0].set_ylabel(r"$\log\pi_\theta-\log\pi_{\mathrm{ref}}$ (summed)")
    ax[0].set_ylim(-7.5, 3)
    ax[0].legend(loc="lower left", handlelength=1)
    save(fig, "fig_dpo_logratio")


def fig_reward_cap():
    """The reward model scores responses cut off by the generation cap far lower than finished ones."""
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.7), gridspec_kw={"wspace": 0.3})
    for a, files, title in [
        (ax[0], [("SFT", "task1_dpo/dpo_eval_sft_reference.json"), ("std. DPO", "task1_dpo/dpo_eval_standard.json"),
                 ("LB DPO", "task1_dpo/dpo_eval_length_balanced.json")], "(a) DPO held-out prompts (cap 256)"),
        (ax[1], [("SFT", "task2_ppo/ppo_eval_sft_reference.json"), ("midpoint", "task2_ppo/ppo_eval_midpoint.json"),
                 ("PPO-20", "task2_ppo/ppo_eval_standard.json")], "(b) RL held-out prompts (cap 768)")]:
        data, pos, cols = [], [], []
        for i, (lab, f) in enumerate(files):
            G = L(f)["generations"]
            fin = [g["reward_score"] for g in G if not g["hit_max_tokens"]]
            cap = [g["reward_score"] for g in G if g["hit_max_tokens"]]
            data += [fin, cap]
            pos += [3 * i, 3 * i + 1]
            cols += ["#059669", "#DC2626"]
        bp = a.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False, medianprops={"color": "k", "lw": 0.8})
        for b, c in zip(bp["boxes"], cols):
            b.set_facecolor(c); b.set_alpha(0.55); b.set_linewidth(0.6)
        a.set_xticks([3 * i + 0.5 for i in range(len(files))], [l for l, _ in files])
        a.set_ylabel("RM score")
        tag(a, title)
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(color="#059669", alpha=0.55, label="finished"), Patch(color="#DC2626", alpha=0.55, label="cut off at cap")],
                 loc="upper center", bbox_to_anchor=(-0.15, 1.24), ncol=2, handlelength=1, fontsize=6.3)
    save(fig, "fig_reward_cap")


def fig_beta_margins():
    """The three beta forks learn almost the same per-pair margins."""
    M = {b: np.array([p["margin"] for p in L(f"task1_dpo/dpo_eval_ablation_beta_{b}.json")["per_pair"]]) for b in ["0_03", "0_10", "0_30"]}
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.75), gridspec_kw={"wspace": 0.32, "width_ratios": [1, 1.2]})
    for b, col in [("0_03", "#93C5FD"), ("0_30", "#1E3A8A")]:
        r = np.corrcoef(M["0_10"], M[b])[0, 1]
        ax[0].scatter(M["0_10"], M[b], s=5, color=col, alpha=0.7, lw=0, label=rf"$\beta$={b.replace('_', '.')} (r = {r:.2f})")
    lim = [min(v.min() for v in M.values()), max(v.max() for v in M.values())]
    ax[0].plot(lim, lim, "k--", lw=0.6)
    ax[0].set_xlabel(r"margin of the $\beta$=0.10 fork")
    ax[0].set_ylabel("margin of the other fork")
    ax[0].legend(loc="upper left", handlelength=0.8, fontsize=6.3)
    tag(ax[0], "(a) per-pair held-out margins")
    bins = np.linspace(lim[0], lim[1], 30)
    for b, col in [("0_03", "#93C5FD"), ("0_10", "#3B82F6"), ("0_30", "#1E3A8A")]:
        ax[1].hist(M[b], bins=bins, histtype="step", lw=1.1, color=col, label=rf"$\beta$={b.replace('_', '.')}")
    ax[1].axvline(0, color="k", lw=0.5, ls=":")
    ax[1].set_xlabel(r"held-out margin $m_\theta$")
    ax[1].set_ylabel("pairs")
    ax[1].legend(loc="upper right", handlelength=1)
    tag(ax[1], "(b) margin distributions")
    save(fig, "fig_beta_margins")


def fig_diagnostics():
    """What the verifier and the AI judge prefer on each controlled perturbation."""
    D = L("task5_feedback/perturbation_scores.json")
    comps = ["reasoning_sensitivity", "outcome_sensitivity", "filler_susceptibility", "distractor_robustness"]
    names = ["corrupted reasoning", "wrong final answer", "persuasive filler", "gold distractor"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.85), gridspec_kw={"wspace": 0.55, "width_ratios": [1.25, 1]})
    y, labels = [], []
    for i, (c, n) in enumerate(zip(comps, names)):
        for j, (mech, lab) in enumerate([("rlvr", "verifier"), ("rlaif", "AI judge")]):
            r = D["comparisons"][c][mech]
            yy = i * 2.6 + j
            left = 0
            for k, col in [("better_rate", "#059669"), ("tie_rate", "#D1D5DB"), ("worse_rate", "#DC2626")]:
                ax[0].barh(yy, r[k], left=left, color=col, height=0.85)
                left += r[k]
            y.append(yy); labels.append(f"{n} · {lab}")
    ax[0].set_yticks(y, labels, fontsize=6.2)
    ax[0].invert_yaxis()
    ax[0].set_xlim(0, 1)
    ax[0].set_xlabel("share of the 20 problems")
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color="#059669", label="prefers clean"), Patch(color="#D1D5DB", label="tie"), Patch(color="#DC2626", label="prefers perturbed")],
                 loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, fontsize=6.2, handlelength=0.9, columnspacing=0.8)
    tag(ax[0], "")
    ax[0].set_title("(a) clean vs. perturbed response", loc="left", fontweight="bold", pad=15)
    variants = ["clean_correct", "corrupt_reasoning_correct_final", "persuasive_filler_correct", "good_reasoning_wrong_final", "gold_distractor_wrong_final"]
    vlab = ["clean", "corrupt\nreasoning", "filler", "wrong\nfinal", "gold\ndistractor"]
    x = np.arange(len(variants))
    g = D["group_rewards"]
    ax[1].bar(x - 0.2, [g["rlvr"][v] for v in variants], 0.4, color="#2563EB", label="verifier (0/1)")
    ax[1].bar(x + 0.2, [g["rlaif"][v] for v in variants], 0.4, color="#9333EA", label="RLAIF win rate")
    ax[1].set_xticks(x, vlab, fontsize=6.0)
    ax[1].set_ylabel("mean reward")
    ax[1].set_ylim(0, 1.15)
    ax[1].legend(loc="upper right", fontsize=6.2, handlelength=0.9)
    ax[1].set_title("(b) group reward per variant", loc="left", fontweight="bold", pad=15)
    save(fig, "fig_diagnostics")


def fig_drift():
    """Every trained condition stays close to its starting point."""
    rows = [("DPO", "standard", "task1_dpo/dpo_eval_standard.json", "task1_dpo/dpo_eval_sft_reference.json"),
            ("DPO", "length-balanced", "task1_dpo/dpo_eval_length_balanced.json", "task1_dpo/dpo_eval_sft_reference.json"),
            ("DPO", r"$\beta$=0.03", "task1_dpo/dpo_eval_ablation_beta_0_03.json", "task1_dpo/dpo_eval_sft_reference.json"),
            ("DPO", r"$\beta$=0.10", "task1_dpo/dpo_eval_ablation_beta_0_10.json", "task1_dpo/dpo_eval_sft_reference.json"),
            ("DPO", r"$\beta$=0.30", "task1_dpo/dpo_eval_ablation_beta_0_30.json", "task1_dpo/dpo_eval_sft_reference.json"),
            ("PPO", "standard (20)", "task2_ppo/ppo_eval_standard.json", "task2_ppo/ppo_eval_midpoint.json"),
            ("PPO", r"$\epsilon$=0.05", "task2_ppo/ppo_eval_fork_eps0_05_kl0_10.json", "task2_ppo/ppo_eval_midpoint.json"),
            ("PPO", r"$\epsilon$=0.2 / 0.5", "task2_ppo/ppo_eval_fork_eps0_20_kl0_10.json", "task2_ppo/ppo_eval_midpoint.json"),
            ("PPO", r"$\beta_{KL}$=0", "task2_ppo/ppo_eval_fork_eps0_20_kl0_00.json", "task2_ppo/ppo_eval_midpoint.json"),
            ("PPO", r"$\beta_{KL}$=0.2", "task2_ppo/ppo_eval_fork_eps0_20_kl0_20.json", "task2_ppo/ppo_eval_midpoint.json"),
            ("GRPO", "standard (20)", "task3_grpo/grpo_eval_standard.json", "task3_grpo/grpo_eval_midpoint.json"),
            ("GRPO", "canonical", "task3_grpo/grpo_eval_fork_norm_grpo.json", "task3_grpo/grpo_eval_midpoint.json"),
            ("GRPO", "Dr. GRPO", "task3_grpo/grpo_eval_fork_norm_dr_grpo.json", "task3_grpo/grpo_eval_midpoint.json")]
    rng = np.random.default_rng(0)
    kl, d, lo, hi, labs, cols = [], [], [], [], [], []
    for meth, lab, f, base in rows:
        e, b = L(f), L(base)
        kl.append(e["kl_token_mean"] * 1e3)
        bb = {g["prompt_id"]: g["reward_score"] for g in b["generations"]}
        diff = np.array([g["reward_score"] - bb[g["prompt_id"]] for g in e["generations"] if g["prompt_id"] in bb])
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        d.append(diff.mean()); lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
        labs.append(f"{meth}: {lab}"); cols.append({"DPO": C["dpo"], "PPO": C["ppo"], "GRPO": C["grpo"]}[meth])
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.35), sharey=True, gridspec_kw={"wspace": 0.08})
    ax[0].barh(y, kl, color=cols, height=0.7)
    ax[0].axvline(0, color="k", lw=0.6)
    ax[0].set_xlabel(r"sampled KL to reference ($10^{-3}$ nats/token)")
    ax[0].set_yticks(y, labs, fontsize=6.4)
    tag(ax[0], "(a) drift on held-out prompts")
    ax[1].errorbar(d, y, xerr=[np.array(d) - np.array(lo), np.array(hi) - np.array(d)], fmt="none", lw=0.8, ecolor="#6B7280", capsize=1.5)
    for yy, c, dv in zip(y, cols, d):
        ax[1].plot(dv, yy, "o", ms=3.4, color=c)
    ax[1].axvline(0, color="k", lw=0.6, ls=":")
    ax[1].set_xlabel("paired RM change vs. start (95% CI)")
    tag(ax[1], "(b) reward change")
    save(fig, "fig_drift")


if __name__ == "__main__":
    fig_dpo_length()
    fig_rl_trajectories()
    fig_grpo()
    fig_safety()
    fig_ppo_clipping()
    fig_kl_forks()
    fig_dpo_training()
    fig_math()
    fig_dpo_logratio()
    fig_reward_cap()
    fig_beta_margins()
    fig_diagnostics()
    fig_drift()
