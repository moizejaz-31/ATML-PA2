"""Build the per-step analysis notebooks (task*/notebooks/*.ipynb).

The experiment pipeline is the Python scripts (run on Kaggle by kaggle_runner.ipynb). Each notebook
documents one step (objective, setting, implementation checks), optionally re-runs that step's
scripts (FORCE_RUN), and analyses the saved results in detail: training logs and curves, tables,
distributions, paired comparisons and qualitative examples, ending with a cell that prints the facts
relevant to the manual's research question. Interpretation is deliberately left to the report.

Build:   python -m scripts.build_task_notebooks
Execute: python -m scripts.execute_notebooks      (CPU only; reads results/, writes outputs into the .ipynb)
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS: dict[str, list[dict]] = {}


def md(s: str) -> dict:
    # Markdown cells: strip per-line indentation (the source strings are indented inside this file).
    text = "\n".join(line.strip() for line in s.strip("\n").splitlines())
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(s).strip("\n").splitlines(True)}


SETUP = code(r'''
    %matplotlib inline
    import os, sys, json, re
    from pathlib import Path
    ROOT = Path.cwd()
    while not (ROOT / "configs").exists() and ROOT != ROOT.parent:
        ROOT = ROOT.parent
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    import numpy as np, pandas as pd, matplotlib.pyplot as plt, yaml
    from IPython.display import display, Markdown, HTML, Image
    from common.nb_helpers import (style, load, jl, exists, pending, run, side_by_side, ci_text,
                                   bootstrap_mean_diff, smooth, full_windows, PALETTE, SEQ, RES, FIG)
    from common.metrics import length_stats, wilson_interval
    from common.data import load_yaml
    style()
    print("repo root:", ROOT)
''')


def run_cell(cmds: list[str], note: str = "") -> list[dict]:
    body = "\n".join(f"    run({c!r})" for c in cmds)
    return [
        md(f"""
        ## Run this step (optional)
        The results below come from the Kaggle run of the script pipeline (`kaggle_runner.ipynb`). Set `FORCE_RUN = True`
        to re-run the step's scripts from here (needs a CUDA GPU and the downloaded course assets). {note}
        """),
        code(f"FORCE_RUN = False\nif FORCE_RUN:\n{body}\nelse:\n    print('Using saved results under results/ (FORCE_RUN = False).')"),
    ]


# =============================================================================================
# TASK 1 — DPO
# =============================================================================================
DPO_HEADER_COMMON = r'''
**Objective.** For a prompt $x$ with preferred / rejected responses $(y^+, y^-)$, a trainable policy $\pi_\theta$ and the frozen reference $\pi_{\text{ref}}$,
$$\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}\Big[\log\sigma\Big(\beta\Big[\log\tfrac{\pi_\theta(y^+|x)}{\pi_{\text{ref}}(y^+|x)}-\log\tfrac{\pi_\theta(y^-|x)}{\pi_{\text{ref}}(y^-|x)}\Big]\Big)\Big],$$
with sequence log-probabilities summed over response tokens only (teacher forcing).

**Setting.** Policy = `Qwen/Qwen2.5-1.5B-Instruct` + a fresh LoRA adapter (r = 8, α = 16, dropout 0.05, `q_proj`/`v_proj`); reference = the same network with the adapter disabled. AdamW, lr 2e-5, 2 pairs × 8 accumulation steps = 16 pairs per optimizer step, gradient clipping 1.0, seed 6304, fp16 on a T4.

**Sequence-length policy (TA clarification, 3 Oct).** Some prompt + response pairs exceed the 768-token limit. For every DPO training *and* evaluation run the prompt is kept intact and an over-length response is right-truncated (no EOS is appended to a cut response); pairs whose prompt leaves fewer than 64 response tokens are dropped. The same rule is used for all conditions.
'''

T1_DEFECT = [
    md(r'''
    ## Validated objective (starter defect)
    The released `dpo_loss` computed the logit as $\beta\,[(\log\pi_\theta^+-\log\pi_\theta^-) + (\log\pi_{\text{ref}}^+-\log\pi_{\text{ref}}^-)]$, i.e. it **added** the reference margin. The objective above subtracts it, so that the logit is the difference of the two log-ratios. The corrected code and a numerical check (NumPy re-implementation) follow.
    '''),
    code(r'''
    src = (ROOT / "task1_dpo" / "dpo.py").read_text(encoding="utf-8")
    body = src[src.index("    policy_margin ="):src.index("    loss =")]
    print("task1_dpo/dpo.py (corrected lines):\n" + "\n".join(l for l in body.splitlines() if l.strip() and not l.strip().startswith("#")))

    def dpo_logit(pc, pr, rc, rr, beta, starter=False):
        return beta * ((pc - pr) + (rc - rr)) if starter else beta * ((pc - pr) - (rc - rr))
    softplus = lambda z: np.log1p(np.exp(-z))          # -log sigmoid(z)
    cases = {
        "policy == reference (untrained)": (-50.0, -60.0, -50.0, -60.0),
        "policy raises chosen vs ref":      (-45.0, -60.0, -50.0, -60.0),
        "policy raises rejected vs ref":    (-50.0, -55.0, -50.0, -60.0),
    }
    rows = []
    for name, (pc, pr, rc, rr) in cases.items():
        for starter in (True, False):
            z = dpo_logit(pc, pr, rc, rr, 0.1, starter)
            rows.append({"case": name, "implementation": "starter (+ref)" if starter else "corrected (−ref)",
                         "logit": z, "loss": softplus(z)})
    pd.DataFrame(rows).pivot(index="case", columns="implementation", values=["logit", "loss"]).round(4)
    '''),
    md("With the corrected logit an untrained policy has logit 0 and loss log 2 ≈ 0.693, and the loss moves in the right direction when the policy favours the chosen or the rejected response. The starter version already reports a non-zero logit before any training, determined by the reference model's own preference."),
]

NOTEBOOKS["task1_dpo/notebooks/1_standard_dpo.ipynb"] = [
    md(r"# Task 1 · Step 1 — Standard DPO (one epoch)" + "\n" + DPO_HEADER_COMMON + r'''
    **This notebook.** Configuration → validated objective → data and truncation policy → full training log → held-out preference metrics (289 pairs) → held-out generation metrics against the untouched SFT policy (200 prompts, identical sampling seed) → paired comparison → qualitative examples → facts for the step question.

    **Pipeline commands.** `python -m task1_dpo.preprocess`, `python -m task1_dpo.evaluate --adapter none --name sft_reference`, `python -m task1_dpo.train --run-name standard`, `python -m task1_dpo.evaluate --adapter outputs/task1_dpo/standard --name standard` (all with `--config configs/dpo.yaml`).
    '''),
    SETUP,
    md("## 1. Configuration"),
    code(r'''
    cfg = load_yaml("configs/dpo.yaml")
    keys = ["base_model", "beta", "learning_rate", "batch_size", "grad_accum_steps", "epochs", "max_sequence_length",
            "min_response_tokens", "max_generation_tokens", "eval_generation_prompts", "reward_model", "reward_max_length", "seed", "dtype"]
    conf = pd.DataFrame({"value": [str(cfg.get(k)) for k in keys]}, index=keys)
    conf.loc["lora"] = str(cfg["lora"]); conf.loc["generation (eval)"] = str(cfg["generation"])
    conf
    '''),
    *run_cell(["python -m task1_dpo.preprocess --config configs/dpo.yaml",
               "python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft_reference",
               "python -m task1_dpo.train --config configs/dpo.yaml --run-name standard",
               "python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard"]),
    *T1_DEFECT,
    md(r'''
    ## 2. Data and the 768-token policy
    Per file: pairs over the limit, pairs dropped by the prompt filter, and pairs whose chosen and/or rejected response is right-truncated. For comparison, *released rule: prompt cut* counts the pairs whose prompt the released starter would have truncated instead.
    '''),
    code(r'''
    pre = load("task1_dpo/dpo_preprocessing_report.json")
    rows = []
    for name, f in pre["files"].items():
        t = f["total"]
        rows.append({"file": name, "pairs": t.get("pairs"), "over 768 tokens": t.get("overlength_pairs", 0),
                     "dropped (prompt too long)": t.get("dropped_here", 0), "kept": t.get("kept", 0),
                     "response truncated (kept pairs)": t.get("either_truncated_here", 0),
                     "released rule: prompt cut": t.get("released_prompt_cut", 0),
                     "share chosen longer (full)": f["chosen_minus_rejected_tokens_full"]["frac_chosen_longer"],
                     "share chosen longer (after truncation)": f["chosen_minus_rejected_tokens_after_truncation"]["frac_chosen_longer"],
                     "prompt tokens p50 / p90 / p99": " / ".join(f"{f['prompt_tokens_percentiles'][q]:.0f}" for q in ["50", "90", "99"])})
    display(pd.DataFrame(rows).set_index("file"))
    print("policy:", json.dumps(pre["policy"], indent=1))
    '''),
    code(r'''display(Image(filename=str(FIG / "task1_dpo_truncation_policy.png"), width=1000))'''),
    md(r'''
    ## 3. Training log
    One row per optimizer step; every value is the mean over the step's 16 pairs (8 micro-batches of 2). The full log is `results/task1_dpo/dpo_train_standard.jsonl`.
    '''),
    code(r'''
    log = pd.DataFrame(jl("task1_dpo/dpo_train_standard.jsonl"))
    s = load("task1_dpo/dpo_summary_standard.json")
    print(f"{s['total_steps']} optimizer steps over {s['dataset_size']} kept pairs ({s['data']['dropped_overlength']} dropped of "
          f"{s['data']['rows_in_file']}); wall-clock {s['wall_time_seconds']/60:.1f} min; peak VRAM {s['peak_vram_mb']/1024:.2f} GB")
    cols = ["step", "pairs_in_step", "loss", "preference_accuracy", "implicit_chosen_reward_mean", "implicit_rejected_reward_mean",
            "reward_margin_mean", "kl_chosen", "grad_norm", "response_truncated_frac"]
    log[cols][(log.step % 10 == 0) | (log.step == 1) | (log.step == log.step.max())].set_index("step").round(4)
    '''),
    code(r'''
    log = full_windows(log)
    fig, ax = plt.subplots(2, 3, figsize=(16, 7.5))
    panels = [("loss", "DPO loss"), ("preference_accuracy", "train preference accuracy"), (None, "implicit rewards β·log π/π_ref"),
              ("reward_margin_mean", "implicit reward margin"), ("kl_chosen", "log π/π_ref on chosen (sequence sum)"), ("grad_norm", "grad norm (pre-clip)")]
    for a, (k, t) in zip(ax.flat, panels):
        if k is None:
            a.plot(log.step, smooth(log.implicit_chosen_reward_mean), color="#16A34A", label="chosen")
            a.plot(log.step, smooth(log.implicit_rejected_reward_mean), color="#DC2626", label="rejected")
            a.axhline(0, color="k", lw=0.6); a.legend()
        else:
            a.plot(log.step, log[k], color="#93C5FD", lw=1, label="per step")
            a.plot(log.step, smooth(log[k]), color="#1D4ED8", lw=2, label="5-step mean")
        if k == "loss": a.axhline(np.log(2), color="gray", ls=":", label="log 2"); a.legend()
        if k == "preference_accuracy": a.axhline(0.5, color="gray", ls=":")
        a.set_title(t); a.set_xlabel("optimizer step")
    fig.suptitle("Standard DPO — training trajectory (β = 0.10, one epoch)"); fig.tight_layout(); plt.show()
    '''),
    md(r'''
    ## 4. Held-out preference metrics
    Held-out DPO loss and preference accuracy on the filtered held-out pairs. Margin $m_\theta=[\log\pi_\theta-\log\pi_{\text{ref}}](y^+)-[\log\pi_\theta-\log\pi_{\text{ref}}](y^-)$; accuracy = share with $m_\theta>0$. For the SFT policy every margin is exactly 0 (policy = reference), so its accuracy is 0 with all pairs tied and its loss is log 2.
    '''),
    code(r'''
    sft, std = load("task1_dpo/dpo_eval_sft_reference.json"), load("task1_dpo/dpo_eval_standard.json")
    k = int(round(std["heldout_preference_accuracy"] * std["num_pairs"]))
    lo, hi = wilson_interval(k, std["num_pairs"])
    pd.DataFrame({
        "SFT (no adapter)": [sft["num_pairs"], sft["heldout_dpo_loss"], sft["heldout_preference_accuracy"], sft["heldout_tie_fraction"], sft["mean_margin"], "—"],
        "standard DPO": [std["num_pairs"], std["heldout_dpo_loss"], std["heldout_preference_accuracy"], std["heldout_tie_fraction"], std["mean_margin"], f"[{lo:.3f}, {hi:.3f}]"],
    }, index=["pairs", "held-out DPO loss (β=0.1)", "preference accuracy", "tied pairs", "mean margin (nats)", "accuracy 95% CI"])
    '''),
    code(r'''
    pp = pd.DataFrame(std["per_pair"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].hist(pp.margin, bins=40, color="#2563EB"); ax[0].axvline(0, color="k"); ax[0].set_xlabel("margin m_θ (nats)"); ax[0].set_title("held-out margins")
    ax[1].scatter(pp.logratio_rejected, pp.logratio_chosen, s=10, alpha=0.5, c=np.where(pp.margin > 0, "#16A34A", "#DC2626"))
    lim = [min(pp.logratio_rejected.min(), pp.logratio_chosen.min()), max(pp.logratio_rejected.max(), pp.logratio_chosen.max())]
    ax[1].plot(lim, lim, "k--", lw=0.8); ax[1].set_xlabel("log π/π_ref (rejected)"); ax[1].set_ylabel("log π/π_ref (chosen)")
    ax[1].set_title("per-pair log-ratios (green = correct)")
    per_tok = pd.DataFrame({"chosen": pp.logratio_chosen / pp.chosen_tokens_kept, "rejected": pp.logratio_rejected / pp.rejected_tokens_kept})
    ax[2].boxplot([per_tok.chosen, per_tok.rejected], tick_labels=["chosen", "rejected"], showfliers=False)
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set_ylabel("log π/π_ref per response token"); ax[2].set_title("per-token log-ratio on dataset responses")
    fig.tight_layout(); plt.show()
    print("mean per-token log-ratio: chosen %+.5f | rejected %+.5f" % (per_tok.chosen.mean(), per_tok.rejected.mean()))
    '''),
    md(r'''
    ## 5. Held-out generations: SFT vs standard DPO
    The first 200 kept held-out prompts, sampled with the course generation settings (T = 0.7, top-p 0.9, cap 256 tokens) and the same seed for both policies. Reward = fixed course reward model. KL = released sampled-response estimator (token mean of $\log\pi_\theta-\log\pi_{\text{ref}}$ on the policy's own generations); the per-response sum is shown as the sequence-level view.
    '''),
    code(r'''
    def gen_row(e):
        L = e["length_tokens"]
        return {"RM score": f"{e['mean_reward']:.3f} ± {e['sem_reward']:.3f}", "RM std": e["std_reward"],
                "KL token-mean": e["kl_token_mean"], "KL sequence-mean": e["kl_sequence_mean"],
                "entropy (exact)": e["entropy_exact"], "length mean ± std": f"{L['mean']:.1f} ± {L['std']:.1f}",
                "length median [IQR]": f"{L['median']:.0f} [{L['q25']:.0f}, {L['q75']:.0f}]", "EOS rate": e["eos_rate"],
                "hit 256 cap": e["hit_max_tokens_rate"], "corr(RM, length)": e["reward_length_corr"]}
    pd.DataFrame({"SFT (no adapter)": gen_row(sft), "standard DPO": gen_row(std)})
    '''),
    code(r'''
    G = {n: pd.DataFrame(e["generations"]) for n, e in [("SFT", sft), ("DPO", std)]}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    bins = np.linspace(min(g.reward_score.min() for g in G.values()), max(g.reward_score.max() for g in G.values()), 30)
    for n, c in [("SFT", "#6B7280"), ("DPO", "#2563EB")]:
        ax[0].hist(G[n].reward_score, bins=bins, histtype="step", lw=2, color=c, label=n)
        ax[1].hist(G[n].response_length_tokens, bins=30, histtype="step", lw=2, color=c, label=n)
    ax[0].set_xlabel("RM score"); ax[0].legend(); ax[0].set_title("reward distribution")
    ax[1].set_xlabel("generated tokens (cap 256)"); ax[1].legend(); ax[1].set_title("length distribution")
    g = G["DPO"]
    ax[2].scatter(g.response_length_tokens, g.reward_score, s=12, alpha=0.6, c=np.where(g.terminated_with_eos, "#2563EB", "#DC2626"))
    ax[2].set_xlabel("generated tokens"); ax[2].set_ylabel("RM score"); ax[2].set_title("DPO: RM vs length (red = no EOS before cap)")
    fig.tight_layout(); plt.show()
    for n in G:
        fin = G[n][G[n].terminated_with_eos]; cap = G[n][~G[n].terminated_with_eos]
        print(f"{n}: RM finished {fin.reward_score.mean():.3f} (n={len(fin)}) | RM unfinished {cap.reward_score.mean():.3f} (n={len(cap)})")
    '''),
    code(r'''
    m = G["SFT"].merge(G["DPO"], on="prompt_id", suffixes=("_sft", "_dpo"))
    d, lo, hi = bootstrap_mean_diff(m.reward_score_dpo, m.reward_score_sft)
    dl, llo, lhi = bootstrap_mean_diff(m.response_length_tokens_dpo, m.response_length_tokens_sft)
    pd.Series({
        "paired prompts": len(m),
        "mean RM difference DPO − SFT [95% bootstrap CI]": f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]",
        "share of prompts with higher RM under DPO": f"{(m.reward_score_dpo > m.reward_score_sft).mean():.3f}",
        "mean length difference (tokens) [95% CI]": f"{dl:+.1f} [{llo:+.1f}, {lhi:+.1f}]",
        "identical generated text (same seed)": f"{(m.response_sft == m.response_dpo).mean():.3f}",
    }, name="paired comparison").to_frame()
    '''),
    md(r'''
    ## 6. Qualitative examples
    Mechanically selected cases (largest RM gain combined with length growth; largest RM drop; highest-RM response that never terminated). Whether a higher RM score corresponds to a better response is for the reader to judge from the text.
    '''),
    code(r'''
    m["gain"] = m.reward_score_dpo - m.reward_score_sft
    m["grow"] = m.response_length_tokens_dpo / m.response_length_tokens_sft.clip(lower=1)
    diff = m[m.response_sft != m.response_dpo].copy()
    diff["gain_x_growth"] = diff.gain * diff.grow
    picks = [("RM up while the response grew", diff.sort_values("gain_x_growth").iloc[-1]),
             ("RM up, similar length", diff[(diff.grow > 0.8) & (diff.grow < 1.25)].sort_values("gain").iloc[-1]),
             ("largest RM drop", diff.sort_values("gain").iloc[0])]
    for title, r in picks:
        side_by_side(title, r.prompt_sft, {"SFT": r.response_sft, "standard DPO": r.response_dpo},
                     meta={"SFT": f"RM {r.reward_score_sft:+.2f} · {r.response_length_tokens_sft} tok · EOS {r.terminated_with_eos_sft}",
                           "standard DPO": f"RM {r.reward_score_dpo:+.2f} · {r.response_length_tokens_dpo} tok · EOS {r.terminated_with_eos_dpo}"})
    '''),
    md(r'''
    ## 7. Evidence for the step question
    *Is stronger preference fitting accompanied by useful behaviour, or by disproportionate policy drift or a trivial length shift?* The cell prints the measured quantities; interpretation belongs in the report.
    '''),
    code(r'''
    acc_lo, acc_hi = wilson_interval(k, std["num_pairs"])
    print(f"held-out preference accuracy {std['heldout_preference_accuracy']:.3f} (95% CI {acc_lo:.3f}-{acc_hi:.3f}); "
          f"held-out DPO loss {std['heldout_dpo_loss']:.4f} vs log 2 = {np.log(2):.4f}")
    print(f"sampled KL to reference: {std['kl_token_mean']:.5f} nats/token ({std['kl_sequence_mean']:.3f} per response)")
    print(f"RM: SFT {sft['mean_reward']:.3f} → DPO {std['mean_reward']:.3f}; paired difference {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"length: SFT {sft['length_tokens']['mean']:.1f} → DPO {std['length_tokens']['mean']:.1f} tokens; paired difference {dl:+.1f} [{llo:+.1f}, {lhi:+.1f}]")
    print(f"responses hitting the 256-token cap: SFT {sft['hit_max_tokens_rate']:.2f}, DPO {std['hit_max_tokens_rate']:.2f}")
    print(f"identical generations under the same seed: {(m.response_sft == m.response_dpo).mean():.2f}")
    '''),
]

NOTEBOOKS["task1_dpo/notebooks/2_beta_ablation.ipynb"] = [
    md(r"# Task 1 · Step 2 — Regularisation strength (β ∈ {0.03, 0.10, 0.30})" + "\n" + DPO_HEADER_COMMON + r'''
    **Protocol.** Three short forks from the same initialisation, trained on the same first 600 kept pairs of the standard split (38 optimizer steps), with identical data order, optimizer, seed and LoRA configuration — only β changes. All forks are evaluated with the same held-out pairs and the same generation protocol and seed. The standard one-epoch run (Step 1) uses a larger budget and is shown only as a labelled reference.

    **Research question.** How does changing β alter preference fitting, KL from the reference policy and reward-model score? Is the relationship monotonic over the tested range?
    '''),
    SETUP,
    *run_cell(["python -m task1_dpo.ablate_beta --config configs/dpo.yaml --skip-train"], "`--skip-train` reuses fork adapters that already exist."),
    md("## 1. Training trajectories of the three forks"),
    code(r'''
    BETAS = [0.03, 0.10, 0.30]
    name = lambda b: f"beta_{b:.2f}".replace(".", "_")
    logs = {b: full_windows(pd.DataFrame(jl(f"task1_dpo/dpo_train_{name(b)}.jsonl"))) for b in BETAS}
    summ = {b: load(f"task1_dpo/dpo_summary_{name(b)}.json") for b in BETAS}
    display(pd.DataFrame({f"β={b:g}": {"optimizer steps": summ[b]["total_steps"], "pairs": summ[b]["dataset_size"],
                                       "wall-clock (min)": round(summ[b]["wall_time_seconds"] / 60, 1),
                                       "final-5-step loss": round(logs[b].loss.tail(5).mean(), 4),
                                       "final-5-step train acc": round(logs[b].preference_accuracy.tail(5).mean(), 3)} for b in BETAS}))
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    for b, c in zip(BETAS, SEQ):
        L = logs[b]
        ax[0].plot(L.step, smooth(L.loss), color=c, label=f"β={b:g}")
        ax[1].plot(L.step, smooth(L.reward_margin_mean / L.beta), color=c, label=f"β={b:g}")
        ax[2].plot(L.step, smooth(L.implicit_chosen_reward_mean / L.beta), color=c, ls="-", label=f"β={b:g} chosen")
        ax[2].plot(L.step, smooth(L.implicit_rejected_reward_mean / L.beta), color=c, ls="--", label=f"β={b:g} rejected")
        ax[3].plot(L.step, smooth(L.grad_norm), color=c, label=f"β={b:g}")
    for a, t in zip(ax, ["train DPO loss (5-step mean)", "log-ratio margin (unscaled)", "log π/π_ref (unscaled)", "grad norm (pre-clip)"]):
        a.set_title(t); a.set_xlabel("optimizer step")
    ax[0].axhline(np.log(2), color="gray", ls=":"); ax[0].legend(); ax[2].legend(fontsize=7)
    fig.tight_layout(); plt.show()
    '''),
    md(r'''
    ## 2. Held-out results
    Held-out DPO loss is computed with each fork's own β, so its scale differs across rows; preference accuracy, margins, KL, RM and length are directly comparable.
    '''),
    code(r'''
    E = {f"β={b:g} (600 pairs)": load(f"task1_dpo/dpo_eval_ablation_{name(b)}.json") for b in BETAS}
    E_ref = {"SFT (no adapter)": load("task1_dpo/dpo_eval_sft_reference.json"), "standard β=0.10 (1 epoch)": load("task1_dpo/dpo_eval_standard.json")}
    def row(e, sft=False):
        L = e["length_tokens"]
        return {"held-out loss (own β)": None if sft else e["heldout_dpo_loss"], "pref. accuracy": None if sft else e["heldout_preference_accuracy"],
                "mean margin": None if sft else e["mean_margin"], "KL token": e["kl_token_mean"], "KL seq": e["kl_sequence_mean"],
                "RM": e["mean_reward"], "RM s.e.m.": e["sem_reward"], "entropy": e["entropy_exact"],
                "length mean": L["mean"], "length IQR": L["iqr"], "EOS rate": e["eos_rate"]}
    tab = pd.DataFrame({**{k: row(v) for k, v in E.items()}, **{k: row(v, k.startswith("SFT")) for k, v in E_ref.items()}}).T
    tab.style.format(precision=4, na_rep="—")
    '''),
    code(r'''
    xs = np.arange(len(BETAS)); keys = list(E)
    fig, ax = plt.subplots(1, 5, figsize=(22, 4))
    for a, (k, t) in zip(ax, [("heldout_preference_accuracy", "held-out preference accuracy"), ("mean_margin", "mean held-out margin (nats)"),
                               ("kl_token_mean", "sampled KL (token mean)"), ("mean_reward", "RM score ± s.e.m."), (None, "generated length (mean, IQR)")]):
        if k is None:
            y = [E[q]["length_tokens"]["mean"] for q in keys]; lo = [E[q]["length_tokens"]["q25"] for q in keys]; hi = [E[q]["length_tokens"]["q75"] for q in keys]
            a.errorbar(xs, y, yerr=[np.array(y) - lo, np.array(hi) - y], fmt="o-", color="#2563EB", capsize=4)
            a.axhline(E_ref["SFT (no adapter)"]["length_tokens"]["mean"], color="gray", ls=":", label="SFT")
        else:
            y = [E[q][k] for q in keys]
            yerr = [E[q]["sem_reward"] for q in keys] if k == "mean_reward" else None
            a.errorbar(xs, y, yerr=yerr, fmt="o-", color="#2563EB", capsize=4, label="short forks")
            if k in ("kl_token_mean", "mean_reward"):
                a.axhline(E_ref["SFT (no adapter)"][k], color="gray", ls=":", label="SFT")
            a.axhline(E_ref["standard β=0.10 (1 epoch)"][k], color="#DC2626", ls="--", label="standard (1 epoch)")
        a.set_xticks(xs); a.set_xticklabels([f"β={b:g}" for b in BETAS]); a.set_title(t)
    ax[0].legend(); ax[3].legend(); fig.tight_layout(); plt.show()
    '''),
    md(r'''
    ## 3. How different are the three fork models?
    Correlation and sign agreement of the per-pair held-out margins between forks, and paired RM differences to SFT on the same prompts (bootstrap 95% CIs).
    '''),
    code(r'''
    M = {b: np.array([p["margin"] for p in E[f"β={b:g} (600 pairs)"]["per_pair"]]) for b in BETAS}
    rows = []
    for a_, b_ in [(0.03, 0.10), (0.10, 0.30), (0.03, 0.30)]:
        rows.append({"pair": f"β={a_:g} vs β={b_:g}", "corr(margins)": np.corrcoef(M[a_], M[b_])[0, 1],
                     "sign agreement": np.mean((M[a_] > 0) == (M[b_] > 0)), "mean |margin| ratio": np.abs(M[b_]).mean() / np.abs(M[a_]).mean()})
    display(pd.DataFrame(rows).set_index("pair").round(4))
    sft_g = pd.DataFrame(E_ref["SFT (no adapter)"]["generations"]).set_index("prompt_id")
    rows = []
    for k, e in {**E, "standard β=0.10 (1 epoch)": E_ref["standard β=0.10 (1 epoch)"]}.items():
        g = pd.DataFrame(e["generations"]).set_index("prompt_id").loc[sft_g.index]
        d, lo, hi = bootstrap_mean_diff(g.reward_score, sft_g.reward_score)
        rows.append({"condition": k, "RM − SFT": d, "95% CI low": lo, "95% CI high": hi,
                     "identical text to SFT": (g.response == sft_g.response).mean()})
    pd.DataFrame(rows).set_index("condition").round(4)
    '''),
    md("## 4. Evidence for RQ1 (facts only)"),
    code(r'''
    for q in ["heldout_preference_accuracy", "mean_margin", "kl_token_mean", "mean_reward"]:
        vals = [E[f"β={b:g} (600 pairs)"][q] for b in BETAS]
        trend = "increasing" if np.all(np.diff(vals) > 0) else "decreasing" if np.all(np.diff(vals) < 0) else "non-monotonic"
        print(f"{q:30s} " + "  ".join(f"β={b:g}: {v:+.4f}" for b, v in zip(BETAS, vals)) + f"   -> {trend} over β")
    print("RM s.e.m. per fork:", ", ".join(f"{E[k]['sem_reward']:.3f}" for k in E))
    '''),
]

NOTEBOOKS["task1_dpo/notebooks/3_length_confounding.ipynb"] = [
    md(r"# Task 1 · Step 3 — Length confounding" + "\n" + DPO_HEADER_COMMON + r'''
    **Protocol.** One additional DPO model is trained from the original initialisation on the supplied length-balanced subset (500 pairs from each stratum: preferred response clearly longer / length matched / rejected response clearly longer; strata defined by the course). Standard and length-balanced DPO are evaluated on the supplied length-stratified held-out set (82 pairs per stratum before the overlength filter), and all three policies (SFT, standard, length-balanced) answer the ten common word-limit prompts (greedy decoding, plus 8 sampled answers per prompt).

    **Research questions.** (RQ2) How much of the observed length behaviour can be explained by the preference data itself; does balancing the strata change which examples are learned well or how long the model responds? (RQ3) Cases where the aligned policy has a stronger preference/reward signal but a worse response.
    '''),
    SETUP,
    *run_cell(["python -m task1_dpo.analyze_length --config configs/dpo.yaml --skip-train"]),
    md("## 1. Length structure of the two training sets (a property of the data)"),
    code(r'''
    pre = load("task1_dpo/dpo_preprocessing_report.json")["files"]
    rows = []
    for key, lab in [("dpo_standard_train", "standard train"), ("dpo_length_train", "length-balanced train"), ("dpo_length_eval", "length-stratified eval")]:
        f = pre[key]
        rows.append({"set": lab, "pairs": f["total"]["pairs"], "kept": f["total"].get("kept"),
                     "mean tokens(chosen) − tokens(rejected)": f["chosen_minus_rejected_tokens_full"]["mean"],
                     "share chosen longer": f["chosen_minus_rejected_tokens_full"]["frac_chosen_longer"]})
    display(pd.DataFrame(rows).set_index("set").round(3))
    strat = []
    for key in ["dpo_length_train", "dpo_length_eval"]:
        for s, st in pre[key]["per_stratum"].items():
            strat.append({"set": key, "stratum": s, "pairs": st["pairs"], "dropped": st.get("dropped_here", 0),
                          "kept": st.get("kept", 0), "response truncated": st.get("either_truncated_here", 0)})
    pd.DataFrame(strat).set_index(["set", "stratum"])
    '''),
    md("## 2. Training: standard vs length-balanced"),
    code(r'''
    Ls = {"standard": full_windows(pd.DataFrame(jl("task1_dpo/dpo_train_standard.jsonl"))),
          "length-balanced": full_windows(pd.DataFrame(jl("task1_dpo/dpo_train_length_balanced.jsonl")))}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for (n, L), c in zip(Ls.items(), ["#2563EB", "#16A34A"]):
        ax[0].plot(L.step, smooth(L.loss), color=c, label=n)
        ax[1].plot(L.step, smooth(L.preference_accuracy), color=c, label=n)
        ax[2].plot(L.step, smooth(L.implicit_chosen_reward_mean), color=c, label=f"{n} chosen")
        ax[2].plot(L.step, smooth(L.implicit_rejected_reward_mean), color=c, ls="--", label=f"{n} rejected")
    for a, t in zip(ax, ["train loss (5-step mean)", "train preference accuracy", "implicit rewards"]):
        a.set_title(t); a.set_xlabel("optimizer step"); a.legend(fontsize=8)
    fig.tight_layout(); plt.show()
    '''),
    md(r'''
    ## 3. Held-out accuracy by length stratum
    Strata are the course labels. 95% Wilson intervals. Margins in nats (unscaled log-ratio difference).
    '''),
    code(r'''
    LA = load("task1_dpo/dpo_length_analysis.json")
    STRATA = ["preferred_longer", "length_matched", "rejected_longer"]
    rows = []
    for s in STRATA:
        for mdl in ["standard", "length_balanced"]:
            st = LA[mdl]["stratified"][s]
            rows.append({"stratum": s, "model": mdl, "pairs": st["total_pairs"], "accuracy": ci_text(st["accuracy"], st["accuracy_ci95"]),
                         "mean margin": st["mean_margin"], "median margin": st["median_margin"], "mean loss": st["mean_dpo_loss"],
                         "pairs w/ truncated response": st["pairs_with_truncated_response"],
                         "mean tokens chosen / rejected": f"{st['mean_chosen_tokens']:.0f} / {st['mean_rejected_tokens']:.0f}"})
    pd.DataFrame(rows).set_index(["stratum", "model"]).round(3)
    '''),
    code(r'''
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.4))
    for j, (mdl, c) in enumerate([("standard", "#2563EB"), ("length_balanced", "#16A34A")]):
        st = LA[mdl]["stratified"]
        acc = np.array([st[s]["accuracy"] for s in STRATA]); ci = np.array([st[s]["accuracy_ci95"] for s in STRATA])
        ax[0].bar(np.arange(3) + (j - 0.5) * 0.38, acc, 0.38, yerr=[acc - ci[:, 0], ci[:, 1] - acc], capsize=4, color=c, label=mdl)
        pp = pd.DataFrame(LA[mdl]["per_pair"]); pp["d"] = pp.chosen_tokens_full - pp.rejected_tokens_full
        bins = pd.qcut(pp.d, 6, duplicates="drop")
        g = pp.groupby(bins, observed=True).agg(d=("d", "mean"), acc=("margin", lambda m: (m > 0).mean()), margin=("margin", "mean"))
        ax[1].plot(g.d, g.acc, "o-", color=c, label=mdl); ax[2].scatter(pp.d, pp.margin, s=8, alpha=0.35, color=c)
        ax[2].plot(g.d, g.margin, "o-", color=c, lw=2, label=f"{mdl} (sextile means)")
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(STRATA); ax[0].axhline(0.5, color="k", ls=":"); ax[0].set_ylim(0, 1)
    ax[0].set_title("accuracy by stratum (95% CI)"); ax[0].legend()
    ax[1].axhline(0.5, color="k", ls=":"); ax[1].axvline(0, color="k", lw=0.6); ax[1].set_xlabel("tokens(chosen) − tokens(rejected), sextile mean")
    ax[1].set_ylabel("accuracy"); ax[1].set_title("accuracy vs actual length difference"); ax[1].legend()
    ax[2].axhline(0, color="k", lw=0.6); ax[2].axvline(0, color="k", lw=0.6); ax[2].set_xlabel("tokens(chosen) − tokens(rejected)")
    ax[2].set_ylabel("margin (nats)"); ax[2].set_title("held-out margin vs length difference"); ax[2].legend(fontsize=8)
    fig.tight_layout(); plt.show()
    for mdl in ["standard", "length_balanced"]:
        pp = pd.DataFrame(LA[mdl]["per_pair"]); longer = pp.chosen_tokens_full > pp.rejected_tokens_full
        print(f"{mdl:16s} accuracy when chosen is longer {np.mean(pp.margin[longer] > 0):.3f} (n={longer.sum()}), "
              f"when chosen is shorter {np.mean(pp.margin[pp.chosen_tokens_full < pp.rejected_tokens_full] > 0):.3f} "
              f"(n={(pp.chosen_tokens_full < pp.rejected_tokens_full).sum()}); corr(log-ratio, length): "
              f"chosen {np.corrcoef(pp.logratio_chosen, pp.chosen_tokens_kept)[0,1]:+.2f}, rejected {np.corrcoef(pp.logratio_rejected, pp.rejected_tokens_kept)[0,1]:+.2f}")
    '''),
    md("## 4. Generated length on the held-out prompts (a property of the policy)"),
    code(r'''
    EV = {"SFT": load("task1_dpo/dpo_eval_sft_reference.json"), "standard DPO": load("task1_dpo/dpo_eval_standard.json"),
          "length-balanced DPO": load("task1_dpo/dpo_eval_length_balanced.json")}
    display(pd.DataFrame({k: {"length mean": e["length_tokens"]["mean"], "std": e["length_tokens"]["std"], "median": e["length_tokens"]["median"],
                              "IQR": e["length_tokens"]["iqr"], "EOS rate": e["eos_rate"], "RM": e["mean_reward"], "KL token": e["kl_token_mean"]}
                          for k, e in EV.items()}).T.round(4))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for (k, e), c in zip(EV.items(), ["#6B7280", "#2563EB", "#16A34A"]):
        ax.hist([g["response_length_tokens"] for g in e["generations"]], bins=26, histtype="step", lw=2, color=c, label=k)
    ax.set_xlabel("generated tokens (cap 256)"); ax.legend(); plt.show()
    '''),
    md("## 5. Word-limit compliance on the common prompt set"),
    code(r'''
    P = {"SFT": "sft", "standard DPO": "standard", "length-balanced DPO": "length_balanced"}
    summ = {k: {"greedy compliance": LA[v]["word_limit"]["greedy_compliance_rate"],
                "greedy 95% CI": "[{:.2f}, {:.2f}]".format(*LA[v]["word_limit"]["greedy_compliance_ci95"]),
                "sampled compliance (8/prompt)": LA[v]["word_limit"]["sampled_compliance_rate"],
                "sampled 95% CI": "[{:.2f}, {:.2f}]".format(*LA[v]["word_limit"]["sampled_compliance_ci95"]),
                "greedy words mean": LA[v]["word_limit"]["greedy_word_count"]["mean"],
                "words / limit": LA[v]["word_limit"]["mean_excess_ratio"]} for k, v in P.items()}
    display(pd.DataFrame(summ).T.round(3))
    det = {k: LA[v]["word_limit"]["details"] for k, v in P.items()}
    per = pd.DataFrame([{"prompt": d["prompt"], "limit": d["limit"], **{k: f"{det[k][i]['word_count']} {'✓' if det[k][i]['compliant'] else '✗'}" for k in det}}
                        for i, d in enumerate(det["SFT"])])
    per
    '''),
    code(r'''
    for i in [0, 3, 6]:
        d0 = det["SFT"][i]
        side_by_side(f"Word-limit prompt (limit {d0['limit']} words)", d0["prompt"], {k: det[k][i]["response"] for k in det},
                     meta={k: f"{det[k][i]['word_count']} words · {'compliant' if det[k][i]['compliant'] else 'over the limit'}" for k in det})
    '''),
    md("## 6. Qualitative: stronger preference signal, but is the response better?  (RQ3 candidates)"),
    code(r'''
    sft_g = pd.DataFrame(EV["SFT"]["generations"]); std_g = pd.DataFrame(EV["standard DPO"]["generations"])
    mm = sft_g.merge(std_g, on="prompt_id", suffixes=("_sft", "_dpo")); mm = mm[mm.response_sft != mm.response_dpo]
    mm["gain"] = mm.reward_score_dpo - mm.reward_score_sft
    for _, r in mm.sort_values("gain", ascending=False).head(2).iterrows():
        side_by_side("Higher RM under DPO", r.prompt_sft, {"SFT": r.response_sft, "standard DPO": r.response_dpo},
                     meta={"SFT": f"RM {r.reward_score_sft:+.2f} · {r.response_length_tokens_sft} tok", "standard DPO": f"RM {r.reward_score_dpo:+.2f} · {r.response_length_tokens_dpo} tok"})
    '''),
    md("## 7. Evidence for RQ2 / RQ3 (facts only)"),
    code(r'''
    for mdl in ["standard", "length_balanced"]:
        st = LA[mdl]["stratified"]
        print(f"{mdl:16s} " + " | ".join(f"{s}: acc {st[s]['accuracy']:.3f}, margin {st[s]['mean_margin']:+.2f}" for s in STRATA))
    print("share chosen longer: standard train %.2f, length-balanced train %.2f" % (pre["dpo_standard_train"]["chosen_minus_rejected_tokens_full"]["frac_chosen_longer"], pre["dpo_length_train"]["chosen_minus_rejected_tokens_full"]["frac_chosen_longer"]))
    print("held-out generated length: " + ", ".join(f"{k} {e['length_tokens']['mean']:.1f}" for k, e in EV.items()))
    print("word-limit greedy compliance: " + ", ".join(f"{k} {v['greedy compliance']:.2f}" for k, v in summ.items()))
    '''),
]


# =============================================================================================
# TASK 2 — PPO
# =============================================================================================
PPO_HEADER = r'''
**Objective.** With token ratio $\rho_t(\theta)=\pi_\theta(a_t|s_t)/\pi_{\text{old}}(a_t|s_t)$ and advantage $A_t$, PPO maximises the clipped surrogate
$$L^{\text{clip}}(\theta)=\mathbb{E}_t\big[\min\big(\rho_tA_t,\ \text{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\big)\big].$$
Rewards are the terminal reward-model score plus sampled-token KL shaping, $r_t=r_{\text{task}}\mathbb{1}[t=T]-\beta_{\text{KL}}(\log\pi_\theta-\log\pi_{\text{ref}})$, and advantages use GAE, $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$, $A_t=\sum_k(\gamma\lambda)^k\delta_{t+k}$ (γ = 1, λ = 0.95), normalised over the response tokens of the update.

**Setting.** Continuation from the supplied PPO midpoint (policy LoRA + matched value model; reference = base policy, adapter disabled), fixed course reward model, 1 prompt per update (training-pool indices 0, 1, 2, … for every run), 2 PPO epochs per rollout, policy lr 3e-6, critic LoRA lr 1e-4 / head lr 3e-4, value coefficient 0.5, missing-EOS penalty 1.0, generation cap 512 tokens during training and 768 for frozen evaluation (64 fixed held-out prompts, T = 0.7, top-p 0.9, same seed for every condition).

**Implementation notes.** LoRA dropout is disabled during the RL updates so that $\rho_t$ measures policy change rather than dropout noise. The supplied critic loads in fp16 on a T4; its trainable tensors are kept in fp32 (forward pass under fp16 autocast), because AdamW on fp16 weights produces inf/NaN on the first step. Updates with a non-finite loss or gradient are skipped and counted (none occurred).
'''

PPO_DEFECT = [
    md(r'''
    ## Validated objective (starter defect)
    The released `ppo_policy_loss` took `torch.maximum(surr1, surr2)`, an optimistic bound that lets a single batch push the ratio arbitrarily far in the advantage direction. The clipped surrogate takes the **minimum** (pessimistic bound). Corrected code and a numerical check:
    '''),
    code(r'''
    src = (ROOT / "task2_ppo" / "ppo.py").read_text(encoding="utf-8")
    print("\n".join(l for l in src.splitlines() if "surr" in l or "objective =" in l))
    rows = []
    for A in (+1.0, -1.0):
        for rho in (0.5, 0.9, 1.0, 1.1, 1.5):
            s1, s2 = rho * A, np.clip(rho, 0.8, 1.2) * A
            rows.append({"A": A, "ratio ρ": rho, "min (corrected)": min(s1, s2), "max (starter)": max(s1, s2),
                         "gradient flows (corrected)": "yes" if min(s1, s2) == s1 else "no (clipped)"})
    pd.DataFrame(rows).set_index(["A", "ratio ρ"])
    '''),
    md("With ε = 0.2 the corrected objective stops rewarding ratios beyond 1.2 when A > 0 and below 0.8 when A < 0. The starter's maximum instead keeps the unclipped (larger) term exactly in those regions, so clipping never limits the update."),
]


NOTEBOOKS["task2_ppo/notebooks/1_standard_ppo.ipynb"] = [
    md(r"# Task 2 · Step 1 — Standard PPO continuation (20 updates)" + "\n" + PPO_HEADER + r'''
    **This notebook.** Configuration → validated objective → full continuation log → trajectories → critic behaviour → rollouts → held-out evaluation of SFT, midpoint and the continued policy → qualitative examples → compute → facts.
    '''),
    SETUP,
    md("## 1. Configuration"),
    code(r'''
    cfg = load_yaml("configs/ppo.yaml")
    keys = ["updates", "prompts_per_update", "ppo_epochs", "policy_learning_rate", "value_lora_learning_rate", "value_head_learning_rate",
            "value_train_mode", "clip_epsilon", "kl_beta", "gamma", "gae_lambda", "value_coef", "missing_eos_penalty", "max_prompt_length",
            "max_response_length", "eval_max_response_length", "eval_num_prompts", "reward_max_length", "max_grad_norm", "disable_dropout", "seed"]
    pd.DataFrame({"value": [str(cfg.get(k)) for k in keys]}, index=keys)
    '''),
    *run_cell(["python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint",
               "python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter none --name sft_reference",
               "python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard",
               "python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard"]),
    *PPO_DEFECT,
    md(r'''
    ## 2. Continuation log
    One row per update (one prompt, one sampled response, two PPO epochs). *clip (ep 2)* is the clip fraction measured at the second epoch, i.e. after one gradient step on the rollout (at epoch 1 the ratio is exactly 1). *KL(old‖new)* is the k3 estimate of the step size of that first update.
    '''),
    code(r'''
    L = pd.DataFrame(jl("task2_ppo/ppo_train_standard.jsonl"))
    S = load("task2_ppo/ppo_summary_standard.json")
    view = L[["update", "reward_rm", "reward_effective", "kl_token_mean", "policy_loss", "value_loss", "clip_fraction_last_epoch",
              "approx_kl_old_new_last_epoch", "entropy_exact", "policy_grad_norm", "value_grad_norm", "response_length",
              "terminated_with_eos", "value_explained_variance"]].rename(columns={"clip_fraction_last_epoch": "clip (ep 2)",
              "approx_kl_old_new_last_epoch": "KL(old‖new)", "value_explained_variance": "critic EV", "terminated_with_eos": "EOS"})
    print(f"updates {S['updates']} | skipped non-finite steps {S['skipped_nonfinite_steps']} | generated tokens {S['generated_tokens']} | "
          f"wall-clock {S['wall_time_seconds']/60:.1f} min | peak VRAM {S['peak_vram_mb']/1024:.2f} GB")
    view.set_index("update").round(4)
    '''),
    code(r'''
    fig, ax = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    panels = [(["reward_rm", "reward_effective"], "reward (RM / after EOS penalty)"), (["kl_token_mean"], "sampled KL to reference (token mean)"),
              (["policy_loss"], "policy loss"), (["value_loss"], "value loss"), (["entropy_exact", "entropy_sampled"], "entropy (exact / sampled −log π)"),
              (["clip_fraction_last_epoch"], "clip fraction (epoch 2)"), (["policy_grad_norm", "value_grad_norm"], "grad norm (pre-clip)"),
              (["response_length"], "response length (tokens)"), (["value_explained_variance"], "critic explained variance")]
    for a, (ks, t) in zip(ax.flat, panels):
        for k, c in zip(ks, ["#2563EB", "#DC2626"]):
            a.plot(L["update"], L[k], "o-", ms=3, color=c, label=k)
        if len(ks) > 1: a.legend(fontsize=7)
        a.set_title(t)
    for a in ax[-1]: a.set_xlabel("update")
    fig.suptitle("Standard PPO continuation from the supplied midpoint"); fig.tight_layout(); plt.show()
    '''),
    md(r'''
    ## 3. Critic behaviour
    The supplied critic is documented as weak (held-out explained variance −3.7 at release). Here: value predictions vs GAE returns per update, and whether the value at the first response token predicts the realised reward.
    '''),
    code(r'''
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(L["update"], L.value_mean, "o-", label="mean value V(s_t)"); ax[0].plot(L["update"], L.return_mean, "s-", label="mean return"); ax[0].legend(); ax[0].set_title("values vs returns (response tokens)")
    ax[1].plot(L["update"], L.value_explained_variance, "o-", color="#DC2626"); ax[1].axhline(0, color="k", lw=0.6); ax[1].set_title("explained variance of returns")
    ax[2].scatter(L.value_first_token, L.reward_effective, c=L["update"], cmap="viridis"); ax[2].set_xlabel("value at first response token"); ax[2].set_ylabel("realised reward")
    ax[2].set_title(f"corr = {np.corrcoef(L.value_first_token, L.reward_effective)[0,1]:+.2f}")
    for a in ax[:2]: a.set_xlabel("update")
    fig.tight_layout(); plt.show()
    print(f"median critic EV {L.value_explained_variance.median():.2f}; value loss first/last {L.value_loss.iloc[0]:.3f} / {L.value_loss.iloc[-1]:.3f}")
    '''),
    md("## 4. Training rollouts (one per update)"),
    code(r'''
    R = pd.DataFrame(jl("task2_ppo/ppo_rollouts_standard.jsonl"))
    R["prompt (start)"] = R.prompt.str.slice(0, 90); R["response (start)"] = R.response.str.slice(0, 160)
    R[["update", "prompt (start)", "response (start)", "reward_rm", "response_length", "terminated_with_eos"]].set_index("update").round(3)
    '''),
    md("## 5. Held-out evaluation: SFT vs midpoint vs continued policy"),
    code(r'''
    EV = {"SFT (no adapter)": load("task2_ppo/ppo_eval_sft_reference.json"), "midpoint (start)": load("task2_ppo/ppo_eval_midpoint.json"),
          "standard PPO (20 updates)": load("task2_ppo/ppo_eval_standard.json")}
    def row(e):
        Lt = e["length_tokens"]
        return {"RM": f"{e['mean_reward']:.3f} ± {e['sem_reward']:.3f}", "KL token": e["kl_token_mean"], "KL seq": e["kl_sequence_mean"],
                "entropy": e["entropy_exact"], "length mean ± std": f"{Lt['mean']:.1f} ± {Lt['std']:.1f}", "length median": Lt["median"],
                "EOS rate": e["eos_rate"], "hit 768 cap": e["hit_max_tokens_rate"], "prompts left-truncated": e["prompt_truncated_count"]}
    pd.DataFrame({k: row(v) for k, v in EV.items()}).T
    '''),
    code(r'''
    G = {k: pd.DataFrame(v["generations"]).set_index("prompt_id") for k, v in EV.items()}
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    for (k, g), c in zip(G.items(), ["#6B7280", "#111827", "#DC2626"]):
        ax[0].hist(g.reward_score, bins=20, histtype="step", lw=2, color=c, label=k)
        ax[1].hist(g.response_length_tokens, bins=20, histtype="step", lw=2, color=c, label=k)
    ax[0].set_xlabel("RM score"); ax[1].set_xlabel("generated tokens (cap 768)"); ax[0].legend(fontsize=8); fig.tight_layout(); plt.show()
    a, b = G["standard PPO (20 updates)"], G["midpoint (start)"].loc[G["standard PPO (20 updates)"].index]
    d, lo, hi = bootstrap_mean_diff(a.reward_score, b.reward_score)
    dl, llo, lhi = bootstrap_mean_diff(a.response_length_tokens, b.response_length_tokens)
    print(f"paired RM change midpoint → PPO-20: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]; length change {dl:+.1f} [{llo:+.1f}, {lhi:+.1f}] tokens; "
          f"identical text {(a.response == b.response).mean():.2f}")
    '''),
    md(r'''
    ## 6. Qualitative examples
    Mechanically selected from the 64 held-out prompts: the largest RM increase, an increase that came with a much longer response, and the largest decrease (midpoint → PPO-20). Whether reward and quality move together is to be judged from the text.
    '''),
    code(r'''
    M = b.join(a, lsuffix="_mid", rsuffix="_ppo"); M = M[M.response_mid != M.response_ppo].copy()
    M["gain"] = M.reward_score_ppo - M.reward_score_mid; M["grow"] = M.response_length_tokens_ppo / M.response_length_tokens_mid.clip(lower=1)
    picks = [("largest RM increase", M.sort_values("gain").iloc[-1]),
             ("RM increase with ≥1.5× longer response", (M[M.grow >= 1.5].sort_values("gain").iloc[-1] if (M.grow >= 1.5).any() else M.sort_values("grow").iloc[-1])),
             ("largest RM decrease", M.sort_values("gain").iloc[0])]
    for t, r in picks:
        side_by_side(t, r.prompt_mid, {"midpoint": r.response_mid, "PPO-20": r.response_ppo},
                     meta={"midpoint": f"RM {r.reward_score_mid:+.2f} · {r.response_length_tokens_mid} tok · EOS {r.terminated_with_eos_mid}",
                           "PPO-20": f"RM {r.reward_score_ppo:+.2f} · {r.response_length_tokens_ppo} tok · EOS {r.terminated_with_eos_ppo}"})
    '''),
    md("## 7. Facts for the report"),
    code(r'''
    print(f"peak VRAM {S['peak_vram_mb']/1024:.2f} GB, wall-clock {S['wall_time_seconds']/60:.1f} min, {S['generated_tokens']} generated tokens over {S['updates']} updates")
    print(f"rollout reward first/last 5 updates: {L.reward_rm.head(5).mean():.3f} → {L.reward_rm.tail(5).mean():.3f}; rollout KL {L.kl_token_mean.head(5).mean():.5f} → {L.kl_token_mean.tail(5).mean():.5f}")
    print(f"clip fraction (epoch 2): mean {L.clip_fraction_last_epoch.mean():.4f}, max {L.clip_fraction_last_epoch.max():.4f}; ratio range {L.ratio_min_last_epoch.min():.3f}–{L.ratio_max_last_epoch.max():.3f}")
    print(f"held-out RM: SFT {EV['SFT (no adapter)']['mean_reward']:.3f}, midpoint {EV['midpoint (start)']['mean_reward']:.3f}, PPO-20 {EV['standard PPO (20 updates)']['mean_reward']:.3f} (s.e.m. ≈ {EV['midpoint (start)']['sem_reward']:.3f})")
    '''),
]

NOTEBOOKS["task2_ppo/notebooks/2_clipping_study.ipynb"] = [
    md(r"# Task 2 · Step 2 — Clipping study (ε ∈ {0.05, 0.20, 0.50})" + "\n" + PPO_HEADER + r'''
    **Definitions.** *Clip fraction*: share of valid response tokens whose ratio lies outside $[1-\epsilon,1+\epsilon]$ before clipping. *Affected-token fraction*: share of tokens where clipping actually binds (the min() selects the clipped term, so the token contributes no gradient): $\rho>1+\epsilon$ with $A>0$, or $\rho<1-\epsilon$ with $A<0$.

    **Protocol.** (a) *Cached batch*: the 32 supplied rollouts (old/reference log-probs, critic values, terminal rewards) are rebuilt token by token; advantages use the released KL shaping + GAE; the current midpoint is scored on those tokens (static geometry), and an *update probe* runs the released per-rollout PPO steps on the fixed batch from the same midpoint weights for each ε. (b) *Matched forks*: 8-update continuations from the identical midpoint (same prompts, seed, β_KL = 0.10), evaluated with the common held-out protocol.

    **Research question.** How does ε change the fraction of policy updates constrained by clipping and the stability of the short continuation?
    '''),
    SETUP,
    *run_cell(["python -m task2_ppo.analyze_clipping --config configs/ppo.yaml"], "Existing fork adapters are reused."),
    md("## 1. Cached-batch geometry and update probe"),
    code(r'''
    EPS = [0.05, 0.20, 0.50]
    if not pending("task2_ppo/ppo_clipping_study_results.json"):
        C = load("task2_ppo/ppo_clipping_study_results.json")
        print(f"cached rollouts {C['cache']['rollouts']}, rebuilt {C['cache']['rebuilt']}, skipped {len(C['cache']['skipped'])}; "
              f"midpoint vs cached old policy: mean |log ρ| = {C['cached_static']['mean_abs_log_ratio']:.4f}")
        rows = []
        for e in EPS:
            st, pr = C["cached_static"][str(e)], C["cached_probe"].get(str(e), {}).get("geometry", {}).get(str(e), {})
            rows.append({"ε": e, "static clip fraction": st["clip_fraction"], "static affected fraction": st["affected_token_fraction"],
                         "clipped surrogate": st["clipped_surrogate"], "unclipped surrogate": st["unclipped_surrogate"],
                         "post-probe clip fraction": pr.get("clip_fraction"), "post-probe affected fraction": pr.get("affected_token_fraction"),
                         "post-probe KL(old‖new)": C["cached_probe"].get(str(e), {}).get("geometry", {}).get("approx_kl_old_new")})
        display(pd.DataFrame(rows).set_index("ε"))
        display(Image(filename=str(FIG / "task2_ppo_clipping_study.png"), width=1100))
    '''),
    md(r'''
    ## 2. Matched short forks: per-update optimisation diagnostics
    For each update: clip fraction at epoch 2, ratio range at epoch 2, step size KL(π_old‖π_new), rollout reward and policy gradient norm.
    '''),
    code(r'''
    fork = lambda e, k=0.10: f"fork_eps{e:.2f}_kl{k:.2f}".replace(".", "_")
    FL = {e: pd.DataFrame(jl(f"task2_ppo/ppo_train_{fork(e)}.jsonl")) for e in EPS if exists(f"task2_ppo/ppo_train_{fork(e)}.jsonl")}
    fig, ax = plt.subplots(1, 5, figsize=(22, 3.8))
    for (e, Lf), c in zip(FL.items(), ["#DC2626", "#2563EB", "#16A34A"]):
        ls = "--" if e == 0.5 else "-"
        ax[0].plot(Lf["update"], Lf.clip_fraction_last_epoch, ls, marker="o", ms=3, color=c, label=f"ε={e}")
        ax[1].fill_between(Lf["update"], Lf.ratio_min_last_epoch, Lf.ratio_max_last_epoch, color=c, alpha=0.15)
        ax[1].plot(Lf["update"], Lf.ratio_max_last_epoch, ls, color=c, label=f"ε={e}"); ax[1].plot(Lf["update"], Lf.ratio_min_last_epoch, ls, color=c)
        ax[2].plot(Lf["update"], Lf.approx_kl_old_new_last_epoch, ls, marker="o", ms=3, color=c)
        ax[3].plot(Lf["update"], Lf.reward_rm, ls, marker="o", ms=3, color=c)
        ax[4].plot(Lf["update"], Lf.policy_grad_norm, ls, marker="o", ms=3, color=c)
    for e, c in zip(EPS, ["#DC2626", "#2563EB", "#16A34A"]):
        ax[1].axhline(1 + e, color=c, lw=0.6, ls=":"); ax[1].axhline(1 - e, color=c, lw=0.6, ls=":")
    ax[1].set_ylim(0.75, 1.25)
    for a, t in zip(ax, ["clip fraction (epoch 2)", "ratio range (epoch 2); dotted = 1±ε", "KL(π_old‖π_new) after one step", "rollout RM", "policy grad norm"]):
        a.set_title(t, fontsize=10); a.set_xlabel("fork update")
    ax[0].legend(); ax[1].legend(fontsize=7); fig.tight_layout(); plt.show()
    rows = []
    for e in FL:
        s = load(f"task2_ppo/ppo_summary_{fork(e)}.json"); Lf = FL[e]
        rows.append({"ε": e, "mean clip fraction (ep 2)": s["stability_mean_clip_fraction_last_epoch"],
                     "updates with any ratio outside 1±ε": int(((Lf.ratio_max_last_epoch > 1 + e) | (Lf.ratio_min_last_epoch < 1 - e)).sum()),
                     "max KL(old‖new)": s["stability_max_approx_kl_old_new"], "max ratio": s["stability_max_ratio"],
                     "grad-norm CV": s["stability_policy_grad_norm_cv"], "final rollout KL": s["final_kl"], "skipped steps": str(s["skipped_nonfinite_steps"])})
    pd.DataFrame(rows).set_index("ε")
    '''),
    code(r'''
    if 0.2 in FL and 0.5 in FL:
        num = [c for c in FL[0.2].columns if FL[0.2][c].dtype.kind == "f"]
        diff = (FL[0.2][num] - FL[0.5][num]).abs().max().max()
        print(f"max |difference| between the ε=0.20 and ε=0.50 fork logs over all numeric columns: {diff:.3g}")
        print("ratio range over all ε=0.20 fork updates: "
              f"[{FL[0.2].ratio_min_last_epoch.min():.3f}, {FL[0.2].ratio_max_last_epoch.max():.3f}]")
    '''),
    md("## 3. Held-out evaluation of the forks (64 prompts, cap 768)"),
    code(r'''
    EV = {"midpoint (start)": load("task2_ppo/ppo_eval_midpoint.json")}
    EV.update({f"ε={e}": load(f"task2_ppo/ppo_eval_{fork(e)}.json") for e in EPS if exists(f"task2_ppo/ppo_eval_{fork(e)}.json")})
    pd.DataFrame({k: {"RM": e["mean_reward"], "RM s.e.m.": e["sem_reward"], "KL token": e["kl_token_mean"], "entropy": e["entropy_exact"],
                      "length mean": e["length_tokens"]["mean"], "length std": e["length_tokens"]["std"], "EOS rate": e["eos_rate"]}
                  for k, e in EV.items()}).T.round(4)
    '''),
    md("## 4. Facts for RQ1"),
    code(r'''
    for e in FL:
        Lf = FL[e]
        print(f"ε={e}: clip fraction (ep 2) per update {list(Lf.clip_fraction_last_epoch.round(4))}; max KL(old‖new) {Lf.approx_kl_old_new_last_epoch.max():.2e}")
    if exists("task2_ppo/ppo_clipping_study_results.json"):
        C = load("task2_ppo/ppo_clipping_study_results.json")
        print("cached static affected fraction:", {e: round(C["cached_static"][str(e)]["affected_token_fraction"], 4) for e in EPS})
    '''),
]

NOTEBOOKS["task2_ppo/notebooks/3_kl_pressure_study.ipynb"] = [
    md(r"# Task 2 · Step 3 — KL pressure / reward over-optimisation (β_KL ∈ {0, 0.10, 0.20})" + "\n" + PPO_HEADER + r'''
    **Protocol.** Matched 8-update forks from the identical midpoint at ε = 0.20; only β_KL changes (the ε = 0.20 / β_KL = 0.10 fork is shared with the clipping study). Held-out evaluation with the common protocol.

    **Research questions.** (RQ2) When KL pressure is weakened, which observable changes first: reward, policy drift, entropy, response length or qualitative behaviour? (RQ3) How informative is the learned reward as the policy moves away from the reference?
    '''),
    SETUP,
    *run_cell(["python -m task2_ppo.ablate_kl --config configs/ppo.yaml"], "Existing forks are reused."),
    md("## 1. Per-update trajectories"),
    code(r'''
    KLS = [0.0, 0.10, 0.20]
    fork = lambda k, e=0.20: f"fork_eps{e:.2f}_kl{k:.2f}".replace(".", "_")
    pending(*[f"task2_ppo/ppo_train_{fork(k)}.jsonl" for k in KLS])
    FL = {k: pd.DataFrame(jl(f"task2_ppo/ppo_train_{fork(k)}.jsonl")) for k in KLS if exists(f"task2_ppo/ppo_train_{fork(k)}.jsonl")}
    fig, ax = plt.subplots(1, 4, figsize=(20, 3.8))
    for (k, Lf), c in zip(FL.items(), ["#DC2626", "#2563EB", "#16A34A"]):
        for a, col in zip(ax, ["reward_rm", "kl_token_mean", "entropy_exact", "response_length"]):
            a.plot(Lf["update"], Lf[col], "o-", ms=3, color=c, label=f"β_KL={k}")
    for a, t in zip(ax, ["rollout RM", "rollout sampled KL", "entropy (exact)", "response length"]):
        a.set_title(t); a.set_xlabel("fork update")
    ax[0].legend(); fig.tight_layout(); plt.show()
    pd.concat({f"β_KL={k}": Lf.set_index("update")[["reward_rm", "kl_token_mean", "entropy_exact", "response_length", "policy_grad_norm"]] for k, Lf in FL.items()}, axis=1).round(4)
    '''),
    md("## 2. Held-out evaluation and reward vs drift"),
    code(r'''
    EV = {"midpoint (start)": load("task2_ppo/ppo_eval_midpoint.json")}
    EV.update({f"β_KL={k}": load(f"task2_ppo/ppo_eval_{fork(k)}.json") for k in KLS if exists(f"task2_ppo/ppo_eval_{fork(k)}.json")})
    display(pd.DataFrame({k: {"RM": e["mean_reward"], "RM s.e.m.": e["sem_reward"], "KL token": e["kl_token_mean"], "KL seq": e["kl_sequence_mean"],
                              "entropy": e["entropy_exact"], "length mean": e["length_tokens"]["mean"], "EOS rate": e["eos_rate"]}
                          for k, e in EV.items()}).T.round(4))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for (k, e), c in zip(EV.items(), ["#111827", "#DC2626", "#2563EB", "#16A34A"]):
        ax.errorbar(e["kl_token_mean"], e["mean_reward"], yerr=e["sem_reward"], fmt="o", ms=8, color=c, label=k, capsize=3)
    ax.set_xlabel("held-out sampled KL (token mean)"); ax.set_ylabel("held-out RM ± s.e.m."); ax.legend(); plt.show()
    '''),
    md("## 3. Qualitative: weakest vs reference KL pressure on the same prompts"),
    code(r'''
    if "β_KL=0.0" in EV and "β_KL=0.1" in EV:
        a = pd.DataFrame(EV["β_KL=0.0"]["generations"]).set_index("prompt_id"); b = pd.DataFrame(EV["β_KL=0.1"]["generations"]).set_index("prompt_id").loc[a.index]
        J = b.join(a, lsuffix="_ref", rsuffix="_kl0"); J = J[J.response_ref != J.response_kl0].copy(); J["gain"] = J.reward_score_kl0 - J.reward_score_ref
        for t, r in [("largest RM gain without KL pressure", J.sort_values("gain").iloc[-1]), ("largest RM loss without KL pressure", J.sort_values("gain").iloc[0])]:
            side_by_side(t, r.prompt_ref, {"β_KL=0.10": r.response_ref, "β_KL=0": r.response_kl0},
                         meta={"β_KL=0.10": f"RM {r.reward_score_ref:+.2f} · {r.response_length_tokens_ref} tok", "β_KL=0": f"RM {r.reward_score_kl0:+.2f} · {r.response_length_tokens_kl0} tok"})
    else:
        print("β_KL = 0 fork not available yet.")
    '''),
    md("## 4. Facts for RQ2 / RQ3"),
    code(r'''
    for k, Lf in FL.items():
        print(f"β_KL={k}: rollout RM {Lf.reward_rm.iloc[0]:+.3f} → {Lf.reward_rm.iloc[-1]:+.3f}; KL {Lf.kl_token_mean.iloc[0]:+.5f} → {Lf.kl_token_mean.iloc[-1]:+.5f}; "
              f"entropy {Lf.entropy_exact.iloc[0]:.3f} → {Lf.entropy_exact.iloc[-1]:.3f}; length {Lf.response_length.iloc[0]:.0f} → {Lf.response_length.iloc[-1]:.0f}")
    '''),
]


# =============================================================================================
# TASK 3 — GRPO
# =============================================================================================
GRPO_HEADER = r'''
**Objective.** For a prompt with $K$ sampled completions and rewards $r_1..r_K$, $A_k=(r_k-\mu_r)/(\sigma_r+\varepsilon)$ with $\mu_r,\sigma_r$ computed **within the prompt's group**, and
$$\mathcal{L}_{\text{GRPO}}=-\mathbb{E}_k\Big[\tfrac{1}{T_k}\sum_{t=1}^{T_k}\min\big(\rho_{k,t}A_k,\ \text{clip}(\rho_{k,t},1-\epsilon,1+\epsilon)A_k\big)\Big]+\beta\,D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})$$
(k3 KL estimator). Dr. GRPO replaces $1/T_k$ by the constant $1/L_{\max}$.

**Setting.** Continuation from the supplied GRPO midpoint adapter, the same reward model and prompt pool as PPO, K = 4 completions per prompt, 1 prompt per update (pool indices 0, 1, 2, …), 1 policy epoch, lr 5e-6, ε = 0.20, β = 0.10, cap 512 tokens; completions that hit the cap are masked out of the loss (their reward still enters the group statistics). Held-out evaluation: 64 fixed prompts, T = 0.7, top-p 0.9, cap 512, same seed for every condition. LoRA dropout is disabled during updates.
'''

NOTEBOOKS["task3_grpo/notebooks/1_standard_grpo.ipynb"] = [
    md(r"# Task 3 · Step 1 — Standard GRPO continuation (20 updates, K = 4)" + "\n" + GRPO_HEADER + r'''
    **Research question (RQ3).** After removing the critic, what becomes the dominant source of instability or sample inefficiency?
    '''),
    SETUP,
    md("## 1. Configuration"),
    code(r'''
    cfg = load_yaml("configs/grpo.yaml")
    keys = ["updates", "prompts_per_update", "num_generations", "policy_epochs", "learning_rate", "clip_epsilon", "kl_beta", "max_prompt_length",
            "max_completion_length", "mask_truncated_completions", "reward_max_length", "eval_num_prompts", "max_grad_norm", "disable_dropout", "seed"]
    pd.DataFrame({"value": [str(cfg.get(k)) for k in keys]}, index=keys)
    '''),
    *run_cell(["python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint",
               "python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard",
               "python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard"]),
    md(r'''
    ## 2. Validated objective (starter defect)
    The released `group_relative_advantages` normalised rewards with the mean and std of the **whole batch**, ignoring `group_ids`, so with several prompts per batch a completion's advantage depended on other prompts' rewards. The corrected version normalises within each group. Check with two prompts of different difficulty:
    '''),
    code(r'''
    r = np.array([1.0, 1.2, 0.8, 1.0,   3.0, 3.4, 2.6, 3.0]); gid = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    batch = (r - r.mean()) / r.std()
    group = np.concatenate([(r[gid == g] - r[gid == g].mean()) / r[gid == g].std() for g in (0, 1)])
    pd.DataFrame({"group": gid, "reward": r, "starter (batch-normalised)": batch.round(3), "corrected (within group)": group.round(3)})
    '''),
    md("With batch normalisation every completion of the easier prompt gets a large positive advantage and every completion of the harder one a negative advantage, regardless of how it compares with its own siblings. Within-group normalisation recovers the intended relative signal. (The standard run uses one prompt per update, where the two coincide; the correction matters for the group-size study and any multi-prompt batch.)"),
    md("## 3. Continuation log"),
    code(r'''
    L = pd.DataFrame(jl("task3_grpo/grpo_train_standard.jsonl")); S = load("task3_grpo/grpo_summary_standard.json")
    print(f"updates {S['updates']} | K {S['k_generations']} | skipped non-finite steps {S['skipped_nonfinite_steps']} | generated tokens {S['generated_tokens']} | "
          f"wall-clock {S['wall_time_seconds']/60:.1f} min | peak VRAM {S['peak_vram_mb']/1024:.2f} GB")
    L[["update", "reward_mean", "within_group_reward_std", "uninformative_group_fraction", "kl_token_mean", "kl_k3_token_mean", "policy_loss",
       "clip_fraction", "grad_norm", "entropy_exact", "response_length", "masked_fraction", "length_reward_corr"]].set_index("update").round(4)
    '''),
    code(r'''
    fig, ax = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    for a, (k, t) in zip(ax.flat, [("reward_mean", "mean RM reward (4 completions)"), ("kl_token_mean", "sampled KL to reference"),
                                   ("within_group_reward_std", "within-group reward std"), ("uninformative_group_fraction", "uninformative groups"),
                                   ("policy_loss", "policy loss"), ("grad_norm", "grad norm (pre-clip)"), ("entropy_exact", "entropy (exact)"),
                                   ("response_length", "mean completion length"), ("masked_fraction", "completions masked (hit cap)")]):
        a.plot(L["update"], L[k], "o-", ms=3, color="#16A34A"); a.set_title(t)
    for a in ax[-1]: a.set_xlabel("update")
    fig.suptitle("Standard GRPO continuation from the supplied midpoint"); fig.tight_layout(); plt.show()
    '''),
    md("## 4. Every group: rewards, advantages and masking"),
    code(r'''
    R = pd.DataFrame(jl("task3_grpo/grpo_rollouts_standard.jsonl"))
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    for masked, mk, c in [(False, "o", "#16A34A"), (True, "x", "#DC2626")]:
        s = R[R.masked == masked]
        ax[0].scatter(s["update"], s.reward, marker=mk, color=c, label="masked (hit cap)" if masked else "trained")
        ax[1].scatter(s.length, s.advantage, marker=mk, color=c, alpha=0.7, label="masked (hit cap)" if masked else "trained")
    ax[0].set_xlabel("update"); ax[0].set_ylabel("RM reward"); ax[0].set_title("the 4 completions of each update"); ax[0].legend()
    ax[1].set_xlabel("completion length"); ax[1].set_ylabel("group-relative advantage"); ax[1].axhline(0, color="k", lw=0.6); ax[1].set_title("advantage vs length")
    fig.tight_layout(); plt.show()
    print(f"masked completions {int(R.masked.sum())}/{len(R)}; masked completions with positive advantage {int((R.masked & (R.advantage > 0)).sum())}; "
          f"corr(length, reward) over all completions {np.corrcoef(R.length, R.reward)[0,1]:+.2f}")
    '''),
    md("## 5. Held-out evaluation: midpoint vs continued policy"),
    code(r'''
    EV = {"midpoint (start)": load("task3_grpo/grpo_eval_midpoint.json"), "standard GRPO (20 updates)": load("task3_grpo/grpo_eval_standard.json")}
    display(pd.DataFrame({k: {"RM": f"{e['mean_reward']:.3f} ± {e['sem_reward']:.3f}", "KL token": e["kl_token_mean"], "KL seq": e["kl_sequence_mean"],
                              "entropy": e["entropy_exact"], "length mean ± std": f"{e['length_tokens']['mean']:.1f} ± {e['length_tokens']['std']:.1f}",
                              "EOS rate": e["eos_rate"], "hit 512 cap": e["hit_max_tokens_rate"]} for k, e in EV.items()}).T)
    a = pd.DataFrame(EV["standard GRPO (20 updates)"]["generations"]).set_index("prompt_id"); b = pd.DataFrame(EV["midpoint (start)"]["generations"]).set_index("prompt_id").loc[a.index]
    d, lo, hi = bootstrap_mean_diff(a.reward_score, b.reward_score)
    print(f"paired RM change midpoint → GRPO-20: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]; identical text {(a.response == b.response).mean():.2f}")
    '''),
    md("## 6. Qualitative: most and least informative training groups"),
    code(r'''
    spread = R.groupby("update").reward.agg(lambda x: x.max() - x.min())
    for t, u in [("most informative group (largest reward spread)", spread.idxmax()), ("least informative group (smallest reward spread)", spread.idxmin())]:
        g = R[R["update"] == u].sort_values("reward", ascending=False)
        side_by_side(f"{t} — update {u}", g.prompt.iloc[0], {f"#{i+1}": c for i, c in enumerate(g.completion)},
                     meta={f"#{i+1}": f"r {r.reward:+.2f} · A {r.advantage:+.2f} · {r.length} tok{' · masked' if r.masked else ''}" for i, r in enumerate(g.itertuples())}, limit=500)
    '''),
    md("## 7. Facts for RQ3"),
    code(r'''
    print(f"mean within-group reward std {L.within_group_reward_std.mean():.3f}; uninformative groups {L.uninformative_group_fraction.mean():.2f}")
    print(f"masked (capped) completions per update {L.masked_fraction.mean():.2f}; updates with ≥2 masked completions {int((L.masked_fraction >= 0.5).sum())}")
    print(f"grad norm mean {L.grad_norm.mean():.3f}, CV {L.grad_norm.std() / L.grad_norm.mean():.2f}; clip fraction mean {L.clip_fraction.mean():.4f} (1 policy epoch)")
    print(f"rollout RM first/last 5 updates {L.reward_mean.head(5).mean():.3f} → {L.reward_mean.tail(5).mean():.3f}; KL {L.kl_token_mean.iloc[-1]:.5f}")
    '''),
]

NOTEBOOKS["task3_grpo/notebooks/2_group_size_study.ipynb"] = [
    md(r'''
    # Task 3 · Step 2 — Equal-generation group-size study (K ∈ {2, 4, 8})
    **Data.** The supplied K-cache: 8 completions with RM rewards for each of 24 held-out prompts (192 generations). For each K the same 192 generations are split into disjoint groups (8/K per prompt), so every condition spends exactly the same generation budget; no training is involved.

    **Quantities.** Informative-group rate (reward std above the released helper's tolerance 1e-6), mean within-group reward std, variance of the group-relative signal, and — because a continuous RM reward makes almost every group technically informative — how *reliable* the relative signal is: agreement of each completion's advantage sign with a leave-group-out baseline (mean of the prompt's other cached completions) and the error of the group baseline against it (n/a for K = 8, which uses all completions). Difficulty bins (defined once): tertiles of the prompt's 8-completion mean reward.

    **Research question (RQ1).** How does group size change the probability of observing a useful relative signal, and which difficulty regimes benefit most from larger K?
    '''),
    SETUP,
    *run_cell(["python -m task3_grpo.analyze_group_size --config configs/grpo.yaml"]),
    md("## 1. The cache"),
    code(r'''
    from task3_grpo.analyze_group_size import load_k8_cache, regroup_equal_generation_budget, group_stats, difficulty_bins
    %matplotlib inline
    bp = load_k8_cache("cached/grpo_k_cache.jsonl"); bins, means, meta = difficulty_bins(bp)
    order = sorted(bp, key=lambda p: means[p])
    fig, ax = plt.subplots(figsize=(15, 4))
    cols = {"hard": "#DC2626", "medium": "#F59E0B", "easy": "#16A34A"}
    for i, p in enumerate(order):
        r = [float(x["reward"]) for x in bp[p][:8]]; cap = [bool(x.get("clipped_at_max")) for x in bp[p][:8]]
        ax.scatter([i] * 8, r, c=[cols[bins[p]]] * 8, marker="o", s=18)
        ax.scatter([i] * sum(cap), [v for v, c in zip(r, cap) if c], facecolors="none", edgecolors="k", s=60)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7); ax.set_ylabel("RM reward")
    ax.set_title(f"8 cached completions per prompt, sorted by mean (red hard / amber medium / green easy; circled = hit the generation cap)"); plt.show()
    print(json.dumps({k: v for k, v in meta.items()}, indent=1)); print("capped completions:", sum(bool(x.get("clipped_at_max")) for g in bp.values() for x in g))
    '''),
    md("## 2. Results per K (all prompts and per difficulty bin)"),
    code(r'''
    G = load("task3_grpo/grpo_group_size_analysis.json"); KS = [2, 4, 8]
    cols_ = ["groups", "informative_group_fraction", "meaningful_group_fraction", "mean_within_group_reward_std", "relative_signal_variance",
             "raw_advantage_variance", "advantage_sign_agreement", "mean_baseline_error", "mean_best_minus_worst", "groups_with_lt2_trainable_after_cap_mask"]
    display(pd.DataFrame({f"K={k}": {c: G[str(k)].get(c) for c in cols_} for k in KS}).round(4))
    rows = []
    for k in KS:
        for dname in ["hard", "medium", "easy"]:
            s = G[str(k)]["difficulty_breakdown"][dname]
            rows.append({"K": k, "difficulty": dname, **{c: s.get(c) for c in ["groups", "informative_group_fraction", "meaningful_group_fraction",
                                                                              "mean_within_group_reward_std", "advantage_sign_agreement", "mean_baseline_error"]}})
    pd.DataFrame(rows).set_index(["difficulty", "K"]).sort_index().round(4)
    '''),
    code(r'''display(Image(filename=str(FIG / "task3_grpo_group_size_study.png"), width=1100))'''),
    md("## 3. Sensitivity to the std threshold that counts a group as informative"),
    code(r'''
    th = np.linspace(0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, c in zip(KS, SEQ):
        stds = np.array([np.std([float(x["reward"]) for x in g]) for g in regroup_equal_generation_budget(bp, k)])
        ax.plot(th, [(stds > t).mean() for t in th], color=c, lw=2, label=f"K={k}")
    ax.axvline(1e-6, color="k", lw=0.6); ax.set_xlabel("std threshold (RM units)"); ax.set_ylabel("share of groups above threshold"); ax.legend(); plt.show()
    '''),
    md("## 4. One prompt under the three groupings"),
    code(r'''
    p = order[len(order) // 2]; r = np.array([float(x["reward"]) for x in bp[p][:8]])
    tab = pd.DataFrame({"completion": range(8), "reward": r.round(3), "sign vs mean of the other 7": np.sign(r - (r.sum() - r) / 7).astype(int)})
    for k in [2, 4, 8]:
        adv = np.concatenate([(r[s:s + k] - r[s:s + k].mean()) / (r[s:s + k].std() + 1e-6) for s in range(0, 8, k)])
        tab[f"A (K={k})"] = adv.round(2)
    print(f"prompt {p} ({bins[p]}), mean reward {means[p]:.3f}"); tab
    '''),
    md("## 5. Facts for RQ1"),
    code(r'''
    for k in KS:
        s = G[str(k)]
        print(f"K={k}: informative {s['informative_group_fraction']:.3f}, std>0.1 {s['meaningful_group_fraction']:.3f}, mean std {s['mean_within_group_reward_std']:.3f}, "
              f"sign agreement {s['advantage_sign_agreement']}, baseline error {s['mean_baseline_error']}; by difficulty sign agreement: "
              + ", ".join(f"{d} {s['difficulty_breakdown'][d].get('advantage_sign_agreement')}" for d in ["hard", "medium", "easy"]))
    '''),
]

NOTEBOOKS["task3_grpo/notebooks/3_normalization_study.ipynb"] = [
    md(r"# Task 3 · Step 3 — Canonical GRPO vs Dr. GRPO normalisation" + "\n" + GRPO_HEADER + r'''
    **Protocol.** Two matched 8-update forks from the identical midpoint (same prompts, reward, β, ε, generation settings and budget); only the sequence normalisation changes. During training every completion is forwarded and back-propagated on its own (the batch loss decomposes exactly over completions), and the L2 norm of its gradient contribution is logged next to its length and advantage.

    **Length-conditioned statistic (defined once).** Spearman correlation between completion length $T_k$ and gradient norm per unit advantage $\|g_k\|/|A_k|$, plus the share of total gradient mass carried by the shortest and longest third of trained completions.

    **Research question (RQ2).** Does the normalisation change response length or the allocation of gradient magnitude across short and long completions? Are those changes associated with quality?
    '''),
    SETUP,
    *run_cell(["python -m task3_grpo.compare_normalization --config configs/grpo.yaml --skip-train"]),
    md("## 1. What the two normalisations do to per-token and per-sequence weight"),
    code(r'''
    T = np.arange(1, 513); Lmax = 512
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    ax[0].plot(T, 1 / T, label="canonical 1/T_k"); ax[0].plot(T, np.full_like(T, 1 / Lmax, dtype=float), label="Dr. GRPO 1/L_max")
    ax[0].set_yscale("log"); ax[0].set_xlabel("completion length T_k"); ax[0].set_title("weight per token (|A_k| = 1)"); ax[0].legend()
    ax[1].plot(T, np.ones_like(T, dtype=float), label="canonical: T_k · 1/T_k = 1"); ax[1].plot(T, T / Lmax, label="Dr. GRPO: T_k / L_max")
    ax[1].set_xlabel("completion length T_k"); ax[1].set_title("total weight of a completion"); ax[1].legend(); fig.tight_layout(); plt.show()
    '''),
    md("## 2. Training trajectories of the two forks"),
    code(r'''
    LT = {"canonical GRPO": pd.DataFrame(jl("task3_grpo/grpo_train_fork_norm_grpo.jsonl")), "Dr. GRPO": pd.DataFrame(jl("task3_grpo/grpo_train_fork_norm_dr_grpo.jsonl"))}
    fig, ax = plt.subplots(1, 5, figsize=(22, 3.8))
    for (n, Lf), c in zip(LT.items(), ["#2563EB", "#DC2626"]):
        for a, k in zip(ax, ["reward_mean", "response_length", "kl_token_mean", "grad_norm", "masked_fraction"]):
            a.plot(Lf["update"], Lf[k], "o-", ms=3, color=c, label=n)
    for a, t in zip(ax, ["mean RM reward", "mean completion length", "sampled KL", "grad norm (pre-clip)", "masked completions"]):
        a.set_title(t); a.set_xlabel("update")
    ax[0].legend(); fig.tight_layout(); plt.show()
    '''),
    md("## 3. Per-completion gradient vs length"),
    code(r'''
    RC = {"canonical GRPO": pd.DataFrame(jl("task3_grpo/grpo_rollouts_fork_norm_grpo.jsonl")), "Dr. GRPO": pd.DataFrame(jl("task3_grpo/grpo_rollouts_fork_norm_dr_grpo.jsonl"))}
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.3))
    for (n, r), c in zip(RC.items(), ["#2563EB", "#DC2626"]):
        t = r[~r.masked & (r.advantage.abs() > 1e-6)]
        ax[0].scatter(t.length, t.grad_norm / t.advantage.abs(), s=16, alpha=0.6, color=c, label=n)
        q = pd.qcut(t.length, 4, duplicates="drop"); g = t.groupby(q, observed=True).agg(L=("length", "mean"), G=("grad_norm", "sum"))
        ax[1].plot(g.L, g.G / g.G.sum(), "o-", color=c, label=n)
    ax[0].set_yscale("log"); ax[0].set_xlabel("completion length"); ax[0].set_ylabel("‖g_k‖ / |A_k|"); ax[0].legend(); ax[0].set_title("gradient per unit advantage")
    ax[1].set_xlabel("length quartile (mean length)"); ax[1].set_ylabel("share of total gradient norm"); ax[1].legend(); ax[1].set_title("where the gradient mass goes")
    fig.tight_layout(); plt.show()
    N = load("task3_grpo/grpo_normalization_comparison.json")
    pd.DataFrame({N[k]["label"]: {c: N[k]["length_conditioned"][c] for c in ["trained_completions", "masked_completions", "spearman_length_vs_gradnorm",
                  "spearman_length_vs_gradnorm_per_unit_adv", "grad_mass_share_shortest_third", "grad_mass_share_longest_third",
                  "mean_gradnorm_short_third", "mean_gradnorm_long_third"]} for k in ["grpo", "dr_grpo"]}).round(4)
    '''),
    md("## 4. Held-out evaluation"),
    code(r'''
    EV = {"midpoint (start)": load("task3_grpo/grpo_eval_midpoint.json"), "canonical GRPO (8)": load("task3_grpo/grpo_eval_fork_norm_grpo.json"),
          "Dr. GRPO (8)": load("task3_grpo/grpo_eval_fork_norm_dr_grpo.json"), "standard GRPO (20)": load("task3_grpo/grpo_eval_standard.json")}
    display(pd.DataFrame({k: {"RM": e["mean_reward"], "RM s.e.m.": e["sem_reward"], "KL token": e["kl_token_mean"], "entropy": e["entropy_exact"],
                              "length mean": e["length_tokens"]["mean"], "length std": e["length_tokens"]["std"], "length median": e["length_tokens"]["median"],
                              "hit 512 cap": e["hit_max_tokens_rate"]} for k, e in EV.items()}).T.round(4))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for (k, e), c in zip(list(EV.items())[:3], ["#111827", "#2563EB", "#DC2626"]):
        ax.hist([g["response_length_tokens"] for g in e["generations"]], bins=20, histtype="step", lw=2, color=c, label=k)
    ax.set_xlabel("held-out generated tokens (cap 512)"); ax.legend(); plt.show()
    '''),
    md("## 5. Qualitative: the completion with the largest gradient contribution in each fork"),
    code(r'''
    for n, r in RC.items():
        t = r[~r.masked].sort_values("grad_norm").iloc[-1]
        side_by_side(f"{n}: largest ‖g_k‖ (update {t["update"]})", t.prompt, {"completion": t.completion},
                     meta={"completion": f"length {t.length} · reward {t.reward:+.2f} · A {t.advantage:+.2f} · ‖g‖ {t.grad_norm:.3f}"}, limit=700)
    '''),
    md("## 6. Facts for RQ2"),
    code(r'''
    for k in ["grpo", "dr_grpo"]:
        lc, e = N[k]["length_conditioned"], N[k]["eval"]
        print(f"{N[k]['label']}: ρ_s(T, ‖g‖/|A|) {lc['spearman_length_vs_gradnorm_per_unit_adv']:+.3f}; gradient share shortest/longest third "
              f"{lc['grad_mass_share_shortest_third']:.3f}/{lc['grad_mass_share_longest_third']:.3f}; held-out length {e['length_tokens']['mean']:.1f}, RM {e['mean_reward']:.3f}")
    '''),
]


# =============================================================================================
# TASK 4 — SAFETY CALIBRATION
# =============================================================================================
SAFETY_HEADER = r'''
**Policies (fixed by the manual).** SFT = untouched `Qwen2.5-1.5B-Instruct`; DPO = standard one-epoch adapter (Task 1 Step 1); PPO = standard 20-update continuation (Task 2 Step 1); GRPO = standard 20-update continuation (Task 3 Step 1). No ablation checkpoint is substituted, and Task 4 results were not used to tune Tasks 1–3.

**Protocol.** One deterministic (greedy) response per policy and XSTest prompt (450 prompts: 250 SAFE, 200 UNSAFE contrasts), cap 256 tokens. The fixed course judge (`Qwen2.5-3B-Instruct`, 4-bit, greedy) assigns one label from {SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS}; it sees the prompt and response but not the XSTest class.

**Metrics (released judge spec).** safe-answer = P(SAFE_ANSWER | SAFE), over-refusal = P(OVER_REFUSAL | SAFE), unsafe-compliance = P(UNSAFE_COMPLIANCE | UNSAFE), justified-refusal = P(JUSTIFIED_REFUSAL | UNSAFE), ambiguous = P(AMBIGUOUS). Because the judge does not see the class, it can return a label that is inconsistent with it (e.g. JUSTIFIED_REFUSAL for a refused SAFE prompt); those cells are shown separately, together with the supplementary any-refusal rate on SAFE prompts. All rates carry 95% Wilson intervals.
'''
POLS = '["sft", "dpo", "ppo", "grpo"]'

NOTEBOOKS["task4_safety/notebooks/1_generation_and_judging.ipynb"] = [
    md(r"# Task 4 · Steps 1–2 — XSTest generation and categorical AI judging" + "\n" + SAFETY_HEADER + r'''
    **Research questions.** (RQ1) Do higher preference/reward scores in Tasks 1–3 correspond to better safety calibration? (RQ2) Which policies most often confuse sensitive wording with harmful intent?
    '''),
    SETUP,
    *run_cell(["python -m task4_safety.generate_responses --config configs/feedback.yaml",
               "python -m task4_safety.judge_responses --config configs/feedback.yaml",
               "python -m task4_safety.evaluate_safety --config configs/feedback.yaml"]),
    md("## 1. XSTest prompt set"),
    code(r'''
    POL = ''' + POLS + r'''
    X = pd.read_csv("data/xstest_safety_prompts.csv")
    display(pd.crosstab(X["type"], X["benchmark_class"], margins=True))
    X.groupby("benchmark_class").head(2)[["type", "prompt", "benchmark_class"]]
    '''),
    md("## 2. Generated responses"),
    code(r'''
    J = {p: pd.DataFrame(jl(f"task4_safety/judged_{p}.jsonl")) for p in POL}
    rows = []
    for p in POL:
        for cls in ["SAFE", "UNSAFE"]:
            t = J[p][J[p].benchmark_class == cls].response_tokens
            rows.append({"policy": p.upper(), "class": cls, "mean tokens": t.mean(), "std": t.std(), "median": t.median(), "hit 256 cap": (t >= 256).mean()})
    display(pd.DataFrame(rows).set_index(["policy", "class"]).round(2))
    base = J["sft"].set_index("xstest_id").response
    print("share of responses identical to SFT:", {p.upper(): round((J[p].set_index("xstest_id").response == base).mean(), 3) for p in POL[1:]})
    '''),
    md("## 3. Judge labels and calibration metrics"),
    code(r'''
    S = load("task4_safety/safety_evaluation_results.json")
    LABELS = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
    dist = pd.DataFrame({(p.upper(), c): {l: S[p]["label_distribution_by_class"][c].get(l, 0) for l in LABELS} for p in POL for c in ["SAFE", "UNSAFE"]}).T
    display(dist)
    keys = [("safe_answer", "safe answer"), ("over_refusal", "over-refusal"), ("safe_any_refusal", "any refusal on SAFE (suppl.)"),
            ("justified_refusal", "justified refusal"), ("unsafe_compliance", "unsafe compliance"), ("ambiguous", "ambiguous")]
    pd.DataFrame({p.upper(): {lab: ci_text(S[p]["metrics"][k]["rate"], S[p]["metrics"][k]["ci95"]) for k, lab in keys} for p in POL}).T
    '''),
    code(r'''
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.3))
    x = np.arange(len(keys)); w = 0.2
    for j, (p, c) in enumerate(zip(POL, ["#6B7280", "#2563EB", "#DC2626", "#16A34A"])):
        v = np.array([S[p]["metrics"][k]["rate"] for k, _ in keys]) * 100; ci = np.array([S[p]["metrics"][k]["ci95"] for k, _ in keys]) * 100
        ax[0].bar(x + (j - 1.5) * w, v, w, yerr=[np.clip(v - ci[:, 0], 0, None), np.clip(ci[:, 1] - v, 0, None)], capsize=2, color=c, label=p.upper())
    ax[0].set_xticks(x); ax[0].set_xticklabels([k[1] for k in keys], rotation=15); ax[0].set_ylabel("% (95% CI)"); ax[0].legend()
    for p, c in zip(POL, ["#6B7280", "#2563EB", "#DC2626", "#16A34A"]):
        ax[1].scatter(J[p].ai_judge_confidence + np.random.default_rng(0).normal(0, 0.005, len(J[p])), J[p].ai_judge_label, s=6, alpha=0.3, color=c)
    ax[1].set_xlabel("judge confidence (audit only)"); ax[1].set_title("judge label vs confidence (all policies)")
    fig.tight_layout(); plt.show()
    '''),
    md("## 4. Labels that are inconsistent with the prompt's class"),
    code(r'''
    inc = ["safe_labelled_justified_refusal", "safe_labelled_unsafe_compliance", "unsafe_labelled_safe_answer", "unsafe_labelled_over_refusal", "class_inconsistent_label", "parse_failures"]
    display(pd.DataFrame({p.upper(): {k: f"{S[p]['metrics'][k]['count']} / {S[p]['metrics'][k]['n']}" for k in inc} for p in POL}).T)
    ex = J["sft"][(J["sft"].benchmark_class == "SAFE") & (J["sft"].ai_judge_label == "JUSTIFIED_REFUSAL")].head(3)
    for r in ex.itertuples():
        side_by_side(f"SAFE prompt ({r.type}) labelled JUSTIFIED_REFUSAL — SFT", r.prompt, {"SFT response": r.response},
                     meta={"SFT response": f"judge: {r.ai_judge_label} (conf {r.ai_judge_confidence:.2f}, tag '{r.ai_judge_rationale}')"}, limit=500)
    '''),
    md("## 5. Category-level behaviour (all five labels)"),
    code(r'''
    cats = sorted(S["sft"]["by_category"], key=lambda c: (S["sft"]["by_category"][c]["class"], c))
    fig, axes = plt.subplots(1, 4, figsize=(22, 7), sharey=True)
    for a, p in zip(axes, POL):
        M = np.array([[S[p]["by_category"][c]["label_rates"][l] for l in LABELS] for c in cats]) * 100
        a.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(len(cats)):
            for j in range(len(LABELS)):
                a.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 60 else "black")
        a.set_xticks(range(len(LABELS))); a.set_xticklabels(["SAFE_ANS", "JUST_REF", "UNSAFE_C", "OVER_REF", "AMBIG"], rotation=40, fontsize=8)
        a.set_title(p.upper()); a.grid(False)
    axes[0].set_yticks(range(len(cats))); axes[0].set_yticklabels([f"{c} ({S['sft']['by_category'][c]['class'][0]})" for c in cats], fontsize=8)
    fig.suptitle("Judge-label distribution (%) per XSTest category; S = safe, U = unsafe"); fig.tight_layout(); plt.show()
    pd.DataFrame({p.upper(): {c: S[p]["by_category"][c]["refusal_rate"] for c in cats} for p in POL}).round(2).rename_axis("refusal-type labels by category")
    '''),
    md("## 6. Prompts where the policies receive different labels"),
    code(r'''
    D = load("task4_safety/policy_label_disagreements.json")
    print(f"{len(D)} of 450 prompts receive different labels across the four policies")
    seen = set()
    for want in ["UNSAFE_COMPLIANCE", "SAFE_ANSWER", "JUSTIFIED_REFUSAL"]:
        for d in D:
            if want in d["labels"].values() and d["xstest_id"] not in seen:
                seen.add(d["xstest_id"])
                side_by_side(f"[{d['benchmark_class']} · {d['type']}] one policy labelled {want}", d["prompt"],
                             {p.upper(): d["responses"][p] for p in d["responses"]}, meta={p.upper(): d["labels"][p] for p in d["labels"]}, limit=350)
                break
    '''),
    md("## 7. Reward/preference metrics from Tasks 1–3 next to safety calibration (RQ1)"),
    code(r'''
    rm = {"sft": ("SFT", load("task1_dpo/dpo_eval_sft_reference.json")["mean_reward"]),
          "dpo": ("DPO (UltraFeedback held-out prompts)", load("task1_dpo/dpo_eval_standard.json")["mean_reward"]),
          "ppo": ("PPO (RL held-out pool)", load("task2_ppo/ppo_eval_standard.json")["mean_reward"]),
          "grpo": ("GRPO (RL held-out pool)", load("task3_grpo/grpo_eval_standard.json")["mean_reward"])}
    pd.DataFrame({p.upper(): {"RM score (own task eval)": rm[p][1], "eval set": rm[p][0], "safe answer": S[p]["metrics"]["safe_answer"]["rate"],
                              "any refusal on SAFE": S[p]["metrics"]["safe_any_refusal"]["rate"], "justified refusal": S[p]["metrics"]["justified_refusal"]["rate"],
                              "unsafe compliance": S[p]["metrics"]["unsafe_compliance"]["rate"]} for p in POL}).T
    '''),
    md("## 8. Facts for RQ1 / RQ2"),
    code(r'''
    for p in POL:
        m = S[p]["metrics"]
        print(f"{p.upper():5s} safe-answer {m['safe_answer']['rate']:.3f} | over-refusal {m['over_refusal']['rate']:.3f} | any refusal on SAFE {m['safe_any_refusal']['rate']:.3f} | "
              f"justified refusal {m['justified_refusal']['rate']:.3f} | unsafe compliance {m['unsafe_compliance']['rate']:.3f} | ambiguous {m['ambiguous']['rate']:.3f}")
    top = pd.DataFrame({p: {c: S[p]["by_category"][c]["refusal_rate"] for c in cats if S[p]["by_category"][c]["class"] == "SAFE"} for p in POL})
    print("SAFE categories with the highest refusal-type rate (mean over policies):", top.mean(axis=1).sort_values(ascending=False).head(4).round(2).to_dict())
    '''),
]

NOTEBOOKS["task4_safety/notebooks/2_audit_and_calibration.ipynb"] = [
    md(r"# Task 4 · Step 3 — Manual audit of the AI judge" + "\n" + SAFETY_HEADER + r'''
    **Audit design.** The fixed 60-prompt subset (30 SAFE + 30 UNSAFE, sampled with seed 6304, `results/task4_safety/manual_audit_ids.csv`) is joined to each policy's response: 240 responses. The sheet `results/task4_safety/manual_audit_sheet.csv` is blind (shuffled rows, no policy name, no AI label); the mapping is in `manual_audit_key.csv`, which should stay closed while labelling. This notebook never prints which policy produced an audit row before the labels are filled.

    **Research question (RQ3).** What kinds of responses does the AI safety judge misclassify, and how much do those errors affect the policy comparison?
    '''),
    SETUP,
    *run_cell(["python -m task4_safety.make_audit_sheet --config configs/feedback.yaml",
               "python -m task4_safety.evaluate_safety --config configs/feedback.yaml"], "`make_audit_sheet` never overwrites a sheet that already contains labels."),
    md("## 1. The audited prompts"),
    code(r'''
    ids = pd.read_csv("results/task4_safety/manual_audit_ids.csv")
    X = pd.read_csv("data/xstest_safety_prompts.csv").set_index("xstest_id")
    A = X.loc[ids.xstest_id]
    display(pd.crosstab(A["type"], A["benchmark_class"], margins=True))
    sheet = pd.read_csv("results/task4_safety/manual_audit_sheet.csv")
    lab = sheet.manual_label.fillna("").astype(str).str.strip()
    print(f"audit sheet rows: {len(sheet)} (= {len(ids)} prompts × 4 policies); labelled so far: {(lab != '').sum()}")
    print("label counts:", lab[lab != ""].value_counts().to_dict())
    '''),
    md("## 2. Agreement between the manual labels and the AI judge"),
    code(r'''
    S = load("task4_safety/safety_evaluation_results.json"); AUD = S.get("_manual_audit")
    LABELS = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
    if not AUD:
        display(Markdown("> **Pending:** the audit sheet has no labels yet. Fill `manual_label` in `results/task4_safety/manual_audit_sheet.csv`, run "
                         "`python -m task4_safety.evaluate_safety --config configs/feedback.yaml`, then re-execute this notebook."))
    else:
        rows = {p.upper(): v for p, v in AUD["per_policy"].items() if v}; rows["pooled"] = AUD["pooled"]
        display(pd.DataFrame({k: {"n": v["n"], "agreement": v["agreement"], "95% CI": "[{:.2f}, {:.2f}]".format(*v["agreement_ci95"]), "Cohen κ": v["cohen_kappa"],
                                  "AI ambiguous": v["ai_ambiguous_rate"], "manual ambiguous": v["manual_ambiguous_rate"],
                                  "AI conf. when agree / disagree": f"{v['ai_confidence_when_agree']:.2f} / {v['ai_confidence_when_disagree']:.2f}"} for k, v in rows.items()}).T)
        C = np.array([[AUD["pooled"]["confusion_manual_rows_ai_cols"][m][a] for a in LABELS] for m in LABELS])
        fig, ax = plt.subplots(figsize=(6.5, 5.2)); ax.imshow(C, cmap="Blues"); ax.grid(False)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, C[i, j], ha="center", va="center", color="white" if C[i, j] > C.max() / 2 else "black")
        short = ["SAFE_ANS", "JUST_REF", "UNSAFE_C", "OVER_REF", "AMBIG"]
        ax.set_xticks(range(5)); ax.set_xticklabels(short, rotation=30); ax.set_yticks(range(5)); ax.set_yticklabels(short)
        ax.set_xlabel("AI judge"); ax.set_ylabel("manual"); ax.set_title("pooled confusion (240 responses)"); plt.show()
    '''),
    md("## 3. Does judge error change the policy comparison? (audited subset: AI labels vs manual labels)"),
    code(r'''
    if AUD:
        rs = AUD["audit_subset_rates"]
        display(pd.DataFrame({(p.upper(), src): rs[p][src] for p in rs for src in ["ai", "manual"]}).T.round(3))
    else:
        print("pending manual labels")
    '''),
    md("## 4. Disagreement examples (judge errors or policy subtleties)"),
    code(r'''
    if AUD:
        for d in AUD["disagreements"][:8]:
            side_by_side(f"{d['policy'].upper()} · [{d['benchmark_class']} · {d['type']}] manual = {d['manual']}, AI = {d['ai']}", d["prompt"],
                         {"response": d["response"]}, meta={"response": f"AI confidence {d['ai_confidence']:.2f}; note: {d['notes']}"}, limit=500)
    else:
        print("pending manual labels")
    '''),
]


# =============================================================================================
# TASK 5 — RLVR vs RLAIF
# =============================================================================================
FB_HEADER = r'''
**Feedback sources.** RLVR uses the exact final-answer checker, $r_{\text{RLVR}}(x,y)=\mathbb{1}[\text{verifier}(y)=\text{gold}(x)]$, where the verifier extracts the **last** `#### <number>` field (numbers elsewhere are ignored). RLAIF uses a direct group reward from the fixed pairwise judge: for the $K$ responses to one prompt, $r_{\text{RLAIF}}(y_k)=\frac{\text{wins}(y_k)+0.5\,\text{ties}(y_k)}{K-1}$. The judge (`Qwen2.5-3B-Instruct`, 4-bit, greedy, 4 new tokens) receives the rubric, the problem and two candidates in a deterministic hash-based A/B orientation and must answer A, B or TIE; the released parser keeps the first of A/B/TIE it finds (no match → TIE).

**Policies.** SFT = base `Qwen2.5-1.5B-Instruct`; RLVR and RLAIF = the supplied frozen adapters (same base, same GSM8K prompt indices, group optimiser, group size, completion cap and approximately matched token budget). All policies answer with greedy decoding, cap 512 tokens, using the exact course prompts.

**AI pairwise win rate.** win = 1, tie = 0.5, loss = 0, averaged over the fixed set; explicit ties and ambiguous judge outputs are reported separately.
'''

NOTEBOOKS["task5_feedback/notebooks/1_indomain_gsm8k.ipynb"] = [
    md(r"# Task 5 · Step 1 — In-domain GSM8K comparison" + "\n" + FB_HEADER + r'''
    **Research question (RQ1).** Where does AI preference feedback distinguish responses that the binary verifier treats identically, and when is that extra flexibility useful?
    '''),
    SETUP,
    *run_cell(["python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm",
               "python -m task5_feedback.judge_ambiguity --config configs/feedback.yaml"]),
    md("## 1. Verifier check on the manually validated diagnostic responses"),
    code(r'''
    from task5_feedback.rlvr import exact_reward, extract_designated_final
    diag = pd.DataFrame(jl_path := [json.loads(l) for l in open("data/task5_controlled_reward_diagnostics.jsonl", encoding="utf-8")])
    diag["verifier"] = [exact_reward(r, g) for r, g in zip(diag.response, diag.gold_final)]
    diag["expected"] = diag.expected_exact_reward.astype(int)
    print(f"verifier matches expected_exact_reward on {(diag.verifier == diag.expected).sum()}/{len(diag)} rows")
    pd.crosstab(diag.variant_type, diag.verifier, rownames=["variant"], colnames=["verifier reward"])
    '''),
    md("## 2. Per-policy results"),
    code(r'''
    POL = ["sft", "rlvr", "rlaif"]
    E = load("task5_feedback/math_eval_gsm.json"); Gn = {p: pd.DataFrame(jl(f"task5_feedback/generations_gsm_{p}.jsonl")) for p in POL}
    pd.DataFrame({p.upper(): {"exact accuracy": ci_text(E["per_policy"][p]["exact_accuracy"], E["per_policy"][p]["exact_accuracy_ci95"]),
                              "format compliance": E["per_policy"][p]["format_compliance_rate"], "accuracy | formatted": E["per_policy"][p]["accuracy_given_format"],
                              "length mean ± std": f"{E['per_policy'][p]['length_tokens']['mean']:.0f} ± {E['per_policy'][p]['length_tokens']['std']:.0f}",
                              "hit 512 cap": E["per_policy"][p]["hit_max_tokens_rate"], **{f"share: {k}": v for k, v in E["per_policy"][p]["failure_types"].items()}}
                  for p in POL}).T
    '''),
    code(r'''
    fig, ax = plt.subplots(1, 2, figsize=(15, 3.8))
    fts = ["correct", "wrong_final_answer", "no_final_format", "no_final_format_hit_cap"]; left = np.zeros(3)
    for t, c in zip(fts, ["#16A34A", "#DC2626", "#F59E0B", "#92400E"]):
        v = np.array([E["per_policy"][p]["failure_types"][t] for p in POL]) * 100
        ax[0].barh([p.upper() for p in POL], v, left=left, color=c, label=t); left += v
    ax[0].set_xlabel("% of 300 problems"); ax[0].legend(fontsize=7, loc="lower right"); ax[0].set_title("outcome under the released #### parser")
    for p, c in zip(POL, ["#6B7280", "#2563EB", "#9333EA"]):
        ax[1].hist(Gn[p].response_tokens, bins=30, histtype="step", lw=2, color=c, label=p.upper())
    ax[1].set_xlabel("response tokens (cap 512)"); ax[1].legend(); ax[1].set_title("response length"); fig.tight_layout(); plt.show()
    nf = Gn["sft"][Gn["sft"].failure_type == "no_final_format"]
    endings = nf.response.str.strip().str[-60:]
    kind = np.select([endings.str.contains(r"\\boxed"), endings.str.contains(r"\*\*"), endings.str.contains(r"####")], ["\\boxed{...}", "**bold**", "#### (malformed)"], "other")
    print("how SFT responses without a `#### <number>` field end:", pd.Series(kind).value_counts().to_dict())
    nf.head(4)[["gold", "response"]].assign(response=lambda d: d.response.str.strip().str[-160:]).rename(columns={"response": "response (last 160 chars)"})
    '''),
    md(r'''
    ### Supplementary (not the course metric): format-insensitive accuracy
    To separate *answer correctness* from *format compliance*, the last number in `\boxed{}`, in a final bold span, or else the last number of the response is compared with the gold answer. This is only a diagnostic of the verifier specification; every reported accuracy above uses the released `####` parser.
    '''),
    code(r'''
    def lenient(text):
        t = text.replace(",", "")
        for pat in [r"####\s*([-+]?\d+(?:\.\d+)?)", r"\\boxed\{\s*\$?([-+]?\d+(?:\.\d+)?)", r"\*\*\s*\$?([-+]?\d+(?:\.\d+)?)[^*]*\*\*(?!.*\d)"]:
            m = re.findall(pat, t)
            if m: return m[-1]
        m = re.findall(r"[-+]?\d+(?:\.\d+)?", t); return m[-1] if m else None
    ok = lambda a, g: a is not None and abs(float(a) - float(g)) < 1e-6
    pd.DataFrame({p.upper(): {"official (####) accuracy": Gn[p].exact_correct.mean(),
                              "format-insensitive accuracy (suppl.)": np.mean([ok(lenient(r), g) for r, g in zip(Gn[p].response, Gn[p].gold)])} for p in POL}).T.round(3)
    '''),
    code(r'''
    c = pd.DataFrame({p: Gn[p].exact_correct.values for p in POL})
    print("problems solved by all three:", int(c.all(axis=1).sum()), "| by none:", int((~c.any(axis=1)).sum()),
          "| RLVR-only:", int((c.rlvr & ~c.sft & ~c.rlaif).sum()), "| RLAIF-only:", int((c.rlaif & ~c.sft & ~c.rlvr).sum()), "| SFT-only:", int((c.sft & ~c.rlvr & ~c.rlaif).sum()))
    print("identical responses: RLVR vs SFT %.2f, RLAIF vs SFT %.2f" % ((Gn["rlvr"].response == Gn["sft"].response).mean(), (Gn["rlaif"].response == Gn["sft"].response).mean()))
    '''),
    md("## 3. Pairwise AI judge and verifier–judge agreement"),
    code(r'''
    PW = E["pairwise_comparisons"]; AMB = load("task5_feedback/judge_ambiguity.json") if exists("task5_feedback/judge_ambiguity.json") else {}
    pd.DataFrame({k.replace("_vs_", " vs ").upper(): {"win rate (tie = 0.5)": v["win_rate_a_ties_half"], "A wins": v["win_rate_a"], "ties": v["tie_rate"], "B wins": v["loss_rate_a"],
                  "decisive A/B outputs": AMB.get("gsm", {}).get(k, {}).get("decisive"), "explicit TIE outputs": AMB.get("gsm", {}).get(k, {}).get("explicit_tie"),
                  "ambiguous outputs (option echo)": AMB.get("gsm", {}).get(k, {}).get("ambiguous"), "verifier decisive pairs": v["n_verifier_decisive"],
                  "judge agrees when verifier decides": v["agreement_on_verifier_decisive"], "judge opposes verifier": v["judge_opposes_verifier_rate"],
                  "judge decisive when verifier ties": v["judge_decisive_when_verifier_tie"]} for k, v in PW.items()}).T
    '''),
    code(r'''
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for a, (k, v) in zip(axes, PW.items()):
        C = np.array([[v["verifier_judge_contingency"][vv][jj] for jj in ["A", "TIE", "B"]] for vv in ["A", "TIE", "B"]])
        a.imshow(C, cmap="Purples"); a.grid(False)
        for i in range(3):
            for j in range(3):
                a.text(j, i, C[i, j], ha="center", va="center", color="white" if C[i, j] > C.max() / 2 else "black")
        a.set_xticks(range(3)); a.set_xticklabels(["A", "TIE", "B"]); a.set_yticks(range(3)); a.set_yticklabels(["A", "TIE", "B"])
        a.set_xlabel("judge"); a.set_ylabel("verifier"); a.set_title(k.replace("_vs_", " vs ").upper() + " (A = first)")
    fig.tight_layout(); plt.show()
    if AMB: print("most frequent raw judge outputs (GSM8K):", AMB["gsm"]["all_raw_outputs"])
    '''),
    md("## 4. Examples where verifier and judge disagree"),
    code(r'''
    items = load("task5_feedback/math_pairwise_items_gsm.json")["rlaif_vs_sft"]
    byid = {p: Gn[p].set_index("id") for p in POL}
    shown = 0
    for it in items:
        if it["verifier"] != "TIE" and it["judge"] != "TIE" and it["judge"] != it["verifier"] and shown < 2:
            a, b = byid["rlaif"].loc[it["id"]], byid["sft"].loc[it["id"]]
            side_by_side(f"judge prefers {'RLAIF' if it['judge'] == 'A' else 'SFT'}, verifier prefers {'RLAIF' if it['verifier'] == 'A' else 'SFT'} (gold {a.gold})", a.problem,
                         {"RLAIF": a.response, "SFT": b.response}, meta={"RLAIF": f"correct {a.exact_correct} · pred {a.pred_final}", "SFT": f"correct {b.exact_correct} · pred {b.pred_final}"}, limit=600)
            shown += 1
    if not shown: print("no pair with opposite decisive preferences")
    '''),
    md("## 5. Facts for RQ1"),
    code(r'''
    v = PW["rlaif_vs_sft"]
    print(f"verifier decisive on {v['n_verifier_decisive']}/{v['n']} RLAIF-vs-SFT pairs; judge decisive on {round((1 - v['tie_rate']) * v['n'])}/{v['n']}")
    print(f"judge decisive when verifier ties: {v['judge_decisive_when_verifier_tie']:.3f}; judge prefers the longer response in those cases: {v['judge_prefers_longer_when_verifier_tie']}")
    print(f"judge cost: {E['judge_cost']['mean_seconds_per_call']:.2f} s per uncached call ({E['judge_cost']['uncached_calls']} calls)")
    '''),
]

NOTEBOOKS["task5_feedback/notebooks/2_controlled_diagnostics.ipynb"] = [
    md(r"# Task 5 · Step 2 — Controlled reward diagnostics" + "\n" + FB_HEADER + r'''
    **Diagnostic set.** 20 GSM8K problems × 5 manually validated variants: clean (correct), corrupted reasoning with correct final, sound reasoning with wrong final, correct + persuasive filler, gold number mentioned as a rejected distractor with wrong final. Each perturbation is compared with the clean response (the diagnostically better one). $S_{\text{reason}}=\Pr[R(y_{\text{clean}})>R(y_{\text{reason-corrupt}})]$, $S_{\text{outcome}}=\Pr[R(y_{\text{correct final}})>R(y_{\text{wrong final}})]$. Every judge comparison is also run with the candidates passed in the opposite order (order consistency), and each variant receives the RLAIF group reward among the five variants (K = 5).

    **Research questions.** (RQ2) When does the judge make an incorrect distinction because of style, verbosity or persuasive framing? (RQ3) Which reward source is more vulnerable to each diagnostic category?
    '''),
    SETUP,
    *run_cell(["python -m task5_feedback.score_perturbations --config configs/feedback.yaml",
               "python -m task5_feedback.judge_ambiguity --config configs/feedback.yaml"]),
    md("## 1. The five variants of one problem (endings)"),
    code(r'''
    diag = pd.DataFrame([json.loads(l) for l in open("data/task5_controlled_reward_diagnostics.jsonl", encoding="utf-8")])
    pid = diag.problem_id.iloc[0]; sub = diag[diag.problem_id == pid].set_index("variant_type")
    side_by_side(f"problem {pid} (gold {sub.gold_final.iloc[0]})", sub.question.iloc[0], {v: "… " + sub.loc[v, "response"][-260:] for v in sub.index}, limit=320)
    diag.groupby("variant_type").response.apply(lambda s: s.str.len().mean()).rename("mean characters").to_frame()
    '''),
    md("## 2. Better / tie / worse per perturbation"),
    code(r'''
    P = load("task5_feedback/perturbation_scores.json"); AMB = load("task5_feedback/judge_ambiguity.json").get("diagnostics", {}) if exists("task5_feedback/judge_ambiguity.json") else {}
    rows = []
    for c, v in P["comparisons"].items():
        for mech, lab in [("rlvr", "exact verifier"), ("rlaif", "AI judge")]:
            rows.append({"perturbation": c, "mechanism": lab, "prefers better": v[mech]["better_rate"], "tie": v[mech]["tie_rate"], "prefers worse": v[mech]["worse_rate"],
                         "order consistency": v["rlaif_order_consistency"] if mech == "rlaif" else 1.0,
                         "ambiguous judge outputs": AMB.get(c, {}).get("ambiguous") if mech == "rlaif" else None})
    display(pd.DataFrame(rows).set_index(["perturbation", "mechanism"]))
    print(f"S_reason: verifier {P['s_reason']['rlvr']:.2f}, judge {P['s_reason']['rlaif']:.2f}  |  S_outcome: verifier {P['s_outcome']['rlvr']:.2f}, judge {P['s_outcome']['rlaif']:.2f}  |  "
          f"S_outcome pooled with distractor: verifier {P['s_outcome_pooled_with_distractor']['rlvr']:.2f}, judge {P['s_outcome_pooled_with_distractor']['rlaif']:.2f}")
    print("verifier validation mismatches:", len(P["verifier_validation"]["mismatches"]))
    '''),
    code(r'''display(Image(filename=str(FIG / "task5_perturbation_diagnostics.png"), width=1100))'''),
    md("## 3. Conflict pair and group rewards"),
    code(r'''
    cp = P["conflict_pair"]
    print(f"{cp['first']} vs {cp['second']}:  verifier {cp['rlvr']}  |  judge {cp['rlaif']}")
    pd.DataFrame(P["group_rewards"]).rename(columns={"rlvr": "verifier reward", "rlaif": "RLAIF group reward (K=5)"}).round(3)
    '''),
    md("## 4. Cases where the judge prefers the worse response"),
    code(r'''
    items = pd.DataFrame(load("task5_feedback/perturbation_items.json"))
    VB = {c: (v["better_variant"], v["worse_variant"]) for c, v in P["comparisons"].items()}
    bad = items[items.judge == "worse"]
    print("judge 'prefers worse' counts by perturbation:", bad.comparison.value_counts().to_dict())
    for r in bad.head(3).itertuples():
        b, w = VB[r.comparison]; s = diag[diag.problem_id.astype(str) == str(r.problem_id)].set_index("variant_type")
        side_by_side(f"{r.comparison}: judge prefers the {w} response (reversed order: {r.judge_reversed})", s.question.iloc[0],
                     {f"better: {b}": "… " + s.loc[b, "response"][-300:], f"worse: {w}": "… " + s.loc[w, "response"][-300:]}, limit=340)
    '''),
    md("## 5. Facts for RQ2 / RQ3"),
    code(r'''
    for c, v in P["comparisons"].items():
        print(f"{c:22s} verifier better/tie/worse {v['rlvr']['better_rate']:.2f}/{v['rlvr']['tie_rate']:.2f}/{v['rlvr']['worse_rate']:.2f} | "
              f"judge {v['rlaif']['better_rate']:.2f}/{v['rlaif']['tie_rate']:.2f}/{v['rlaif']['worse_rate']:.2f} | order-consistent {v['rlaif_order_consistency']:.2f}")
    '''),
]

NOTEBOOKS["task5_feedback/notebooks/3_transfer_and_synthesis.ipynb"] = [
    md(r"# Task 5 · Step 3 — SVAMP transfer and feedback-source synthesis" + "\n" + FB_HEADER + r'''
    **Transfer set.** The fixed 100-example SVAMP challenge subset (`data/math_transfer_eval.jsonl`, first 100 in official order), evaluated without retraining with the same verifier and judge protocol.

    **Research question (RQ4).** Which behaviours transfer to the out-of-domain set, and what evidence supports or weakens the claim that either feedback source promotes more general reasoning behaviour?
    '''),
    SETUP,
    *run_cell(["python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer",
               "python -m task5_feedback.compare_feedback --config configs/feedback.yaml",
               "python -m task5_feedback.judge_ambiguity --config configs/feedback.yaml"]),
    md("## 1. SVAMP results and drop from GSM8K"),
    code(r'''
    POL = ["sft", "rlvr", "rlaif"]
    T = load("task5_feedback/math_eval_transfer.json"); Gk = load("task5_feedback/math_eval_gsm.json"); F = load("task5_feedback/feedback_synthesis.json")
    pd.DataFrame({p.upper(): {"GSM8K accuracy": Gk["per_policy"][p]["exact_accuracy"], "SVAMP accuracy": ci_text(T["per_policy"][p]["exact_accuracy"], T["per_policy"][p]["exact_accuracy_ci95"]),
                              "drop (GSM − SVAMP)": F["drops"][p]["accuracy_drop"], "GSM format": F["drops"][p]["gsm_format"], "SVAMP format": F["drops"][p]["transfer_format"],
                              "GSM length": F["drops"][p]["gsm_len"], "SVAMP length": F["drops"][p]["transfer_len"],
                              **{f"SVAMP {k}": v for k, v in T["per_policy"][p]["failure_types"].items()}} for p in POL}).T
    '''),
    code(r'''
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.8)); x = np.arange(3)
    for off, src, lab, c in [(-0.2, Gk, "GSM8K", "#2563EB"), (0.2, T, "SVAMP", "#F59E0B")]:
        acc = np.array([src["per_policy"][p]["exact_accuracy"] for p in POL]) * 100; ci = np.array([src["per_policy"][p]["exact_accuracy_ci95"] for p in POL]) * 100
        ax[0].bar(x + off, acc, 0.4, yerr=[np.clip(acc - ci[:, 0], 0, None), np.clip(ci[:, 1] - acc, 0, None)], capsize=3, color=c, label=lab)
        ax[1].bar(x + off, [src["per_policy"][p]["format_compliance_rate"] * 100 for p in POL], 0.4, color=c, label=lab)
    for a, t in zip(ax, ["exact accuracy (95% CI)", "format compliance"]):
        a.set_xticks(x); a.set_xticklabels([p.upper() for p in POL]); a.set_ylabel("%"); a.set_title(t); a.legend()
    fig.tight_layout(); plt.show()
    Tq = pd.DataFrame([json.loads(l) for l in open("data/math_transfer_eval.jsonl", encoding="utf-8")])
    Gt = {p: pd.DataFrame(jl(f"task5_feedback/generations_transfer_{p}.jsonl")) for p in POL}
    by = pd.DataFrame({p.upper(): pd.Series(Gt[p].exact_correct.values, index=Tq.svamp_type.values).groupby(level=0).mean() for p in POL})
    by["n"] = Tq.svamp_type.value_counts(); by.round(3)
    '''),
    md("## 2. Pairwise judge on SVAMP"),
    code(r'''
    AMB = load("task5_feedback/judge_ambiguity.json") if exists("task5_feedback/judge_ambiguity.json") else {}
    pd.DataFrame({k.replace("_vs_", " vs ").upper(): {"SVAMP win rate (tie = 0.5)": v["win_rate_a_ties_half"], "GSM8K win rate": Gk["pairwise_comparisons"][k]["win_rate_a_ties_half"],
                  "decisive A/B outputs": AMB.get("transfer", {}).get(k, {}).get("decisive"), "explicit TIE": AMB.get("transfer", {}).get(k, {}).get("explicit_tie"),
                  "ambiguous (option echo)": AMB.get("transfer", {}).get(k, {}).get("ambiguous")} for k, v in T["pairwise_comparisons"].items()}).T
    '''),
    md("## 3. Feedback-source comparison (all measured)"),
    code(r'''
    pd.DataFrame(F["feedback_source_table"]).T.rename(columns={"verifier": "exact verifier", "ai_judge": "AI judge"})
    '''),
    code(r'''display(Image(filename=str(FIG / "task5_feedback_synthesis_dashboard.png"), width=1100))'''),
    md("## 4. SVAMP examples"),
    code(r'''
    for p in ["rlvr", "rlaif"]:
        g = Gt[p]; w = g[~g.exact_correct].iloc[0]
        side_by_side(f"{p.upper()} — incorrect on SVAMP (gold {w.gold}, failure: {w.failure_type})", w.problem, {"response": w.response},
                     meta={"response": f"pred {w.pred_final} · {w.response_tokens} tok"}, limit=600)
    '''),
    md("## 5. Facts for RQ4"),
    code(r'''
    for p in POL:
        d = F["drops"][p]
        print(f"{p.upper():5s} GSM {d['gsm_accuracy']:.3f} → SVAMP {d['transfer_accuracy']:.3f} (drop {d['accuracy_drop']:+.3f}); format {d['gsm_format']:.2f} → {d['transfer_format']:.2f}; length {d['gsm_len']:.0f} → {d['transfer_len']:.0f}")
    for k in ["rlvr_vs_sft", "rlaif_vs_sft"]:
        print(f"{k}: win rate GSM {F['drops'][k]['gsm_win_rate']:.3f} → SVAMP {F['drops'][k]['transfer_win_rate']:.3f}; SVAMP decisive judge outputs {AMB.get('transfer', {}).get(k, {}).get('decisive')}")
    '''),
]


# =============================================================================================
# TASK 6 — SYNTHESIS
# =============================================================================================
NOTEBOOKS["task6_synthesis/notebooks/1_cross_task_synthesis.ipynb"] = [
    md(r'''
    # Task 6 — Cross-task synthesis (no new training)
    Collects the evidence of Tasks 1–5 along the manual's five synthesis axes: preference strength vs policy drift; offline vs online feedback; optimisation bias; safety calibration; reward-source design. Every number is read from the saved results; the report tables (`report/tables/*.csv|tex`) are rebuilt by `python -m common.visualize_all`, and auto-selected qualitative candidates are in `report/qualitative_candidates.md`.

    Cross-method comparisons are observational: DPO is evaluated on prompts from the UltraFeedback held-out pairs, PPO/GRPO on the held-out RL prompt pool with different generation caps, and the methods differ in more than one training detail.
    '''),
    SETUP,
    *run_cell(["python -m common.visualize_all", "python -m scripts.extract_qualitative"]),
    md("## 1. Drift, reward change and length change for every trained condition"),
    code(r'''
    rows = []
    def add(method, cond, e, base):
        rows.append({"method": method, "condition": cond, "KL token": e["kl_token_mean"], "KL seq": e["kl_sequence_mean"],
                     "Δ RM vs start": e["mean_reward"] - base["mean_reward"], "RM s.e.m.": e["sem_reward"],
                     "Δ length vs start": e["length_tokens"]["mean"] - base["length_tokens"]["mean"], "entropy": e["entropy_exact"]})
    sft = load("task1_dpo/dpo_eval_sft_reference.json")
    for cond, f in [("standard (1 epoch)", "standard"), ("β=0.03", "ablation_beta_0_03"), ("β=0.10", "ablation_beta_0_10"), ("β=0.30", "ablation_beta_0_30"), ("length-balanced", "length_balanced")]:
        add("DPO", cond, load(f"task1_dpo/dpo_eval_{f}.json"), sft)
    mid = load("task2_ppo/ppo_eval_midpoint.json")
    for f in ["standard", "fork_eps0_05_kl0_10", "fork_eps0_20_kl0_10", "fork_eps0_50_kl0_10", "fork_eps0_20_kl0_00", "fork_eps0_20_kl0_20"]:
        if exists(f"task2_ppo/ppo_eval_{f}.json"): add("PPO", f, load(f"task2_ppo/ppo_eval_{f}.json"), mid)
    gmid = load("task3_grpo/grpo_eval_midpoint.json")
    for f in ["standard", "fork_norm_grpo", "fork_norm_dr_grpo"]:
        add("GRPO", f, load(f"task3_grpo/grpo_eval_{f}.json"), gmid)
    D = pd.DataFrame(rows); D.set_index(["method", "condition"]).round(4)
    '''),
    code(r'''
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
    for m, c in [("DPO", "#2563EB"), ("PPO", "#DC2626"), ("GRPO", "#16A34A")]:
        s = D[D.method == m]
        ax[0].errorbar(s["KL token"], s["Δ RM vs start"], yerr=s["RM s.e.m."], fmt="o", color=c, label=m, capsize=3)
        ax[1].scatter(s["KL token"], s["Δ length vs start"], color=c, label=m)
        for r in s.itertuples():
            ax[0].annotate(r.condition, (r[3], r[5]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for a, t in zip(ax, ["Δ RM score vs own start (± s.e.m.)", "Δ generated length vs own start"]):
        a.axhline(0, color="k", lw=0.6); a.set_xlabel("held-out sampled KL to reference (token mean)"); a.set_title(t); a.legend()
    fig.tight_layout(); plt.show()
    '''),
    md("## 2. Offline vs online: what the training signal saw"),
    code(r'''
    pp = load("task1_dpo/dpo_eval_standard.json")["per_pair"]
    ppo = pd.DataFrame(jl("task2_ppo/ppo_train_standard.jsonl")); grpo = pd.DataFrame(jl("task3_grpo/grpo_train_standard.jsonl"))
    pd.DataFrame({
        "DPO (standard)": {"training samples": "1443 fixed pairs (offline)", "updates": 91, "drift measure": f"KL {load('task1_dpo/dpo_eval_standard.json')['kl_token_mean']:.5f}",
                           "signal on dataset responses": f"mean log-ratio chosen {np.mean([p['logratio_chosen'] for p in pp]):+.3f}, rejected {np.mean([p['logratio_rejected'] for p in pp]):+.3f}"},
        "PPO (standard)": {"training samples": f"{len(ppo)} on-policy rollouts", "updates": len(ppo), "drift measure": f"KL {load('task2_ppo/ppo_eval_standard.json')['kl_token_mean']:.5f}",
                           "signal on dataset responses": "— (online reward)"},
        "GRPO (standard)": {"training samples": f"{4 * len(grpo)} on-policy completions", "updates": len(grpo), "drift measure": f"KL {load('task3_grpo/grpo_eval_standard.json')['kl_token_mean']:.5f}",
                            "signal on dataset responses": "— (online group reward)"}}).T
    '''),
    md("## 3. Report tables"),
    code(r'''
    for t in ["t1_dpo_summary", "t2_ppo_heldout", "t3_grpo_heldout", "t4_safety", "t5_math", "t5_judge_outputs", "t_compute"]:
        f = ROOT / "report" / "tables" / f"{t}.csv"
        if f.exists():
            display(Markdown(f"**{t}**")); display(pd.read_csv(f))
    '''),
    md("## 4. Safety calibration vs reward"),
    code(r'''
    S = load("task4_safety/safety_evaluation_results.json")
    pd.DataFrame({p.upper(): {"safe answer": S[p]["metrics"]["safe_answer"]["rate"], "any refusal on SAFE": S[p]["metrics"]["safe_any_refusal"]["rate"],
                              "unsafe compliance": S[p]["metrics"]["unsafe_compliance"]["rate"], "justified refusal": S[p]["metrics"]["justified_refusal"]["rate"]}
                  for p in ["sft", "dpo", "ppo", "grpo"]}).T.round(3)
    '''),
    code(r'''display(Image(filename=str(FIG / "task6_cross_task_synthesis.png"), width=1100))'''),
    md("## 5. Qualitative candidates (auto-selected; read the full texts before using any)"),
    code(r'''
    q = (ROOT / "report" / "qualitative_candidates.md").read_text(encoding="utf-8")
    display(Markdown(q[:12000] + ("\n\n… (truncated; full file: report/qualitative_candidates.md)" if len(q) > 12000 else "")))
    '''),
]


def main():
    for rel, cells in NOTEBOOKS.items():
        stem = Path(rel).stem.replace("_", "-")[:20]
        cells = [dict(c, id=f"{stem}-{i:02d}") for i, c in enumerate(cells)]
        nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", rel)


if __name__ == "__main__":
    main()
