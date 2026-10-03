"""Task 5 in-domain (GSM8K) / out-of-domain (SVAMP transfer) evaluation of SFT, RLVR and RLAIF.

Per policy: exact final-answer accuracy (released `####` parser), format compliance, response length,
failure taxonomy. Pairwise (fixed course judge): RLAIF vs SFT (required), RLVR vs SFT and RLVR vs RLAIF;
win = 1, tie = 0.5, loss = 0, with explicit judge ties and unparseable judge outputs counted separately.
Verifier-judge agreement: on every pair the exact verifier also expresses a preference
(correct beats incorrect, otherwise tie), cross-tabulated against the judge's decision.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.metrics import length_stats, wilson_interval, word_count
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.judge_audit import AuditedPairwiseJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

POLICIES = ["sft", "rlvr", "rlaif"]
PAIRS = [("rlaif", "sft"), ("rlvr", "sft"), ("rlvr", "rlaif")]


def policy_specs(cfg):
    return {"sft": None, "rlvr": cfg["policies"]["rlvr"], "rlaif": cfg["policies"]["rlaif"]}


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    return load_policy(cfg, adapter_path=policy_specs(cfg)[name], trainable=False)


def prompt_of(r):
    # The fixed files carry the exact course prompt (with the `#### <number>` instruction) in `messages`.
    if isinstance(r.get("messages"), list):
        return r["messages"]
    q = str(r.get("question", r.get("problem", "")))
    return [{"role": "user", "content": f"{q}\n\nShow your reasoning and end your response with exactly `#### <number>`."}]


def failure_type(rec) -> str:
    if rec["exact_correct"]:
        return "correct"
    if rec["pred_final"] is not None:
        return "wrong_final_answer"
    return "no_final_format_hit_cap" if rec["hit_max_tokens"] else "no_final_format"


def generate_policy(cfg, rows, tokenizer, pol, out_file: Path, batch_size: int, max_tokens: int):
    if out_file.exists():
        recs = read_jsonl(out_file)
        if len(recs) == len(rows):
            print(f"[reuse] {out_file}")
            return recs
    set_seed(int(cfg["seed"]))
    model = load_frozen_policy(cfg, pol)
    recs = []
    t0 = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        gen = batch_generate(model, tokenizer, [prompt_of(r) for r in chunk], max_prompt_length=256,
                             max_new_tokens=max_tokens, temperature=0.0, do_sample=False)
        for r, resp, n_tok, hit in zip(chunk, gen["responses"], gen["response_lengths"], gen["truncated"]):
            gold = str(r.get("gold_final", r.get("answer", r.get("gold", ""))))
            rec = {"id": str(r.get("source_index", r.get("prompt_id"))), "problem": str(r.get("question", r.get("problem", ""))),
                   "gold": gold, "response": resp, "pred_final": extract_designated_final(resp),
                   "exact_correct": bool(exact_reward(resp, gold)), "response_tokens": int(n_tok),
                   "response_words": word_count(resp), "hit_max_tokens": bool(hit)}
            rec["failure_type"] = failure_type(rec)
            recs.append(rec)
        print(f"    {pol}: {len(recs)}/{len(rows)}", flush=True)
    clear_gpu(model)
    write_jsonl(out_file, recs)
    print(f"  generation time {pol}: {time.perf_counter() - t0:.0f}s")
    return recs


def policy_metrics(recs):
    k = sum(r["exact_correct"] for r in recs)
    n = len(recs)
    f = sum(r["pred_final"] is not None for r in recs)
    ft = {t: sum(r["failure_type"] == t for r in recs) / n
          for t in ["correct", "wrong_final_answer", "no_final_format", "no_final_format_hit_cap"]}
    return {"n": n, "exact_accuracy": k / n, "exact_accuracy_ci95": list(wilson_interval(k, n)),
            "format_compliance_rate": f / n, "accuracy_given_format": k / f if f else float("nan"),
            "length_tokens": length_stats(r["response_tokens"] for r in recs),
            "mean_tokens": float(np.mean([r["response_tokens"] for r in recs])),
            "mean_words": float(np.mean([r["response_words"] for r in recs])),
            "hit_max_tokens_rate": float(np.mean([r["hit_max_tokens"] for r in recs])),
            "failure_types": ft}


def pairwise(judge: AuditedPairwiseJudge, gens: dict, a: str, b: str):
    """Judge preference of policy a vs policy b on every problem, plus verifier agreement."""
    W = T = L = unparsed = 0
    cont = {v: {j: 0 for j in ["A", "B", "TIE"]} for v in ["A", "B", "TIE"]}
    items = []
    for ra, rb in zip(gens[a], gens[b]):
        pref = judge.compare(ra["problem"], ra["response"], rb["response"])
        un = judge.is_unparsed(ra["problem"], ra["response"], rb["response"])
        ver = "A" if ra["exact_correct"] > rb["exact_correct"] else "B" if rb["exact_correct"] > ra["exact_correct"] else "TIE"
        cont[ver][pref] += 1
        W += pref == "A"
        L += pref == "B"
        T += pref == "TIE"
        unparsed += bool(un) if pref == "TIE" else 0
        items.append({"id": ra["id"], "judge": pref, "verifier": ver, "judge_unparsed": un,
                      "a_correct": ra["exact_correct"], "b_correct": rb["exact_correct"],
                      "a_tokens": ra["response_tokens"], "b_tokens": rb["response_tokens"]})
    n = len(items)
    decisive = [i for i in items if i["verifier"] != "TIE"]
    v_tie = [i for i in items if i["verifier"] == "TIE"]
    return {
        "policy_a": a, "policy_b": b, "n": n,
        "win_rate_a": W / n, "loss_rate_a": L / n, "tie_rate": T / n,
        "win_rate_a_ties_half": (W + 0.5 * T) / n,
        "explicit_ties": T - unparsed, "unparsed_judge_outputs": unparsed,
        "verifier_judge_contingency": cont,
        "agreement_on_verifier_decisive": (sum(i["judge"] == i["verifier"] for i in decisive) / len(decisive)) if decisive else None,
        "judge_opposes_verifier_rate": (sum(i["judge"] not in (i["verifier"], "TIE") for i in decisive) / len(decisive)) if decisive else None,
        "n_verifier_decisive": len(decisive),
        "judge_decisive_when_verifier_tie": (sum(i["judge"] != "TIE" for i in v_tie) / len(v_tie)) if v_tie else None,
        "judge_prefers_longer_when_verifier_tie": (
            sum((i["judge"] == "A") == (i["a_tokens"] > i["b_tokens"]) for i in v_tie if i["judge"] != "TIE" and i["a_tokens"] != i["b_tokens"])
            / max(1, sum(1 for i in v_tie if i["judge"] != "TIE" and i["a_tokens"] != i["b_tokens"]))) if v_tie else None,
        "items": items,
    }


def plot_math_evaluation(results: dict, dataset: str, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = {"sft": "SFT", "rlvr": "RLVR", "rlaif": "RLAIF"}
    cols = {"sft": "#6B7280", "rlvr": "#2563EB", "rlaif": "#9333EA"}
    pp = results["per_policy"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    x = np.arange(len(POLICIES))
    acc = [pp[p]["exact_accuracy"] * 100 for p in POLICIES]
    ci = np.array([pp[p]["exact_accuracy_ci95"] for p in POLICIES]) * 100
    axes[0].bar(x - 0.2, acc, 0.4, yerr=[np.array(acc) - ci[:, 0], ci[:, 1] - np.array(acc)], capsize=4,
                color=[cols[p] for p in POLICIES], label="exact accuracy")
    axes[0].bar(x + 0.2, [pp[p]["format_compliance_rate"] * 100 for p in POLICIES], 0.4, color="#D1D5DB", label="format compliance")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[p] for p in POLICIES])
    axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=False)
    axes[0].set_title(f"(a) Accuracy / format ({dataset})")

    fts = ["correct", "wrong_final_answer", "no_final_format", "no_final_format_hit_cap"]
    fcols = ["#16A34A", "#DC2626", "#F59E0B", "#92400E"]
    left = np.zeros(len(POLICIES))
    for t, c in zip(fts, fcols):
        v = np.array([pp[p]["failure_types"][t] * 100 for p in POLICIES])
        axes[1].barh(x, v, left=left, color=c, label=t)
        left += v
    axes[1].set_yticks(x)
    axes[1].set_yticklabels([labels[p] for p in POLICIES])
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_title("(b) Outcome / failure types (%)")

    pw = results["pairwise_comparisons"]
    keys = list(pw)
    xx = np.arange(len(keys))
    win = np.array([pw[k]["win_rate_a"] for k in keys]) * 100
    tie = np.array([pw[k]["tie_rate"] for k in keys]) * 100
    loss = np.array([pw[k]["loss_rate_a"] for k in keys]) * 100
    axes[2].bar(xx, win, color="#16A34A", label="A wins")
    axes[2].bar(xx, tie, bottom=win, color="#D1D5DB", label="tie")
    axes[2].bar(xx, loss, bottom=win + tie, color="#DC2626", label="B wins")
    for i, k in enumerate(keys):
        axes[2].text(i, 101, f"{pw[k]['win_rate_a_ties_half']:.2f}", ha="center", fontsize=9)
    axes[2].set_xticks(xx)
    axes[2].set_xticklabels([k.replace("_vs_", " vs ").upper() for k in keys])
    axes[2].set_ylim(0, 110)
    axes[2].legend(frameon=False, fontsize=7)
    axes[2].set_title("(c) AI judge A-vs-B (number = win rate, tie=0.5)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Task 5 — {dataset.upper()} evaluation")
    fig.tight_layout()
    fig.savefig(fig_dir / f"task5_math_eval_{dataset}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def evaluate_dataset(config_path: str, dataset: str, batch_size: int = 8):
    cfg, rows, tokenizer = load_math_evaluation(config_path, dataset)
    max_tokens = int(cfg.get("math_max_new_tokens", 512))
    results_dir = repo_path(cfg["results_dir"]) / "task5_feedback"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Task 5 — {dataset}: {len(rows)} problems, policies {POLICIES}, greedy, cap {max_tokens}")

    gens = {p: generate_policy(cfg, rows, tokenizer, p, results_dir / f"generations_{dataset}_{p}.jsonl", batch_size, max_tokens)
            for p in POLICIES}
    metrics = {p: policy_metrics(gens[p]) for p in POLICIES}
    for p in POLICIES:
        m = metrics[p]
        print(f"  {p.upper():5s} acc={m['exact_accuracy']:.3f} format={m['format_compliance_rate']:.3f} "
              f"len={m['length_tokens']['mean']:.0f} cap={m['hit_max_tokens_rate']:.2f}")

    judge = AuditedPairwiseJudge(cfg, results_dir / f"judge_cache_{dataset}.json")
    t0 = time.perf_counter()
    pw = {}
    for a, b in PAIRS:
        pw[f"{a}_vs_{b}"] = pairwise(judge, gens, a, b)
        judge.flush()
        r = pw[f"{a}_vs_{b}"]
        print(f"  {a.upper()} vs {b.upper()}: win(tie=.5)={r['win_rate_a_ties_half']:.3f} W/T/L={r['win_rate_a']:.2f}/{r['tie_rate']:.2f}/"
              f"{r['loss_rate_a']:.2f} unparsed={r['unparsed_judge_outputs']} verifier-agree={r['agreement_on_verifier_decisive']}")
    cost = judge.cost()
    cost["judge_phase_seconds"] = time.perf_counter() - t0
    clear_gpu(judge.model)

    combined = {"dataset": dataset, "num_problems": len(rows), "per_policy": metrics,
                "pairwise_comparisons": {k: {kk: vv for kk, vv in v.items() if kk != "items"} for k, v in pw.items()},
                "judge_cost": cost,
                "protocol": {"decoding": "greedy", "max_new_tokens": max_tokens, "max_prompt_length": 256}}
    save_json(results_dir / f"math_eval_{dataset}.json", combined)
    save_json(results_dir / f"math_pairwise_items_{dataset}.json", {k: v["items"] for k, v in pw.items()})
    try:
        plot_math_evaluation(combined, dataset, repo_path("report/figures"))
    except Exception as e:
        print(f"[Math Eval] Warning: plotting failed: {e}")
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    evaluate_dataset(args.config, args.dataset, args.batch_size)


if __name__ == "__main__":
    main()
