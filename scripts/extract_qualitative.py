"""Collect CANDIDATE qualitative examples for every task from saved results (no model calls).

Writes report/qualitative_candidates.md. Selection rules are mechanical (largest reward gains with large
length growth, judge-verifier disagreements, ...); whether an example really shows a quality failure
is for the reader to decide after reading the full text in the referenced result file.
Run: python -m scripts.extract_qualitative
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report" / "qualitative_candidates.md"


def load(rel):
    p = R / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def jl(rel):
    p = R / rel
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def cut(s, n=350):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " …"


def paired_generations(a_file, b_file, a_name, b_name, k=5, title=""):
    a, b = load(a_file), load(b_file)
    if not a or not b:
        return []
    ga = {g["prompt_id"]: g for g in a["generations"]}
    pairs = [(ga[g["prompt_id"]], g) for g in b["generations"] if g["prompt_id"] in ga]
    lines = [f"\n#### {title}\n_source: `{a_file}` vs `{b_file}` (same prompt, same sampling seed)_\n"]
    # Reward up with large length growth (possible reward/length exploitation)
    cand = sorted(pairs, key=lambda p: (p[1]["reward_score"] - p[0]["reward_score"]) *
                  (p[1]["response_length_tokens"] / max(p[0]["response_length_tokens"], 1)), reverse=True)[:k]
    lines.append(f"**RM increased while the response grew ({b_name} vs {a_name})**\n")
    for x, y in cand:
        lines.append(f"- `{y['prompt_id'][:10]}` RM {x['reward_score']:+.2f} → {y['reward_score']:+.2f}, "
                     f"len {x['response_length_tokens']} → {y['response_length_tokens']} tok, EOS {x['terminated_with_eos']}→{y['terminated_with_eos']}\n"
                     f"  - prompt: {cut(y['prompt'], 200)}\n  - {a_name}: {cut(x['response'])}\n  - {b_name}: {cut(y['response'])}")
    cand = sorted(pairs, key=lambda p: p[1]["reward_score"] - p[0]["reward_score"])[:3]
    lines.append(f"\n**Largest RM drops ({b_name} vs {a_name})**\n")
    for x, y in cand:
        lines.append(f"- `{y['prompt_id'][:10]}` RM {x['reward_score']:+.2f} → {y['reward_score']:+.2f}; len {x['response_length_tokens']} → {y['response_length_tokens']}\n"
                     f"  - {a_name}: {cut(x['response'])}\n  - {b_name}: {cut(y['response'])}")
    capped = [y for _, y in pairs if y["hit_max_tokens"]]
    if capped:
        y = max(capped, key=lambda g: g["reward_score"])
        lines.append(f"\n**Highest-RM response that never terminated ({b_name})**: RM {y['reward_score']:+.2f}, {y['response_length_tokens']} tok — {cut(y['response'], 500)}")
    return lines


def task1():
    out = ["\n## Task 1 — DPO"]
    out += paired_generations("task1_dpo/dpo_eval_sft_reference.json", "task1_dpo/dpo_eval_standard.json", "SFT", "DPO",
                              title="(i) preference/reward vs quality")
    la = load("task1_dpo/dpo_length_analysis.json")
    if la:
        out.append("\n#### (ii) Word-limit prompts (greedy)\n")
        rows = {k: la[k]["word_limit"]["details"] for k in ["sft", "standard", "length_balanced"] if k in la}
        for i, r in enumerate(rows["sft"]):
            out.append(f"- **{r['prompt']}** (limit {r['limit']})")
            for k, det in rows.items():
                d = det[i]
                out.append(f"  - {k}: {d['word_count']} words {'✓' if d['compliant'] else '✗'} — {cut(d['response'], 220)}")
        std = la.get("standard", {}).get("per_pair", [])
        if std:
            t = sorted([p for p in std if p["any_response_truncated"]], key=lambda p: -abs(p["margin"]))[:3]
            if t:
                out.append("\n**Held-out pairs whose response was truncated at 768 tokens (largest |margin|)**")
                for p in t:
                    out.append(f"- `{p['prompt_id'][:10]}` stratum={p['stratum']} margin={p['margin']:+.2f} "
                               f"chosen {p['chosen_tokens_full']}→{p['chosen_tokens_kept']} tok, rejected {p['rejected_tokens_full']}→{p['rejected_tokens_kept']} tok")
    return out


def task2():
    out = ["\n## Task 2 — PPO"]
    out += paired_generations("task2_ppo/ppo_eval_midpoint.json", "task2_ppo/ppo_eval_standard.json", "midpoint", "PPO-20",
                              title="Reward and quality moving together / apart (standard continuation)")
    out += paired_generations("task2_ppo/ppo_eval_fork_eps0_20_kl0_10.json", "task2_ppo/ppo_eval_fork_eps0_20_kl0_00.json",
                              "β_KL=0.10", "β_KL=0", k=3, title="Weakened KL pressure (over-optimisation candidates)")
    return out


def task3():
    out = ["\n## Task 3 — GRPO"]
    roll = jl("task3_grpo/grpo_rollouts_standard.jsonl")
    if roll:
        by = {}
        for r in roll:
            by.setdefault(r["update"], []).append(r)
        groups = sorted(by.values(), key=lambda g: max(x["reward"] for x in g) - min(x["reward"] for x in g))
        for title, g in [("Least informative group (smallest reward spread)", groups[0]), ("Most informative group", groups[-1])]:
            out.append(f"\n**{title}** — update {g[0]['update']}, prompt: {cut(g[0]['prompt'], 200)}")
            for x in sorted(g, key=lambda x: -x["reward"]):
                out.append(f"- r={x['reward']:+.2f} A={x['advantage']:+.2f} len={x['length']} masked={x['masked']}: {cut(x['completion'], 220)}")
    out += paired_generations("task3_grpo/grpo_eval_fork_norm_grpo.json", "task3_grpo/grpo_eval_fork_norm_dr_grpo.json",
                              "canonical", "Dr.GRPO", k=3, title="Normalisation forks on the same held-out prompts")
    return out


def task4():
    out = ["\n## Task 4 — Safety"]
    dis = load("task4_safety/policy_label_disagreements.json") or []
    want = [("UNSAFE_COMPLIANCE", "harmful compliance"), ("OVER_REFUSAL", "exaggerated refusal"),
            ("JUSTIFIED_REFUSAL", "refusal (check class: SAFE prompt ⇒ judge thought it was unsafe)")]
    for lab, desc in want:
        ex = [d for d in dis if lab in d["labels"].values()][:3]
        out.append(f"\n**{desc}** — prompts where policies differ and one got `{lab}`")
        for d in ex:
            out.append(f"- [{d['benchmark_class']}/{d['type']}] {cut(d['prompt'], 160)}  labels={d['labels']}")
            for p, r in d["responses"].items():
                out.append(f"  - {p}: {cut(r, 200)}")
    s = load("task4_safety/safety_evaluation_results.json") or {}
    a = s.get("_manual_audit")
    if a:
        out.append("\n**Manual-vs-AI disagreements (judge errors or policy subtleties)**")
        for d in a["disagreements"][:8]:
            out.append(f"- {d['policy']} [{d['benchmark_class']}/{d['type']}] manual={d['manual']} AI={d['ai']} (conf {d['ai_confidence']:.2f}): "
                       f"{cut(d['prompt'], 120)} → {cut(d['response'], 200)}")
    return out


def task5():
    out = ["\n## Task 5 — RLVR vs RLAIF"]
    items = load("task5_feedback/perturbation_items.json") or []
    diag = {}
    p = ROOT / "data" / "task5_controlled_reward_diagnostics.jsonl"
    if p.exists():
        for l in p.open(encoding="utf-8"):
            r = json.loads(l)
            diag[(str(r["problem_id"]), r["variant_type"])] = r
    COMPARISONS = {"reasoning_sensitivity": ("clean_correct", "corrupt_reasoning_correct_final", ""),
                   "filler_susceptibility": ("clean_correct", "persuasive_filler_correct", ""),
                   "distractor_robustness": ("clean_correct", "gold_distractor_wrong_final", "")}
    for comp, why in [("reasoning_sensitivity", "reasoning-only corruption"), ("filler_susceptibility", "persuasive filler"),
                      ("distractor_robustness", "gold distractor / wrong final")]:
        ex = [i for i in items if i["comparison"] == comp and i["verifier"] != i["judge"]][:2]
        out.append(f"\n**{why}: verifier and judge disagree**")
        vb, vw, _ = COMPARISONS[comp]
        for i in ex:
            out.append(f"- problem {i['problem_id']}: verifier={i['verifier']} judge={i['judge']} (reversed order: {i['judge_reversed']})")
            if diag:
                out.append(f"  - better ({vb}): {cut(diag[(i['problem_id'], vb)]['response'][-300:], 300)}")
                out.append(f"  - worse ({vw}): {cut(diag[(i['problem_id'], vw)]['response'][-300:], 300)}")
    pw = load("task5_feedback/math_pairwise_items_gsm.json") or {}
    g = {r["id"]: r for r in jl("task5_feedback/generations_gsm_rlaif.jsonl")}
    s = {r["id"]: r for r in jl("task5_feedback/generations_gsm_sft.jsonl")}
    opp = [i for i in pw.get("rlaif_vs_sft", []) if i["verifier"] != "TIE" and i["judge"] not in (i["verifier"], "TIE")][:3]
    if opp:
        out.append("\n**GSM8K: judge prefers the response the verifier marks wrong (RLAIF vs SFT)**")
        for i in opp:
            out.append(f"- id {i['id']}: verifier={i['verifier']} judge={i['judge']}; RLAIF correct={i['a_correct']} SFT correct={i['b_correct']}\n"
                       f"  - RLAIF: {cut(g.get(i['id'], {}).get('response', ''), 250)}\n  - SFT: {cut(s.get(i['id'], {}).get('response', ''), 250)}")
    return out


def main():
    lines = ["# Qualitative candidates (auto-selected; verify by reading the full texts)\n"]
    for fn in (task1, task2, task3, task4, task5):
        try:
            lines += fn()
        except Exception as e:
            lines.append(f"\n_{fn.__name__} skipped: {e}_")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
