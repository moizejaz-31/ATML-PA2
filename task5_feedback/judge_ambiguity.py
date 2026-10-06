"""Offline audit of the pairwise judge's raw outputs (CPU only, no model calls).

The released parser maps the judge text to the FIRST of A/B/TIE it finds, so an output that just echoes
the options (e.g. "A, B" or "A,B,TIE") is silently counted as a preference for A (or B after the
orientation swap). This script rebuilds every comparison's cache key from the saved generations /
diagnostic responses, looks up the raw judge text (judge_cache_*_raw.json, written by
task5_feedback.judge_audit) and classifies each decision as:
  explicit_tie   - the judge answered exactly one label: TIE
  decisive       - exactly one label: A or B
  ambiguous      - several labels (option echo) or none (parse failure; released parser -> TIE)
Run: python -m task5_feedback.judge_ambiguity --config configs/feedback.yaml
Writes results/task5_feedback/judge_ambiguity.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

LABEL_RE = re.compile(r"\b(A|B|TIE)\b")
PAIRS = [("rlaif", "sft"), ("rlvr", "sft"), ("rlvr", "rlaif")]
DIAG = {
    "reasoning_sensitivity": ("clean_correct", "corrupt_reasoning_correct_final"),
    "outcome_sensitivity": ("clean_correct", "good_reasoning_wrong_final"),
    "filler_susceptibility": ("clean_correct", "persuasive_filler_correct"),
    "distractor_robustness": ("clean_correct", "gold_distractor_wrong_final"),
}


def key(model: str, problem: str, a: str, b: str) -> str:
    # identical to task5_feedback.rlaif.PairwiseAIJudge._key
    payload = json.dumps({"model": model, "problem": problem, "a": a, "b": b}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def classify(raw_entry) -> str:
    if raw_entry is None:
        return "no_raw_record"
    labels = set(LABEL_RE.findall(raw_entry["decoded"]))
    if len(labels) != 1:
        return "ambiguous"
    return "explicit_tie" if labels == {"TIE"} else "decisive"


def summarise(classes: list[str]) -> dict:
    c = Counter(classes)
    n = len(classes)
    return {"n": n, **{k: c.get(k, 0) for k in ["decisive", "explicit_tie", "ambiguous", "no_raw_record"]},
            "ambiguous_rate": c.get("ambiguous", 0) / n if n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    model = cfg["ai_judge_model"]
    rd = repo_path(cfg["results_dir"]) / "task5_feedback"
    out = {"rule": "ambiguous = judge text contains zero or several of {A, B, TIE}; the released parser keeps the first one"}

    for ds in ["gsm", "transfer"]:
        raw_f = rd / f"judge_cache_{ds}_raw.json"
        if not raw_f.exists():
            continue
        raw = json.loads(raw_f.read_text(encoding="utf-8"))
        gens = {}
        for p in ["sft", "rlvr", "rlaif"]:
            f = rd / f"generations_{ds}_{p}.jsonl"
            if f.exists():
                gens[p] = read_jsonl(f)
        res = {}
        for a, b in PAIRS:
            if a in gens and b in gens:
                cls = [classify(raw.get(key(model, ra["problem"], ra["response"], rb["response"])))
                       for ra, rb in zip(gens[a], gens[b])]
                res[f"{a}_vs_{b}"] = summarise(cls)
        res["all_raw_outputs"] = dict(Counter(v["decoded"] for v in raw.values()).most_common(12))
        out[ds] = res

    raw_f = rd / "judge_cache_diagnostics_raw.json"
    diag_path = repo_path(cfg["paths"]["task5_diagnostics"])
    if raw_f.exists() and diag_path.exists():
        raw = json.loads(raw_f.read_text(encoding="utf-8"))
        groups = defaultdict(dict)
        for r in read_jsonl(diag_path):
            groups[str(r["problem_id"])][r["variant_type"]] = r
        res = {}
        for name, (vb, vw) in DIAG.items():
            cls = []
            for vs in groups.values():
                q = vs[vb]["question"]
                cls.append(classify(raw.get(key(model, q, vs[vb]["response"], vs[vw]["response"]))))
            res[name] = summarise(cls)
        out["diagnostics"] = res
    elif not diag_path.exists():
        print(f"[skip diagnostics] {diag_path} missing (run python -m scripts.download_assets)")

    save_json(rd / "judge_ambiguity.json", out)
    for ds, res in out.items():
        if isinstance(res, dict):
            for k, v in res.items():
                if isinstance(v, dict) and "n" in v:
                    print(f"{ds:12s} {k:24s} n={v['n']:3d} decisive={v['decisive']:3d} explicit_tie={v['explicit_tie']:3d} "
                          f"ambiguous={v['ambiguous']:3d} missing={v['no_raw_record']}")


if __name__ == "__main__":
    main()
