"""Fixed 60-prompt manual audit (30 SAFE + 30 UNSAFE XSTest prompts, seed from the config).

Writes
  results/task4_safety/manual_audit_ids.csv       the fixed prompt IDs (released format)
  results/task4_safety/manual_audit_sheet.csv     BLIND sheet: one row per (prompt, policy) response,
                                                  rows shuffled, policy name and AI label hidden.
                                                  Fill `manual_label` with one of the five labels.
  results/task4_safety/manual_audit_key.csv       audit_uid -> policy / xstest_id / class / type
                                                  (do not open while labelling)

An existing sheet that already contains manual labels is never overwritten.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path

POLICIES = ["sft", "dpo", "ppo", "grpo"]
LABELS = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--force", action="store_true", help="overwrite a sheet even if it has labels")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)

    sheet_path = outdir / "manual_audit_sheet.csv"
    if sheet_path.exists() and not args.force:
        old = pd.read_csv(sheet_path)
        if "manual_label" in old and old["manual_label"].notna().any() and (old["manual_label"].astype(str).str.strip() != "").any():
            print(f"[keep] {sheet_path} already contains manual labels; not overwritten (use --force).")
            return

    rows = []
    for pol in POLICIES:
        f = outdir / f"generated_{pol}.jsonl"
        if not f.exists():
            print(f"[warn] {f} missing; audit sheet will not include {pol}")
            continue
        for r in read_jsonl(f):
            if int(r["xstest_id"]) in ids:
                rows.append({"policy": pol, "xstest_id": int(r["xstest_id"]), "benchmark_class": r["benchmark_class"],
                             "type": r["type"], "prompt": r["prompt"], "response": r["response"]})
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=int(cfg["seed"])).reset_index(drop=True)
    df.insert(0, "audit_uid", [f"A{i:03d}" for i in range(len(df))])
    df[["audit_uid", "policy", "xstest_id", "benchmark_class", "type"]].to_csv(outdir / "manual_audit_key.csv", index=False)
    blind = df[["audit_uid", "prompt", "response"]].copy()
    blind["manual_label"] = ""
    blind["notes"] = ""
    blind.to_csv(sheet_path, index=False, encoding="utf-8")

    print("=" * 70)
    print(f"Fixed audit prompts: {len(ids)} ({cfg['manual_audit_per_class']} SAFE + {cfg['manual_audit_per_class']} UNSAFE)")
    print(f"Blind sheet: {sheet_path}  ({len(blind)} responses = prompts x policies, shuffled)")
    print(f"Key (keep closed while labelling): {outdir / 'manual_audit_key.csv'}")
    print("Labels:", ", ".join(LABELS))
    print("Then re-run: python -m task4_safety.evaluate_safety --config configs/feedback.yaml")
    print("=" * 70)


if __name__ == "__main__":
    main()
