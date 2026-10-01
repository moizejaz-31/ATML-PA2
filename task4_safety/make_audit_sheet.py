from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


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
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        # Fall back to raw XSTest dataset if generation has not been run yet
        xstest_path = repo_path(cfg["paths"]["xstest"])
        if xstest_path.exists():
            rows = pd.read_csv(xstest_path).to_dict(orient="records")
        else:
            raise FileNotFoundError(f"Neither {src} nor {xstest_path} exists.")
    else:
        rows = read_jsonl(src)

    per_class = int(cfg.get("manual_audit_per_class", 30))
    seed = int(cfg.get("seed", 6304))
    ids = fixed_audit_ids(rows, per_class, seed)

    meta_df = pd.DataFrame(rows)
    audit_subset = meta_df[meta_df["xstest_id"].isin(ids)].copy()
    audit_subset = audit_subset.drop_duplicates(subset=["xstest_id"])
    audit_subset["manual_label"] = ""
    audit_subset["notes"] = ""

    # Keep essential columns for human review
    cols = ["xstest_id", "benchmark_class", "type", "prompt", "manual_label", "notes"]
    available_cols = [c for c in cols if c in audit_subset.columns]
    audit_subset = audit_subset[available_cols].sort_values("xstest_id")

    out_csv = outdir / "manual_audit_sheet.csv"
    audit_subset.to_csv(out_csv, index=False, encoding="utf-8")
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)

    print("=" * 65)
    print("Task 4: Manual Audit Sheet Generation")
    print(f"Generated {len(ids)} balanced audit prompts (30 SAFE, 30 UNSAFE)")
    print(f"Wrote audit sheet: {out_csv}")
    print(f"Wrote audit IDs:   {outdir / 'manual_audit_ids.csv'}")
    print("Labels to use: SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS")
    print("=" * 65)


if __name__ == "__main__":
    main()
