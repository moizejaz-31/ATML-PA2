from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import torch

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import set_seed
from common.models import clear_gpu, load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    print(f"Generating deterministic responses for policy '{policy_name}' ({len(df)} prompts)...")

    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg.get("safety_max_new_tokens", 256)),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })

    clear_gpu(model)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="+", default=["sft", "dpo", "ppo", "grpo"])
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Task 4: Generating Deterministic Responses on XSTest")
    print(f"Policies to evaluate: {args.policies}")
    print(f"Output directory:     {outdir}")
    print("=" * 65)

    for pol in args.policies:
        out_path = outdir / f"generated_{pol}.jsonl"
        print(f"\nProcessing policy '{pol}' -> {out_path}...")
        records = generate_for_policy(cfg, pol, batch_size=args.batch_size)
        write_jsonl(out_path, records)
        print(f"Wrote {len(records)} responses to {out_path}")

    print("\n[Done] All policy responses generated successfully.")


if __name__ == "__main__":
    main()
