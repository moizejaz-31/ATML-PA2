from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.policy_eval import evaluate_rl_policy

NO_ADAPTER = {None, "", "none", "sft", "base"}


def evaluate_grpo(config_path: str, adapter: str | None, name: str = "standard"):
    """Common held-out protocol for every GRPO condition (fixed prompts, sampling config, seed and the
    GRPO completion cap `max_completion_length`)."""
    cfg = load_yaml(config_path)
    adapter = None if adapter in NO_ADAPTER else adapter
    return evaluate_rl_policy(
        cfg,
        adapter,
        name,
        repo_path(cfg["results_dir"]) / f"grpo_eval_{name}.json",
        repo_path("report/figures") / f"task3_grpo_eval_{name}.png",
        max_new_tokens=int(cfg.get("max_completion_length", 512)),
        title=f"Task 3 — GRPO held-out evaluation: {name}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the SFT policy")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    evaluate_grpo(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()
