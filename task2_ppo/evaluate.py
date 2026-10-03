from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.policy_eval import evaluate_rl_policy

NO_ADAPTER = {None, "", "none", "sft", "base"}


def evaluate_ppo(config_path: str, adapter: str | None, name: str = "standard"):
    """Common held-out protocol for every PPO condition: same fixed prompts, sampling config, seed and
    the frozen-evaluation cap `eval_max_response_length` (768)."""
    cfg = load_yaml(config_path)
    adapter = None if adapter in NO_ADAPTER else adapter
    return evaluate_rl_policy(
        cfg,
        adapter,
        name,
        repo_path(cfg["results_dir"]) / f"ppo_eval_{name}.json",
        repo_path("report/figures") / f"task2_ppo_eval_{name}.png",
        max_new_tokens=int(cfg.get("eval_max_response_length", 768)),
        title=f"Task 2 — PPO held-out evaluation: {name}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the SFT policy")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    evaluate_ppo(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()
