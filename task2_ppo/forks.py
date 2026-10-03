"""Matched short PPO forks from the identical supplied midpoint (policy + critic).

Every fork uses the same prompts (train-pool indices 0..fork_updates-1), seed, generation settings
and update budget; only the named control changes. The clipping study (eps sweep at the reference
beta_KL) and the KL study (beta_KL sweep at the reference eps) share one identical configuration
(eps=0.20, beta_KL=0.10); that fork is trained once and reused by both studies.
"""

from __future__ import annotations

from common.data import load_yaml, repo_path
from common.logging_utils import load_json
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_ppo


def fork_name(eps: float, kl: float) -> str:
    return f"fork_eps{eps:.2f}_kl{kl:.2f}".replace(".", "_")


def run_or_load_fork(config_path: str, eps: float, kl: float, reuse: bool = True) -> dict:
    cfg = load_yaml(config_path)
    name = fork_name(eps, kl)
    adapter = repo_path(f"outputs/task2_ppo/{name}")
    results_dir = repo_path(cfg["results_dir"])
    summary_file = results_dir / f"ppo_summary_{name}.json"
    eval_file = results_dir / f"ppo_eval_{name}.json"

    if reuse and (adapter / "adapter_config.json").exists() and summary_file.exists():
        print(f"[fork] reusing trained fork {name}")
        train = load_json(summary_file)
    else:
        print(f"\n>>> PPO fork {name}: eps={eps} beta_KL={kl} updates={cfg['fork_updates']}")
        train = run_ppo(config_path, output=str(adapter), updates=int(cfg["fork_updates"]),
                        clip_epsilon=eps, kl_beta=kl, run_name=name)
    if reuse and eval_file.exists():
        ev = load_json(eval_file)
    else:
        ev = evaluate_ppo(config_path, str(adapter), name=name)
    ev = {k: v for k, v in ev.items() if k != "generations"}
    return {"name": name, "clip_epsilon": eps, "kl_beta": kl, "train": train, "eval": ev}
