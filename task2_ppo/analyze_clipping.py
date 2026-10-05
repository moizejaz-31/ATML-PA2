"""Task 2 clipping study.

Part A (cached batch, no rollouts): the supplied 32-rollout cache stores, per response token, the
old-policy log-prob, the reference log-prob and the critic value, plus the terminal reward. We rebuild
each (prompt, response) token sequence, recompute advantages with the released KL shaping + GAE
(beta_KL and gamma/lambda from the config, normalised over the whole cached batch), and score the
current PPO midpoint on those tokens:
  * static geometry: ratio rho_t = pi_midpoint / pi_old on the cached tokens, and for every eps the
    clip fraction (rho outside [1-eps, 1+eps]), the affected-token fraction (clipping binds, i.e. the
    min() picks the clipped term and the token's gradient is zeroed) and the clipped vs. unclipped
    surrogate;
  * update probe: from the identical midpoint weights, `ppo_epochs` passes of per-rollout PPO steps
    over the fixed cached batch (released lr / grad clip), separately for every eps, logging the
    clip/affected fraction at each step and the post-probe geometry. This shows how quickly each eps
    starts constraining the update when the batch is held fixed.
Part B: matched short continuation forks (task2_ppo.forks) evaluated on the common held-out protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import load_json, save_json, set_seed
from common.metrics import masked_mean
from common.models import (
    clear_gpu,
    disable_dropout,
    load_lora_state,
    load_policy,
    load_tokenizer,
    lora_state,
    trainable_parameters,
)
from task2_ppo.forks import fork_name, run_or_load_fork
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)
    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def rebuild_batch(cfg, tokenizer, rows):
    """Token sequences for the cached rollouts, aligned to the cached per-token arrays."""
    pool = {}
    for key in ("rl_prompt_eval", "rl_prompt_train"):
        for r in read_jsonl(cfg["paths"][key]):
            pool.setdefault(str(r["prompt_id"]), r)
    eos = tokenizer.eos_token_id
    items, skipped = [], []
    for row in rows:
        pid = str(row.get("prompt_id"))
        if pid not in pool:
            skipped.append({"prompt_id": pid, "reason": "prompt not found"})
            continue
        rendered = tokenizer.apply_chat_template(prompt_messages(pool[pid]), tokenize=False, add_generation_prompt=True)
        tokenizer.truncation_side = "left"
        p_ids = tokenizer(rendered, add_special_tokens=False, truncation=True,
                          max_length=int(cfg.get("max_prompt_length", 256)))["input_ids"]
        tokenizer.truncation_side = "right"
        r_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        n = len(row["old_logprobs"])
        if len(r_ids) + 1 == n:
            r_ids = r_ids + [eos]
        if len(r_ids) != n:
            skipped.append({"prompt_id": pid, "reason": f"token mismatch {len(r_ids)} vs {n}"})
            continue
        items.append({
            "prompt_id": pid,
            "prompt_ids": p_ids,
            "response_ids": r_ids,
            "old": torch.as_tensor(row["old_logprobs"], dtype=torch.float32),
            "ref": torch.as_tensor(row["ref_logprobs"], dtype=torch.float32),
            "values": torch.as_tensor(row["values"], dtype=torch.float32) if "values" in row else None,
            "reward": float(row.get("effective_terminal_reward", row.get("raw_terminal_reward", 0.0))),
        })
    return items, skipped


def cached_advantages(items, kl_beta, gamma, lam):
    """Released KL shaping + GAE per rollout, then one normalisation over the whole cached batch."""
    T = max(len(it["old"]) for it in items)
    n = len(items)
    old = torch.zeros(n, T)
    ref = torch.zeros(n, T)
    val = torch.zeros(n, T)
    mask = torch.zeros(n, T)
    for i, it in enumerate(items):
        L = len(it["old"])
        old[i, :L], ref[i, :L], mask[i, :L] = it["old"], it["ref"], 1.0
        if it["values"] is not None:
            val[i, :L] = it["values"]
    rew = torch.tensor([it["reward"] for it in items])
    shaped = shaped_rewards(rew, old, ref, mask, kl_beta)
    adv, _ = compute_gae(shaped, val, mask, gamma=gamma, lam=lam)
    adv = normalize_advantages(adv, mask)
    for i, it in enumerate(items):
        it["adv"] = adv[i, : len(it["old"])].clone()
    return items


def policy_logp(policy, it, device):
    seq = torch.tensor([it["prompt_ids"] + it["response_ids"]], device=device)
    resp = torch.tensor([it["response_ids"]], device=device)
    lp, _ = response_token_logprobs(policy, seq, torch.ones_like(seq), len(it["prompt_ids"]), resp)
    return lp[0]


def geometry(new_list, items, eps_values):
    """Pooled (token-weighted) clipping geometry over the cached batch."""
    logr = torch.cat([n.detach().cpu() - it["old"] for n, it in zip(new_list, items)])
    adv = torch.cat([it["adv"] for it in items])
    rho = torch.exp(logr)
    out = {"tokens": int(rho.numel()),
           "mean_abs_log_ratio": float(logr.abs().mean()),
           "log_ratio_percentiles": {q: float(np.percentile(logr.numpy(), q)) for q in (1, 5, 25, 50, 75, 95, 99)},
           "approx_kl_old_new": float((torch.exp(logr) - 1 - logr).mean())}
    for eps in eps_values:
        outside = ((rho < 1 - eps) | (rho > 1 + eps)).float()
        binds = (((rho > 1 + eps) & (adv > 0)) | ((rho < 1 - eps) & (adv < 0))).float()
        unclipped = rho * adv
        clipped = torch.minimum(unclipped, rho.clamp(1 - eps, 1 + eps) * adv)
        out[str(eps)] = {
            "clip_fraction": float(outside.mean()),
            "affected_token_fraction": float(binds.mean()),
            "clipped_surrogate": float(clipped.mean()),
            "unclipped_surrogate": float(unclipped.mean()),
            "surrogate_gap": float((unclipped - clipped).mean()),
        }
    return out, logr.numpy()


def update_probe(policy, items, eps, cfg, init_state, device, order):
    """`ppo_epochs` passes of per-rollout PPO steps on the fixed cached batch from the midpoint."""
    load_lora_state(policy, init_state)
    opt = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
    traj = []
    policy.train()
    for epoch in range(int(cfg.get("ppo_epochs", 2))):
        for i in order:
            it = items[i]
            new = policy_logp(policy, it, device)
            old = it["old"].to(device)
            adv = it["adv"].to(device)
            m = torch.ones_like(old)
            loss, rho, diag = ppo_policy_loss(new[None], old[None], adv[None], m[None], eps=eps)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg.get("max_grad_norm", 1.0))).item()
            if np.isfinite(gn) and torch.isfinite(loss):
                opt.step()
            traj.append({"epoch": epoch, "rollout": i, "clip_fraction": float(diag["clip_fraction"]),
                         "affected_fraction": float(diag["clip_high_fraction"] + diag["clip_low_fraction"]),
                         "loss": float(loss.item()), "grad_norm": gn})
    policy.eval()
    with torch.no_grad():
        new_list = [policy_logp(policy, it, device) for it in items]
    return traj, new_list


def plot_clipping(res: dict, fig_dir: Path, results_dir: Path):
    eps_values = res["eps_values"]
    colors = dict(zip([str(e) for e in eps_values], ["#DC2626", "#2563EB", "#16A34A"]))
    fig, axes = plt.subplots(1, 4, figsize=(24, 4.5))

    lr = np.asarray(res["cached_static"]["_log_ratio"])
    lim = max(0.6, float(np.percentile(np.abs(lr), 99.5)))
    axes[0].hist(np.clip(lr, -lim, lim), bins=80, color="#6B7280", alpha=0.85, label="midpoint vs cached old")
    if res.get("cached_probe"):
        for e in eps_values:
            pr = np.asarray(res["cached_probe"][str(e)]["_log_ratio"])
            axes[0].hist(np.clip(pr, -lim, lim), bins=80, histtype="step", lw=1.5, color=colors[str(e)], label=f"after probe ε={e}")
    for e in eps_values:
        axes[0].axvspan(np.log(1 - e), np.log(1 + e), color=colors[str(e)], alpha=0.06)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("log ρ_t on cached tokens")
    axes[0].set_title("(a) Ratio distribution; shaded = unclipped band")
    axes[0].legend(frameon=False, fontsize=7)

    x = np.arange(len(eps_values))
    st = res["cached_static"]
    vals = [("static clip", [st[str(e)]["clip_fraction"] for e in eps_values], "#9CA3AF"),
            ("static affected", [st[str(e)]["affected_token_fraction"] for e in eps_values], "#4B5563")]
    if res.get("cached_probe"):
        pb = res["cached_probe"]
        vals += [("post-probe clip", [pb[str(e)]["geometry"][str(e)]["clip_fraction"] for e in eps_values], "#93C5FD"),
                 ("post-probe affected", [pb[str(e)]["geometry"][str(e)]["affected_token_fraction"] for e in eps_values], "#1D4ED8")]
    w = 0.8 / len(vals)
    for j, (lab, v, c) in enumerate(vals):
        axes[1].bar(x + (j - (len(vals) - 1) / 2) * w, np.asarray(v) * 100, w, color=c, label=lab)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"ε={e}" for e in eps_values])
    axes[1].set_ylabel("% of response tokens")
    axes[1].set_title("(b) Cached-batch clip / affected-token fraction")
    axes[1].legend(frameon=False, fontsize=7)

    if res.get("cached_probe"):
        for e in eps_values:
            tr = res["cached_probe"][str(e)]["trajectory"]
            axes[2].plot(np.arange(1, len(tr) + 1), [t["affected_fraction"] * 100 for t in tr], color=colors[str(e)], label=f"ε={e}")
        axes[2].set_xlabel("probe step (one cached rollout per step)")
        axes[2].set_ylabel("% tokens where clipping binds")
        axes[2].set_title("(c) Update probe on the fixed cached batch")
        axes[2].legend(frameon=False)

    for e in eps_values:
        name = fork_name(e, res["kl_beta"])
        f = results_dir / f"ppo_train_{name}.jsonl"
        if f.exists():
            recs = [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
            axes[3].plot([r["update"] for r in recs], [r["approx_kl_old_new_last_epoch"] for r in recs], "-o", ms=3,
                         color=colors[str(e)], label=f"ε={e}")
    axes[3].set_xlabel("fork update")
    axes[3].set_ylabel("KL(π_old‖π_new) after one step (k3)")
    axes[3].set_title("(d) Fork step size (stability)")
    axes[3].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Task 2 — PPO clipping study")
    fig.tight_layout()
    fig.savefig(fig_dir / "task2_ppo_clipping_study.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {fig_dir / 'task2_ppo_clipping_study.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-forks", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--no-reuse", action="store_true", help="retrain forks even if they exist")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    eps_values = [float(e) for e in cfg.get("clip_values", [0.05, 0.20, 0.50])]
    kl_beta = float(cfg["kl_beta"])
    results_dir = repo_path(cfg["results_dir"])
    fig_dir = repo_path("report/figures")
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = load_cached_rollouts(cfg["cached_rollouts"])
    tokenizer = load_tokenizer(cfg["base_model"])
    items, skipped = rebuild_batch(cfg, tokenizer, rows)
    if not items:
        raise SystemExit(f"Could not rebuild any cached rollout: {skipped[:3]}")
    items = cached_advantages(items, kl_beta, float(cfg.get("gamma", 1.0)), float(cfg.get("gae_lambda", 0.95)))
    print(f"[cache] {len(rows)} cached rollouts, rebuilt {len(items)}, skipped {len(skipped)}")

    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    if bool(cfg.get("disable_dropout", True)):
        disable_dropout(policy)
    device = next(policy.parameters()).device
    policy.eval()
    with torch.no_grad():
        mid = [policy_logp(policy, it, device) for it in items]
    static, static_lr = geometry(mid, items, eps_values)
    static["_log_ratio"] = static_lr.tolist()
    print(f"[cache] midpoint vs cached old: mean|log ρ|={static['mean_abs_log_ratio']:.4f}")
    for e in eps_values:
        s = static[str(e)]
        print(f"   ε={e}: clip={s['clip_fraction']:.3%} affected={s['affected_token_fraction']:.3%} "
              f"surrogate clipped={s['clipped_surrogate']:+.4f} unclipped={s['unclipped_surrogate']:+.4f}")

    probe = {}
    if not args.skip_probe:
        init = lora_state(policy)
        order = list(np.random.default_rng(int(cfg["seed"])).permutation(len(items)))
        for e in eps_values:
            traj, new_list = update_probe(policy, items, e, cfg, init, device, order)
            geo, lr = geometry(new_list, items, eps_values)
            probe[str(e)] = {"trajectory": traj, "geometry": geo, "_log_ratio": lr.tolist()}
            g = geo[str(e)]
            print(f"[probe ε={e}] after {len(traj)} steps: clip={g['clip_fraction']:.3%} affected={g['affected_token_fraction']:.3%} "
                  f"KL(old||new)={geo['approx_kl_old_new']:.5f}")
        load_lora_state(policy, init)
    clear_gpu(policy)
    del policy

    forks = {}
    if not args.skip_forks:
        for e in eps_values:
            forks[str(e)] = run_or_load_fork(args.config, e, kl_beta, reuse=not args.no_reuse)

    res = {
        "eps_values": eps_values,
        "kl_beta": kl_beta,
        "cache": {"rollouts": len(rows), "rebuilt": len(items), "skipped": skipped,
                  "prompt_ids": [it["prompt_id"] for it in items]},
        "cached_static": static,
        "cached_probe": probe,
        "forks": forks,
        # Backwards-compatible view.
        "cached_geometry": {str(e): {"clip_epsilon": e, **static[str(e)]} for e in eps_values},
        "fork_results": forks,
    }
    try:
        plot_clipping(res, fig_dir, results_dir)
    except Exception as ex:
        print(f"[Clipping Study] Warning: plotting failed: {ex}")
    slim = dict(res)
    slim["cached_static"] = {k: v for k, v in static.items() if not k.startswith("_")}
    slim["cached_probe"] = {e: {k: v for k, v in p.items() if not k.startswith("_")} for e, p in probe.items()}
    save_json(results_dir / "ppo_clipping_study_results.json", slim)
    np.savez_compressed(results_dir / "ppo_clipping_log_ratios.npz", static=static_lr,
                        **{f"probe_{e}".replace(".", "_"): np.asarray(p["_log_ratio"]) for e, p in probe.items()})

    print("\n" + "=" * 100)
    print(f"{'ε':>5} | {'static clip':>11} | {'static aff.':>11} | {'probe aff.':>10} | {'fork RM':>8} | {'fork KL':>8} | {'len':>6} | {'max KL(old,new)':>15}")
    for e in eps_values:
        f = forks.get(str(e), {})
        ev, tr = f.get("eval", {}), f.get("train", {})
        pa = probe.get(str(e), {}).get("geometry", {}).get(str(e), {}).get("affected_token_fraction", float("nan"))
        print(f"{e:5.2f} | {static[str(e)]['clip_fraction']:11.3%} | {static[str(e)]['affected_token_fraction']:11.3%} | {pa:10.3%} | "
              f"{ev.get('mean_reward', float('nan')):8.3f} | {ev.get('kl_token_mean', float('nan')):8.4f} | "
              f"{ev.get('length_tokens', {}).get('mean', float('nan')):6.1f} | {tr.get('stability_max_approx_kl_old_new', float('nan')):15.5f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
