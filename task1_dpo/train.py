from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, reset_file, save_json, set_seed, wall_timer
from common.models import (
    count_parameters,
    load_policy,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task1_dpo.dpo import dpo_loss
from task1_dpo.preprocess import load_filtered_pairs, pair_id


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def plot_training_curves(log_path: Path, fig_dir: Path, run_name: str):
    """Per-run training dashboard. Every logged value is a mean over one optimizer step's
    accumulation window (batch_size x grad_accum_steps pairs)."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [json.loads(l) for l in log_path.open(encoding="utf-8") if l.strip()]
    if not records:
        return
    steps = [r["step"] for r in records]
    fig_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        ("loss", "DPO loss"),
        ("preference_accuracy", "Train preference accuracy"),
        ("reward_margin_mean", "Implicit reward margin  β·Δlog-ratio"),
        ("kl_chosen", "log π/π_ref on chosen (seq. sum)"),
        ("grad_norm", "Grad norm (pre-clip)"),
        ("response_truncated_frac", "Pairs with a truncated response"),
    ]
    panels = [p for p in panels if p[0] in records[0]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, (key, title) in zip(axes.flat, panels):
        ax.plot(steps, [r[key] for r in records], color="#2563EB", lw=1.4)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("optimizer step")
    fig.suptitle(f"Task 1 — DPO training ({run_name}, β={records[0]['beta']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / f"task1_dpo_training_curves_{run_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if "implicit_chosen_reward_mean" in records[0]:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        chosen = [r["implicit_chosen_reward_mean"] for r in records]
        rejected = [r["implicit_rejected_reward_mean"] for r in records]
        ax2.plot(steps, chosen, label="chosen  β·log π/π_ref", color="#16A34A")
        ax2.plot(steps, rejected, label="rejected  β·log π/π_ref", color="#DC2626")
        ax2.fill_between(steps, chosen, rejected, alpha=0.12, color="#2563EB")
        ax2.axhline(0, color="black", lw=0.6)
        ax2.set_xlabel("optimizer step")
        ax2.set_ylabel("implicit reward")
        ax2.set_title(f"Implicit reward separation ({run_name})")
        ax2.legend(frameon=False)
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(fig_dir / f"task1_dpo_implicit_reward_{run_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
    print(f"[plot] Training curves saved to {fig_dir}")


# ---------------------------------------------------------------------------
# Data collation
# ---------------------------------------------------------------------------
def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected, truncated = [], [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            ic, mc, info_c = encode_prompt_response(tokenizer, prompt, yc, max_length, return_info=True)
            ir, mr, info_r = encode_prompt_response(tokenizer, prompt, yr, max_length, return_info=True)
            chosen.append((ic, mc))
            rejected.append((ir, mr))
            truncated.append(info_c["response_truncated"] or info_r["response_truncated"])
        cb, rb = pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
        cb["response_truncated"] = torch.tensor(truncated, dtype=torch.float32)
        return cb, rb
    return collate


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------
def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]

    tokenizer = load_tokenizer(cfg["base_model"])
    rows, data_info = load_filtered_pairs(cfg, tokenizer, path, max_examples)
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "data_info": data_info,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------
def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model = bundle["model"]
    optimizer = bundle["optimizer"]
    loader = bundle["loader"]
    beta_val = bundle["beta"]
    tokenizer = bundle["tokenizer"]

    output = repo_path(output_path or cfg["standard_output"])
    output.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = repo_path("report/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    grad_accum = int(cfg.get("grad_accum_steps", 1))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    epochs = int(cfg.get("epochs", 1))

    total, trainable = count_parameters(model)
    log_path = reset_file(results_dir / f"dpo_train_{run_name}.jsonl")
    info = bundle["data_info"]
    print(f"[DPO train] run={run_name} beta={beta_val} epochs={epochs} "
          f"grad_accum={grad_accum} trainable={trainable:,}/{total:,}")
    print(f"[DPO train] data={info['source_file']} rows={info['rows_in_file']} "
          f"dropped_overlength={info['dropped_overlength']} used={info['kept_used']}")
    print(f"[DPO train] output -> {output}")
    print(f"[DPO train] log    -> {log_path}")

    timer = wall_timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    global_step = 0
    optimizer.zero_grad()
    window = defaultdict(list)  # per-optimizer-step accumulation of diagnostics
    record = {}

    for epoch in range(epochs):
        for batch_idx, (chosen_batch, rejected_batch) in enumerate(loader):
            device = next(model.parameters()).device
            trunc = chosen_batch.pop("response_truncated")
            chosen_batch = {k: v.to(device) for k, v in chosen_batch.items()}
            rejected_batch = {k: v.to(device) for k, v in rejected_batch.items()}

            # --- Policy log-probs ---
            model.train()
            policy_chosen_seq_logp, _, _ = response_sequence_logprobs(model, chosen_batch)
            policy_rejected_seq_logp, _, _ = response_sequence_logprobs(model, rejected_batch)

            # --- Reference log-probs (same model, adapter disabled) ---
            with torch.no_grad():
                with reference_mode(model):
                    ref_chosen_seq_logp, _, _ = response_sequence_logprobs(model, chosen_batch)
                    ref_rejected_seq_logp, _, _ = response_sequence_logprobs(model, rejected_batch)

            # --- DPO loss ---
            loss, diagnostics = dpo_loss(
                policy_chosen_seq_logp,
                policy_rejected_seq_logp,
                ref_chosen_seq_logp.detach(),
                ref_rejected_seq_logp.detach(),
                beta_val,
            )

            (loss / grad_accum).backward()

            window["loss"].append(float(loss.item()))
            window["kl_chosen"].append(float((policy_chosen_seq_logp - ref_chosen_seq_logp).mean().item()))
            window["response_truncated_frac"].append(float(trunc.mean().item()))
            for k, v in diagnostics.items():
                window[k].append(float(v.item()) if hasattr(v, "item") else float(v))

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(model), max_grad_norm
                ).item()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                record = {
                    "epoch": epoch,
                    "step": global_step,
                    "batch_idx": batch_idx,
                    "pairs_in_step": len(window["loss"]) * int(cfg["batch_size"]),
                    "grad_norm": float(grad_norm),
                    "beta": float(beta_val),
                    "wall_time": float(timer()),
                    **{k: float(sum(v) / len(v)) for k, v in window.items()},
                }
                append_jsonl(log_path, record)
                window = defaultdict(list)

                if global_step % 10 == 0 or global_step == 1:
                    print(
                        f"  [step {global_step:4d}] loss={record['loss']:.4f} "
                        f"pref_acc={record['preference_accuracy']:.3f} "
                        f"margin={record['reward_margin_mean']:.3f} "
                        f"kl_chosen={record['kl_chosen']:.3f} "
                        f"grad_norm={grad_norm:.3f} time={timer():.1f}s"
                    )

    # --- Save adapter ---
    print(f"[DPO train] Saving adapter to {output}")
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    summary = {
        "run_name": run_name,
        "beta": beta_val,
        "epochs": epochs,
        "total_steps": global_step,
        "dataset_size": len(bundle["rows"]),
        "data": info,
        "used_pair_ids": [pair_id(r) for r in bundle["rows"]],
        "learning_rate": float(cfg["learning_rate"]),
        "effective_batch_pairs": int(cfg["batch_size"]) * grad_accum,
        "seed": int(cfg["seed"]),
        "wall_time_seconds": timer(),
        "final_window_loss": record.get("loss"),
        "final_window_preference_accuracy": record.get("preference_accuracy"),
        "adapter_path": str(output),
    }
    if torch.cuda.is_available():
        summary["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    save_json(results_dir / f"dpo_summary_{run_name}.json", summary)

    try:
        plot_training_curves(log_path, fig_dir, run_name)
    except Exception as e:
        print(f"[plot] Warning: could not generate plots: {e}")

    print(f"[DPO train] Done in {timer():.1f}s, {global_step} steps")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
