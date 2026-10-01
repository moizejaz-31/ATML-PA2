from __future__ import annotations

import argparse
import time
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
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import (
    count_parameters,
    load_policy,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task1_dpo.dpo import dpo_loss


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def plot_training_curves(log_path: Path, fig_dir: Path):
    """Generate and save training curve plots from the JSONL log."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        return

    steps = [r["step"] for r in records]
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Multi-panel training dashboard ---
    metrics = {
        "loss": ("DPO Loss", "tab:red"),
        "preference_accuracy": ("Preference Accuracy", "tab:blue"),
        "logit_mean": ("Logit Mean", "tab:green"),
        "policy_margin_mean": ("Policy Margin Mean", "tab:purple"),
        "ref_margin_mean": ("Ref Margin Mean", "tab:orange"),
        "reward_margin_mean": ("Implicit Reward Margin", "tab:brown"),
        "grad_norm": ("Gradient Norm", "tab:cyan"),
    }
    available = {k: v for k, v in metrics.items() if k in records[0]}

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 3.5 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (key, (title, color)) in zip(axes, available.items()):
        vals = [r.get(key, float("nan")) for r in records]
        ax.plot(steps, vals, color=color, linewidth=1.2, alpha=0.85)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=11, fontweight="bold")

    axes[-1].set_xlabel("Optimization Step", fontsize=10)
    fig.suptitle("Task 1 — DPO Training Curves", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "task1_dpo_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Implicit reward distribution over training ---
    if "implicit_chosen_reward_mean" in records[0]:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        chosen = [r["implicit_chosen_reward_mean"] for r in records]
        rejected = [r["implicit_rejected_reward_mean"] for r in records]
        ax2.plot(steps, chosen, label="Chosen (implicit reward)", color="tab:green", linewidth=1.5)
        ax2.plot(steps, rejected, label="Rejected (implicit reward)", color="tab:red", linewidth=1.5)
        ax2.fill_between(steps, chosen, rejected, alpha=0.15, color="tab:blue")
        ax2.set_xlabel("Optimization Step", fontsize=11)
        ax2.set_ylabel("Mean Implicit Reward (β · log π_θ/π_ref)", fontsize=11)
        ax2.set_title("DPO Implicit Reward Separation Over Training", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(fig_dir / "task1_dpo_implicit_reward.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

    print(f"[plot] Training curves saved to {fig_dir}")


# ---------------------------------------------------------------------------
# Data collation
# ---------------------------------------------------------------------------
def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------
def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
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
    log_path = results_dir / f"dpo_train_{run_name}.jsonl"
    print(f"[DPO train] run={run_name} beta={beta_val} epochs={epochs} "
          f"grad_accum={grad_accum} trainable={trainable:,}/{total:,}")
    print(f"[DPO train] dataset={len(bundle['rows'])} examples")
    print(f"[DPO train] output -> {output}")
    print(f"[DPO train] log    -> {log_path}")

    timer = wall_timer()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(epochs):
        for batch_idx, (chosen_batch, rejected_batch) in enumerate(loader):
            device = next(model.parameters()).device
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

            # Gradient accumulation
            scaled_loss = loss / grad_accum
            scaled_loss.backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(model), max_grad_norm
                ).item()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Compute KL divergence: E[log π_θ - log π_ref] on chosen
                kl_chosen = (policy_chosen_seq_logp - ref_chosen_seq_logp).mean().item()

                record = {
                    "epoch": epoch,
                    "step": global_step,
                    "batch_idx": batch_idx,
                    "loss": float(loss.item()),
                    "grad_norm": float(grad_norm),
                    "kl_chosen": float(kl_chosen),
                    "beta": float(beta_val),
                    "wall_time": float(timer()),
                    **{k: float(v.item()) if hasattr(v, "item") else float(v)
                       for k, v in diagnostics.items()},
                }
                append_jsonl(log_path, record)

                if global_step % 10 == 0 or global_step == 1:
                    print(
                        f"  [step {global_step:4d}] loss={loss.item():.4f} "
                        f"pref_acc={diagnostics['preference_accuracy'].item():.3f} "
                        f"logit_mean={diagnostics['logit_mean'].item():.3f} "
                        f"kl={kl_chosen:.4f} "
                        f"grad_norm={grad_norm:.3f} "
                        f"time={timer():.1f}s"
                    )

                    # Track VRAM
                    if torch.cuda.is_available():
                        vram_mb = torch.cuda.max_memory_allocated() / 1024**2
                        print(f"           peak_vram={vram_mb:.0f}MB")

    # --- Save adapter ---
    print(f"[DPO train] Saving adapter to {output}")
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    # --- Save summary ---
    summary = {
        "run_name": run_name,
        "beta": beta_val,
        "epochs": epochs,
        "total_steps": global_step,
        "dataset_size": len(bundle["rows"]),
        "wall_time_seconds": timer(),
        "final_loss": float(loss.item()),
        "final_preference_accuracy": float(diagnostics["preference_accuracy"].item()),
        "adapter_path": str(output),
    }
    if torch.cuda.is_available():
        summary["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    save_json(results_dir / f"dpo_summary_{run_name}.json", summary)

    # --- Generate training curves ---
    try:
        plot_training_curves(log_path, fig_dir)
    except Exception as e:
        print(f"[plot] Warning: could not generate plots: {e}")

    print(f"[DPO train] Done in {timer():.1f}s, {global_step} steps, final loss={loss.item():.4f}")
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
