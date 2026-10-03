"""DPO sequence-length policy: prompt preservation + response truncation + overlength filter.

TA clarification (3 Oct 2026): prompt + response exceeds the 768-token limit for some pairs. The released
starter truncated the PROMPT (keeping response tokens), which for long responses leaves the model
conditioning on almost nothing. This repository instead uses, for every DPO training AND evaluation run:

  1. the rendered prompt is always kept intact;
  2. if prompt + response > max_sequence_length, the response is truncated from the right
     (no EOS is appended to a cut response; see common.data.encode_prompt_response);
  3. a pair is dropped when its prompt leaves fewer than `min_response_tokens` tokens of response
     budget (prompt > max_sequence_length - min_response_tokens). Such pairs cannot express a
     preference between two responses, and the starter patch raises an error on them.

Run `python -m task1_dpo.preprocess --config configs/dpo.yaml` (CPU, tokenizer only) to write
results/task1_dpo/dpo_preprocessing_report.json and the truncation figure used in the report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from common.data import (
    load_yaml,
    preference_responses,
    prompt_messages_from_preference,
    prompt_token_count,
    read_jsonl,
    repo_path,
)
from common.logging_utils import save_json

DATASET_KEYS = ["dpo_standard_train", "dpo_standard_eval", "dpo_length_train", "dpo_length_eval"]


def pair_id(row: dict) -> str:
    return str(row.get("prompt_id", row.get("source_index")))


def stratum_of(row: dict) -> str:
    return str(row.get("length_stratum", row.get("stratum", "all")))


def length_policy(cfg: dict) -> tuple[int, int]:
    return int(cfg["max_sequence_length"]), int(cfg.get("min_response_tokens", 64))


def filter_pairs(tokenizer, rows: list[dict], max_length: int, min_response_tokens: int):
    """Return (kept_rows, dropped) where dropped lists {prompt_id, prompt_tokens, stratum}."""
    kept, dropped = [], []
    limit = max_length - min_response_tokens
    for row in rows:
        n = prompt_token_count(tokenizer, prompt_messages_from_preference(row))
        if n > limit:
            dropped.append({"prompt_id": pair_id(row), "prompt_tokens": n, "stratum": stratum_of(row)})
        else:
            kept.append(row)
    return kept, dropped


def load_filtered_pairs(cfg: dict, tokenizer, path: str, max_examples: int | None = None):
    """Read a DPO file, apply the overlength filter, then take the first `max_examples` kept pairs."""
    rows = read_jsonl(path)
    max_len, min_resp = length_policy(cfg)
    kept, dropped = filter_pairs(tokenizer, rows, max_len, min_resp)
    if max_examples is not None:
        kept = kept[: int(max_examples)]
    info = {
        "source_file": str(path),
        "rows_in_file": len(rows),
        "dropped_overlength": len(dropped),
        "dropped_prompt_ids": [d["prompt_id"] for d in dropped],
        "kept_used": len(kept),
        "max_sequence_length": max_len,
        "min_response_tokens": min_resp,
    }
    return kept, info


def token_length(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"]) + 1  # + EOS


def analyse_file(tokenizer, rows: list[dict], max_length: int, min_response_tokens: int) -> dict:
    """Truncation statistics under both the released rule and the rule used here."""
    limit = max_length - min_response_tokens
    per_stratum = defaultdict(Counter)
    diffs_full, diffs_kept = [], []
    prompt_lens = []
    for row in rows:
        s = stratum_of(row)
        p = prompt_token_count(tokenizer, prompt_messages_from_preference(row))
        yc, yr = preference_responses(row)
        c, r = token_length(tokenizer, yc), token_length(tokenizer, yr)
        prompt_lens.append(p)
        st = per_stratum[s]
        st["pairs"] += 1
        over = (p + c > max_length) or (p + r > max_length)
        st["overlength_pairs"] += int(over)
        # Released starter: prompt is cut to fit the response; when the response alone nearly fills
        # the window the model keeps (almost) no prompt at all.
        st["released_prompt_cut"] += int(over)
        st["released_prompt_mostly_removed"] += int(max(c, r) >= max_length - 16)
        if p > limit:
            st["dropped_here"] += 1
            continue
        budget = max_length - p
        st["kept"] += 1
        st["chosen_truncated_here"] += int(c > budget)
        st["rejected_truncated_here"] += int(r > budget)
        st["either_truncated_here"] += int(c > budget or r > budget)
        diffs_full.append(c - r)
        diffs_kept.append(min(c, budget) - min(r, budget))
    total = Counter()
    for st in per_stratum.values():
        total.update(st)
    d_full, d_kept = np.asarray(diffs_full, float), np.asarray(diffs_kept, float)
    return {
        "per_stratum": {k: dict(v) for k, v in per_stratum.items()},
        "total": dict(total),
        "prompt_tokens_percentiles": {q: float(np.percentile(prompt_lens, q)) for q in (50, 90, 99, 100)},
        "chosen_minus_rejected_tokens_full": {
            "mean": float(d_full.mean()), "median": float(np.median(d_full)),
            "frac_chosen_longer": float((d_full > 0).mean()),
        },
        "chosen_minus_rejected_tokens_after_truncation": {
            "mean": float(d_kept.mean()), "median": float(np.median(d_kept)),
            "frac_chosen_longer": float((d_kept > 0).mean()),
        },
        "_diffs_full": d_full.tolist(),
        "_diffs_kept": d_kept.tolist(),
    }


def plot_report(report: dict, fig_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = {"dpo_standard_train": "Standard train", "dpo_length_train": "Length-balanced train"}
    bins = np.linspace(-600, 600, 61)
    for key, color in [("dpo_standard_train", "#2563EB"), ("dpo_length_train", "#16A34A")]:
        if key in report["files"]:
            d = np.clip(report["files"][key]["_diffs_full"], -600, 600)
            axes[0].hist(d, bins=bins, alpha=0.55, color=color, label=names[key], density=True)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel("tokens(chosen) - tokens(rejected)  [clipped to ±600]")
    axes[0].set_ylabel("density")
    axes[0].set_title("Length structure of the DPO training pairs")
    axes[0].legend(frameon=False)

    rows, labels = [], []
    for key in DATASET_KEYS:
        if key not in report["files"]:
            continue
        for s, st in sorted(report["files"][key]["per_stratum"].items()):
            n = max(st.get("pairs", 0), 1)
            rows.append([st.get("dropped_here", 0) / n, st.get("either_truncated_here", 0) / n,
                         st.get("released_prompt_cut", 0) / n])
            labels.append(f"{key.replace('dpo_', '')}" + ("" if s == "all" else f"\n{s}"))
    rows = np.asarray(rows) * 100
    y = np.arange(len(labels))
    axes[1].barh(y - 0.25, rows[:, 2], 0.25, color="#9CA3AF", label="released rule: prompt cut")
    axes[1].barh(y, rows[:, 1], 0.25, color="#F59E0B", label="ours: response truncated")
    axes[1].barh(y + 0.25, rows[:, 0], 0.25, color="#DC2626", label="ours: pair dropped")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("% of pairs")
    axes[1].set_title("Effect of the 768-token limit")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "task1_dpo_truncation_policy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    from common.models import load_tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tok = load_tokenizer(cfg["base_model"])
    max_len, min_resp = length_policy(cfg)

    report = {
        "policy": {
            "max_sequence_length": max_len,
            "min_response_tokens": min_resp,
            "prompt": "kept intact",
            "response": "right-truncated to the remaining budget; EOS kept only if the full response fits",
            "filter": f"drop pair if rendered prompt > {max_len - min_resp} tokens",
            "applies_to": "all DPO training and evaluation runs (standard, beta forks, length-balanced, all held-out sets)",
        },
        "files": {},
    }
    for key in DATASET_KEYS:
        path = cfg["paths"][key]
        if not repo_path(path).exists():
            print(f"[skip] {path} missing")
            continue
        rows = read_jsonl(path)
        rep = analyse_file(tok, rows, max_len, min_resp)
        _, dropped = filter_pairs(tok, rows, max_len, min_resp)
        rep["dropped_prompt_ids"] = [d["prompt_id"] for d in dropped]
        report["files"][key] = rep
        t = rep["total"]
        print(f"{key:28s} pairs={t['pairs']:5d} overlength={t.get('overlength_pairs', 0):4d} "
              f"dropped={t.get('dropped_here', 0):3d} response-truncated pairs={t.get('either_truncated_here', 0):4d}")

    plot_report(report, repo_path("report/figures"))
    slim = {k: v for k, v in report.items()}
    slim["files"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in report["files"].items()}
    out = repo_path(cfg["results_dir"]) / "dpo_preprocessing_report.json"
    save_json(out, slim)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
