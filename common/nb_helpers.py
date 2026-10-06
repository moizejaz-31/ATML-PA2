"""Small display helpers shared by the analysis notebooks (no torch required).

The notebooks read the saved results of the Python-script pipeline (results/, report/) and render
tables, curves and qualitative examples. Nothing here trains or evaluates a model.
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, Markdown, display

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "report" / "figures"

PALETTE = {
    "sft": "#6B7280", "SFT": "#6B7280", "dpo": "#2563EB", "standard": "#2563EB", "ppo": "#DC2626",
    "grpo": "#16A34A", "midpoint": "#111827", "length_balanced": "#16A34A", "rlvr": "#2563EB", "rlaif": "#9333EA",
}
SEQ = ["#93C5FD", "#2563EB", "#1E3A8A"]


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
    })
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 4)


def exists(rel: str) -> bool:
    return (RES / rel).exists()


def load(rel: str):
    return json.loads((RES / rel).read_text(encoding="utf-8"))


def jl(rel: str) -> list[dict]:
    return [json.loads(l) for l in (RES / rel).open(encoding="utf-8") if l.strip()]


def pending(*rels: str) -> bool:
    """True (and a visible notice) when any of the result files is not there yet."""
    missing = [r for r in rels if not exists(r)]
    if missing:
        display(Markdown("> **Pending:** " + ", ".join(f"`results/{m}`" for m in missing) +
                         " not produced yet. Run the step (Kaggle runner or the command in the run cell) and re-execute this notebook."))
    return bool(missing)


def run(cmd: str):
    """Run one pipeline command from the repo root and stream its output into the notebook."""
    print(f"$ {cmd}")
    p = subprocess.run([sys.executable, "-m", *cmd.split()[2:]] if cmd.startswith("python -m") else cmd.split(),
                       cwd=ROOT, capture_output=True, text=True)
    print(p.stdout[-6000:])
    if p.returncode:
        print(p.stderr[-4000:])
        raise RuntimeError(f"command failed: {cmd}")


def esc(text: str, limit: int | None = 900) -> str:
    t = str(text)
    if limit and len(t) > limit:
        t = t[:limit] + " …"
    return html.escape(t).replace("\n", "<br>")


def side_by_side(title: str, prompt: str | None, columns: dict[str, str], meta: dict[str, str] | None = None, limit: int = 900):
    """Render one prompt and several responses next to each other."""
    head = f"<h4 style='margin:8px 0 4px'>{html.escape(title)}</h4>"
    if prompt is not None:
        head += f"<div style='background:#F3F4F6;padding:6px 8px;border-radius:4px;font-size:12px'><b>Prompt:</b> {esc(prompt, 600)}</div>"
    cells = ""
    for name, text in columns.items():
        m = f"<div style='color:#6B7280;font-size:11px'>{html.escape(meta.get(name, ''))}</div>" if meta else ""
        cells += (f"<td style='vertical-align:top;border:1px solid #E5E7EB;padding:6px;font-size:12px;width:{100 // max(len(columns), 1)}%'>"
                  f"<b>{html.escape(name)}</b>{m}<div style='margin-top:4px'>{esc(text, limit)}</div></td>")
    display(HTML(head + f"<table style='width:100%;border-collapse:collapse;table-layout:fixed'><tr>{cells}</tr></table>"))


def ci_text(rate: float, ci) -> str:
    return f"{100 * rate:.1f}% [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}]"


def bootstrap_mean_diff(a, b, n=2000, seed=6304):
    """Paired bootstrap 95% CI of mean(a - b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def smooth(y, w=5):
    y = np.asarray(y, float)
    if len(y) < w:
        return y
    k = np.ones(w) / w
    pad = np.pad(y, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def full_windows(log: pd.DataFrame) -> pd.DataFrame:
    """Drop DPO optimizer steps whose accumulation window was partial (the last step of an epoch when
    the pair count is not a multiple of 16); their per-step means cover only a few pairs."""
    if "pairs_in_step" not in log:
        return log
    full = log[log.pairs_in_step == log.pairs_in_step.max()]
    dropped = len(log) - len(full)
    if dropped:
        print(f"(curves omit {dropped} partial accumulation window(s): "
              + ", ".join(f"step {int(r.step)} = {int(r.pairs_in_step)} pairs" for r in log[log.pairs_in_step < log.pairs_in_step.max()].itertuples()) + ")")
    return full
