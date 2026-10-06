from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import sys

import numpy as np

from common.data import repo_path


def set_seed(seed: int) -> None:
    import torch  # imported lazily so CPU-only result tools work without torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str | Path) -> Path:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    p = ensure_parent(path)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _json_default(o):
    # numpy / torch scalars and arrays that slip into result dicts
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def save_json(path: str | Path, obj: Any) -> None:
    p = ensure_parent(path)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def wall_timer():
    start = time.perf_counter()
    return lambda: time.perf_counter() - start


def reset_file(path: str | Path) -> Path:
    """Remove a log file before a fresh run so `append_jsonl` never mixes runs."""
    p = ensure_parent(path)
    if p.exists():
        p.unlink()
    return p
