"""Execute the analysis notebooks in place (CPU only; they read results/ and do not train).

Run: python -m scripts.execute_notebooks                 # all notebooks
     python -m scripts.execute_notebooks task1_dpo        # only notebooks whose path contains "task1_dpo"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    filters = sys.argv[1:]
    paths = sorted(ROOT.glob("task*/notebooks/*.ipynb"))
    if filters:
        paths = [p for p in paths if any(f in str(p.relative_to(ROOT)).replace("\\", "/") for f in filters)]
    failed = []
    for p in paths:
        nb = nbformat.read(p, as_version=4)
        client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(p.parent)}})
        try:
            client.execute()
            status = "ok"
        except CellExecutionError as e:
            status = "FAILED"
            failed.append((p, str(e).splitlines()[-1] if str(e) else "error"))
        nbformat.write(nb, p)
        print(f"[{status}] {p.relative_to(ROOT)}", flush=True)
    for p, msg in failed:
        print(f"  {p.relative_to(ROOT)}: {msg}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
