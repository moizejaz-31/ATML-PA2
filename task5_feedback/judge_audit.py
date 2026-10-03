"""Instrumented wrapper around the fixed course pairwise judge.

`AuditedPairwiseJudge.compare` returns exactly what `PairwiseAIJudge.compare` returns (same rubric,
deterministic A/B orientation, greedy decoding, parser and result cache). It additionally keeps

  * the raw decoded judge text per comparison, so explicit "TIE" answers can be separated from
    unparseable outputs that the released parser maps to TIE (manual: report ties and ambiguous
    outputs separately);
  * wall-clock time per uncached judge call (inference-cost evidence for the feedback discussion).
"""

from __future__ import annotations

import json
import re
import time

import torch

from common.data import repo_path
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge


class AuditedPairwiseJudge(PairwiseAIJudge):
    def __init__(self, cfg: dict, cache_path):
        super().__init__(cfg, cache_path)
        self.raw_path = repo_path(str(cache_path).replace(".json", "_raw.json"))
        self.raw = json.loads(self.raw_path.read_text(encoding="utf-8")) if self.raw_path.exists() else {}
        self.call_seconds = []
        self.cache_hits = 0

    @torch.no_grad()
    def compare(self, problem: str, a: str, b: str):
        key = self._key(problem, a, b)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        t0 = time.perf_counter()
        swap = int(key[:8], 16) % 2 == 1
        aa, bb = (b, a) if swap else (a, b)
        text = PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], return_tensors="pt", add_generation_prompt=True,
        ).to(next(self.model.parameters()).device)
        out = self.model.generate(ids, max_new_tokens=4, do_sample=False,
                                  pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
        decoded = self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().upper()
        m = re.search(r"\b(A|B|TIE)\b", decoded)
        result = m.group(1) if m else "TIE"
        if swap:
            result = {"A": "B", "B": "A", "TIE": "TIE"}[result]
        self.call_seconds.append(time.perf_counter() - t0)
        self.cache[key] = result
        self.raw[key] = {"decoded": decoded, "parsed": bool(m), "swapped": swap}
        return result

    def is_unparsed(self, problem: str, a: str, b: str) -> bool | None:
        r = self.raw.get(self._key(problem, a, b))
        return None if r is None else (not r["parsed"])

    def flush(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
        self.raw_path.write_text(json.dumps(self.raw, indent=2), encoding="utf-8")

    def cost(self) -> dict:
        s = self.call_seconds
        return {"uncached_calls": len(s), "cache_hits": self.cache_hits,
                "mean_seconds_per_call": float(sum(s) / len(s)) if s else None, "total_seconds": float(sum(s))}
