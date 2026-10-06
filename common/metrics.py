from __future__ import annotations

import re
import numpy as np

try:  # torch is only needed by the tensor helpers; result analysis works without it
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    torch = None


# A GRPO group is uninformative when its reward std is within the tolerance used by the released
# helper (task3_grpo.grpo.group_relative_advantages clamps the std at eps=1e-6).
GRPO_ZERO_STD_TOL = 1e-6


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def sampled_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor):
    return masked_mean(policy_logp - ref_logp, mask)


def sample_entropy(sampled_logp: torch.Tensor, mask: torch.Tensor):
    return -masked_mean(sampled_logp, mask)


def mean_response_length(mask: torch.Tensor):
    return float(mask.sum(-1).float().mean().item())


def preference_accuracy(chosen_logp, rejected_logp):
    return float((chosen_logp > rejected_logp).float().mean().item())


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def parse_word_limit(prompt: str):
    patterns = [
        r"(?:at most|no more than|under|within)\s+(\d+)\s+words?",
        r"(?:in|use)\s+(\d+)\s+words?\s+(?:or fewer|max(?:imum)?)",
        r"(?:maximum|max)\s+(?:of\s+)?(\d+)\s+words?",
    ]
    lower = str(prompt).lower()
    for pattern in patterns:
        m = re.search(pattern, lower)
        if m:
            return int(m.group(1))
    return None


def word_limit_compliance(prompt: str, response: str):
    limit = parse_word_limit(prompt)
    if limit is None:
        return None
    return float(word_count(response) <= limit)


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def length_stats(values) -> dict:
    """Mean plus dispersion statistics for response lengths (manual: mean + std or IQR)."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"),
                "q25": float("nan"), "q75": float("nan"), "iqr": float("nan")}
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()), "median": float(med),
            "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25)}


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial rate."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


def cohen_kappa(a: list, b: list) -> float:
    """Cohen's kappa between two label lists of equal length."""
    if not a:
        return float("nan")
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


class TokenKLAccumulator:
    """Accumulates the released sampled-response KL estimator over a whole evaluation set.

    Token-level (primary, = `sampled_kl` over all evaluated tokens): sum(log pi - log pi_ref) / #tokens.
    Sequence-level (secondary): mean over responses of the summed per-token log-ratio.
    """

    def __init__(self):
        self.token_sum = 0.0
        self.token_count = 0.0
        self.seq_values = []

    def add(self, policy_logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor):
        diff = ((policy_logp - ref_logp).float() * mask.float()).detach()
        self.token_sum += float(diff.sum().item())
        self.token_count += float(mask.sum().item())
        per_seq = diff.sum(-1).cpu().tolist()
        self.seq_values.extend(per_seq)
        return per_seq

    def token_mean(self) -> float:
        return self.token_sum / max(self.token_count, 1.0)

    def sequence_mean(self) -> float:
        return float(np.mean(self.seq_values)) if self.seq_values else float("nan")
