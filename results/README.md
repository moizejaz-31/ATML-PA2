# Experimental Results & Provenance Directory

This directory stores all machine-readable evaluation outputs, training curves, rollout logs, and verification records generated across Tasks 1–5 for ATML PA2.

In strict compliance with the assignment specification:
- **Traceability**: Every numerical metric, table entry, and visualization curve reported in the final submission directly originates from a verifiable artifact stored in this directory.
- **Git Tracking**: All compact, non-binary evaluation summaries, structured CSV/JSON metrics, and JSONL log files are tracked in Git to provide complete provenance for grading and reproducibility.

---

## Directory Organization & File Manifest

### Task 1: Direct Preference Optimization (DPO) (`results/task1_dpo/`)

Investigates implicit reward modeling, temperature scaling ($\beta \in \{0.03, 0.10, 0.30\}$), and length confounding mitigations.

| File | Description | Key Contents |
| :--- | :--- | :--- |
| `dpo_preprocessing_report.json` | Tokenizer truncation analysis | Token length distributions, count of truncated samples, right-truncation audit without synthetic EOS. |
| `dpo_summary_standard.json` | Standard DPO training run summary | Total wall-clock time, step count, final loss, implicit margin convergence. |
| `dpo_train_standard.jsonl` | Step-by-step training trajectory | Per-step DPO loss, implicit chosen/rejected log probabilities, reward margin $\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$. |
| `dpo_eval_standard.json` | Standard DPO evaluation on test set | Accuracy, mean implicit reward, margin, token-level KL divergence to SFT reference. |
| `dpo_eval_sft_reference.json` | Baseline SFT model evaluation | Baseline log-likelihoods, baseline implicit rewards against reference model ($= 0.0$). |
| `dpo_beta_ablation_results.json` | Consolidated $\beta$ ablation metrics | Cross-comparison of margins, accuracies, and policy divergence across $\beta \in [0.03, 0.30]$. |
| `dpo_summary_beta_*.json` | Per-condition $\beta$ training summaries | Wall-clock time, loss convergence for $\beta=0.03, 0.10, 0.30$. |
| `dpo_train_beta_*.jsonl` | Per-condition $\beta$ training curves | Fine-grained log-likelihood and reward dynamics under varying regularization strength. |
| `dpo_eval_ablation_beta_*.json` | Per-condition evaluation metrics | Held-out evaluation metrics demonstrating margin scaling vs. policy drift tradeoffs. |
| `dpo_length_analysis.json` | Length confounding diagnostics | Correlation between implicit reward and response length; delta accuracy stratified by length discrepancy. |
| `dpo_summary_length_balanced.json`| Length-balanced dataset training summary | Training log under length-balanced preference data. |
| `dpo_eval_length_balanced.json` | Evaluation of length-balanced DPO | Verification of mitigated length bias and word-limit prompt adherence. |

---

### Task 2: Proximal Policy Optimization (PPO) (`results/task2_ppo/`)

Analyzes on-policy RL fine-tuning with learned critic baselines, surrogate clipping dynamics ($\epsilon \in \{0.05, 0.20, 0.50\}$), and explicit KL penalty pressure ($\beta_{KL} \in \{0.00, 0.10, 0.20\}$).

| File | Description | Key Contents |
| :--- | :--- | :--- |
| `ppo_eval_sft_reference.json` | Base reference policy performance | Unaligned base policy reward, sequence length, and benchmark accuracy. |
| `ppo_eval_midpoint.json` | Course-supplied midpoint checkpoint evaluation | Baseline evaluation prior to continuation forks (starting state for all RL runs). |
| `ppo_summary_standard.json` | Standard continuation summary | Hyperparameters ($\epsilon=0.20, \beta_{KL}=0.10$), wall time, peak VRAM, final reward. |
| `ppo_train_standard.jsonl` | Standard continuation training log | Policy loss, critic value loss, mean return, clip fraction, entropy, running KL. |
| `ppo_rollouts_standard.jsonl` | Sample rollouts during training | Prompts, sampled completions, token logprobs, ref logprobs, GAE advantages, returns. |
| `ppo_eval_standard.json` | Standard continuation held-out evaluation | Mean reward, KL divergence, generation length, and accuracy on held-out prompt set. |
| `ppo_clipping_study_results.json` | Clipping sensitivity study | Impact of $\epsilon \in \{0.05, 0.20, 0.50\}$ on update stability, surrogate bounds, and reward. |
| `ppo_clipping_log_ratios.npz` | Cached policy ratio distributions | Empirical distributions of $\pi_\theta(a|s) / \pi_{\text{old}}(a|s)$ across training steps. |
| `ppo_kl_ablation_results.json` | KL pressure ablation metrics | Tradeoff frontier between reward optimization and reference policy drift. |
| `ppo_summary_fork_*.json` | Fork run summaries | Summary statistics for each specific $(\epsilon, \beta_{KL})$ combination. |
| `ppo_train_fork_*.jsonl` | Fork training trajectories | Step-by-step dynamics for ablations ($\epsilon=0.05, 0.50$ and $\beta_{KL}=0.00, 0.20$). |
| `ppo_rollouts_fork_*.jsonl` | Fork rollout samples | Token-level traces and advantage estimates for ablation checkpoints. |
| `ppo_eval_fork_*.json` | Fork held-out evaluations | Evaluated held-out performance across all hyperparameter forks. |

---

### Task 3: Group Relative Policy Optimization (GRPO) (`results/task3_grpo/`)

Investigates critic-free RL post-training using group-relative baseline normalization, group size scaling ($G \in \{2, 4, 8\}$), and variance normalization policies (Standard GRPO vs. DR-GRPO).

| File | Description | Key Contents |
| :--- | :--- | :--- |
| `grpo_eval_midpoint.json` | Course-supplied midpoint evaluation | Initial performance before continuation training. |
| `grpo_summary_standard.json` | Standard GRPO continuation summary | Wall-clock time, GPU memory allocation, group size $G=4$, training speed. |
| `grpo_train_standard.jsonl` | Step-by-step training curves | Group mean reward, reward variance, per-group normalized advantage, KL drift. |
| `grpo_rollouts_standard.jsonl` | Group rollout generations | Group completions ($G=4$ per prompt), individual scalar rewards, centered advantages. |
| `grpo_eval_standard.json` | Standard GRPO held-out evaluation | Held-out reward, token KL divergence, length distribution, and target accuracy. |
| `grpo_group_size_analysis.json` | Group size ablation ($G \in \{2, 4, 8\}$) | Advantage estimation variance, sample efficiency, and compute-performance frontier. |
| `grpo_normalization_comparison.json` | Advantage normalization analysis | Empirical comparison between Standard GRPO ($\frac{r - \mu}{\sigma + \epsilon}$) and decoupled DR-GRPO normalization. |
| `grpo_summary_fork_norm_*.json` | Normalization fork summaries | Run metadata for alternative normalization schemes. |
| `grpo_train_fork_norm_*.jsonl` | Normalization fork training curves | Advantage stability, gradient variance, and convergence characteristics. |
| `grpo_eval_fork_norm_*.json` | Normalization fork evaluations | Final held-out evaluation metrics under standard vs. DR-GRPO normalization. |

---

### Task 4: Safety & Calibration (`results/task4_safety/`)

Evaluates safe compliance, refusal behavior, and over-refusal rates across SFT, DPO, PPO, and GRPO models using the XSTest prompt suite and automated AI judging calibrated against a blind human audit.

| File | Description | Key Contents |
| :--- | :--- | :--- |
| `generated_{sft,dpo,ppo,grpo}.jsonl` | Raw model responses | 450 prompt responses per policy on the full XSTest suite (contrastive benign & adversarial prompts). |
| `judged_{sft,dpo,ppo,grpo}.jsonl` | Automated AI judge outputs | Five-way classifications: `SAFE_ANSWER`, `JUSTIFIED_REFUSAL`, `UNSAFE_COMPLIANCE`, `OVER_REFUSAL`, `AMBIGUOUS`. |
| `safety_evaluation_results.json` | Comprehensive safety benchmark | Safe compliance rate, justified refusal rate, over-refusal rate, and ambiguity rate per policy. |
| `policy_label_disagreements.json` | Cross-policy disagreement cases | Instances where post-training algorithms diverged in safety vs. refusal boundaries. |
| `manual_audit_sheet.csv` | Blind 60-prompt manual audit | Human annotations collected blindly (without policy labels or automated scores). |
| `manual_audit_key.csv` | De-anonymized audit mapping | Ground-truth mapping connecting blinded row IDs back to policy and prompt identities. |
| `manual_audit_ids.csv` | Stratified prompt sample IDs | Prompt identifiers selected to balance contrastive benign and adversarial distributions. |

---

### Task 5: Feedback Mechanisms — RLVR vs. RLAIF (`results/task5_feedback/`)

Compares deterministic rule-based verification (Reinforcement Learning with Verifiable Rewards, RLVR) against LLM-as-a-judge preference feedback (Reinforcement Learning from AI Feedback, RLAIF) on mathematical reasoning and transfer robustness.

| File | Description | Key Contents |
| :--- | :--- | :--- |
| `math_eval_gsm.json` | In-domain mathematical reasoning (GSM8K) | Final exact-match accuracy, format adherence, step count, and error taxonomy for SFT, RLVR, and RLAIF. |
| `math_eval_transfer.json` | Out-of-domain transfer evaluation (SVAMP) | Generalization accuracy and degradation rate when transferring to shifted question formulations. |
| `generations_gsm_*.jsonl` | Model reasoning traces on GSM8K | Full chain-of-thought solutions, extracted final numeric answers, and rule-based verification flags. |
| `generations_transfer_*.jsonl`| Model reasoning traces on SVAMP | Chain-of-thought traces evaluating cross-distribution robustness and reasoning consistency. |
| `perturbation_scores.json` | Controlled prompt diagnostics | Resilience of judges and policies to formatting noise, prompt paraphrasing, and surface perturbations. |
| `perturbation_items.json` | Diagnostic dataset examples | Systematic test items designed to test reward hacking and judge susceptibility. |
| `judge_ambiguity.json` | AI judge reliability analysis | Ambiguity frequency, intransitivity, and position bias in pairwise LLM judge comparisons. |
| `feedback_synthesis.json` | Comparative synthesis report | Cross-paradigm summary contrasting verifier precision against preference judge flexibility. |

---

## Programmatic Inspection Example

To inspect and reproduce summary tables directly from these artifacts using Python:

```python
import json
import pandas as pd
from pathlib import Path

results_dir = Path("results")

# Load Task 1 DPO summary
dpo_std = json.loads((results_dir / "task1_dpo" / "dpo_eval_standard.json").read_text())
print(f"DPO Standard Accuracy: {dpo_std['accuracy']:.3f}, Mean Reward: {dpo_std['mean_reward']:.3f}")

# Inspect Task 4 Safety Summary
safety = json.loads((results_dir / "task4_safety" / "safety_evaluation_results.json").read_text())
for policy, metrics in safety.items():
    print(f"{policy.upper():6s} | Safe Compliance: {metrics['safe_compliance_rate']:.1%} | Over-Refusal: {metrics['over_refusal_rate']:.1%}")

# Load Task 5 GSM8K accuracy
gsm = json.loads((results_dir / "task5_feedback" / "math_eval_gsm.json").read_text())
for model, res in gsm.items():
    print(f"{model.upper():6s} GSM8K Accuracy: {res['accuracy']:.1%}")
```
