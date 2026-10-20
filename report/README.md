# Report Artifacts & Summaries Directory

This directory contains the auto-generated tables, visual figures, and qualitative analysis candidates supporting the experimental findings for ATML PA2.

> **Note on Repository Policy**: The compiled PDF report (`PA2_report.pdf`) and standalone LaTeX source packages (`report/latex/`) are excluded from Git tracking via `.gitignore` to maintain a clean repository focused strictly on code, execution pipelines, and verified evaluation artifacts.

---

## Directory Contents

### 1. `figures/`
Stores all visualization plots, convergence curves, and diagnostic dashboards generated directly from the underlying data in `results/`.

* **Task 1 (DPO)**:
  * `task1_dpo_training_curves.png`: Training loss, chosen/rejected log probability separation, and margin evolution.
  * `task1_dpo_beta_ablation.png`: Comparative analysis of margin scaling and KL regularisation across $\beta \in \{0.03, 0.10, 0.30\}$.
  * `task1_dpo_truncation_policy.png`: Empirical token distributions illustrating the prompt-preservation and response-truncation policy.
  * `task1_dpo_length_confounding.png` & `task1_dpo_reward_vs_length_*.png`: Length confounding diagnostics showing the relationship between response length and implicit reward margins.
* **Task 2 (PPO)**:
  * `task2_ppo_continuation_*.png`: Training trajectories for standard PPO and hyperparameter forks ($\epsilon \in \{0.05, 0.20, 0.50\}$, $\beta_{KL} \in \{0.00, 0.10, 0.20\}$).
  * `task2_ppo_clipping_study.png`: Policy ratio distributions $\pi_\theta(a|s) / \pi_{\text{old}}(a|s)$ and empirical clipping frequencies.
  * `task2_ppo_reward_vs_drift_tradeoff.png`: Pareto frontier trading off mean reward gains against reference policy drift ($\mathbb{D}_{\text{KL}}$).
* **Task 3 (GRPO)**:
  * `task3_grpo_continuation_*.png`: Step-by-step training curves for standard continuation and normalization variants.
  * `task3_grpo_group_size_study.png`: Gradient variance, advantage quality, and sample efficiency across group sizes $G \in \{2, 4, 8\}$.
  * `task3_grpo_normalization_study.png`: Performance comparison between Standard GRPO advantage normalization and decoupled DR-GRPO normalization.
* **Task 4 (Safety)**:
  * `task4_safety_calibration_dashboard.png`: Multi-panel calibration overview comparing safe compliance, refusal rates, and over-refusal across SFT, DPO, PPO, and GRPO.
  * `task4_manual_audit_confusion.png`: Confusion matrix validating the automated AI judge against blind human annotations on XSTest.
* **Task 5 (Feedback)**:
  * `task5_math_eval_gsm.png`: In-domain reasoning accuracy on GSM8K comparing SFT, RLVR (rule-based verifier), and RLAIF (LLM judge).
  * `task5_math_eval_transfer.png`: Out-of-domain robustness and performance degradation on the SVAMP transfer benchmark.
  * `task5_perturbation_diagnostics.png`: Sensitivity of evaluation paradigms under format noise and prompt perturbations.
  * `task5_feedback_synthesis_dashboard.png`: Consolidated comparison of reasoning depth, verifier precision, and judge ambiguity.
* **Task 6 (Cross-Task Synthesis)**:
  * `task6_standard_rl_trajectories.png`: Side-by-side comparison of PPO vs. GRPO training dynamics (reward, KL, entropy).
  * `task6_cross_task_synthesis.png`: Multi-axis benchmark synthesis across post-training paradigms (alignment accuracy, compute efficiency, safety compliance, and reasoning capability).

---

### 2. `tables/`
Stores tabular summaries generated automatically by `python -m common.visualize_all`. Each table is provided in both comma-separated (`.csv`) format for data manipulation and formatted LaTeX (`.tex`) for direct report inclusion.

| Table File | Task | Description |
| :--- | :--- | :--- |
| `t1_dpo_summary.*` | Task 1 | Benchmark performance, implicit reward margins, and KL divergence across DPO runs. |
| `t1_length_strata.*` | Task 1 | Model accuracy stratified by chosen/rejected token length discrepancy. |
| `t1_truncation_policy.*` | Task 1 | Dataset statistics under prompt-preservation and right-truncation rules. |
| `t1_word_limit.*` | Task 1 | Adherence to explicit word-count constraints across base and DPO models. |
| `t2_ppo_heldout.*` | Task 2 | Held-out evaluation metrics for SFT reference, midpoint checkpoint, and PPO continuation forks. |
| `t2_ppo_cached_clipping.*` | Task 2 | Clip fractions, ratio distributions, and surrogate bounds across $\epsilon \in \{0.05, 0.20, 0.50\}$. |
| `t3_grpo_heldout.*` | Task 3 | Held-out evaluation metrics for SFT, midpoint, and GRPO continuation forks. |
| `t3_group_size.*` | Task 3 | Advantage estimation variance, sample throughput, and reward metrics across $G \in \{2, 4, 8\}$. |
| `t3_normalization.*` | Task 3 | Performance comparison between standard GRPO and DR-GRPO normalization policies. |
| `t4_safety.*` | Task 4 | Safe compliance, justified refusal, and over-refusal percentages across all 4 aligned policies. |
| `t4_manual_audit.*` | Task 4 | Inter-annotator agreement metrics and confusion breakdown between AI judge and human audit. |
| `t5_math.*` | Task 5 | Mathematical problem-solving accuracy on in-domain (GSM8K) and transfer (SVAMP) benchmarks. |
| `t5_diagnostics.*` | Task 5 | Controlled perturbation resistance, format resilience, and judge vulnerability scores. |
| `t5_judge_outputs.*` | Task 5 | Breakdown of AI judge preferences, win rates, and intransitive comparison rates. |
| `t_compute.*` | Overall | Resource expenditure, peak VRAM utilization, and wall-clock execution times across all methods. |

---

### 3. `qualitative_candidates.md`
Contains hand-inspected qualitative completions generated across all tasks, highlighting:
- Selected contrastive examples demonstrating DPO length bias vs. length-balanced completions.
- PPO policy drift examples under low vs. high KL penalty ($\beta_{KL} = 0.0$ vs. $0.20$).
- Safety edge cases: benign prompts incorrectly flagged as unsafe (over-refusals) vs. successful refusals of toxic queries.
- Step-by-step mathematical reasoning chains from RLVR and RLAIF, contrasting correct reasoning traces with subtle intermediate logical errors.
