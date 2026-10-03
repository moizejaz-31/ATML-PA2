# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

## Quick start

```bash
git clone https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---
# ATML PA2 - LLM Post-Training

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

## 1. Clone and install

```bash
git clone https://github.com/COURSE_ORG/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points (exact commands, in run order)

`kaggle_runner.ipynb` runs all of these on Kaggle (T4 x2, two parallel lanes, resumable). Every command
writes machine-readable results to `results/` and figures to `report/figures/`.

### Task 1 - DPO

```bash
python -m task1_dpo.preprocess --config configs/dpo.yaml                      # truncation-policy report (CPU)
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft_reference
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml --skip-train         # trains forks that do not exist yet
python -m task1_dpo.analyze_length --config configs/dpo.yaml --skip-train
```

### Task 2 - PPO

```bash
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter none --name sft_reference
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml                 # cached batch + eps forks
python -m task2_ppo.ablate_kl --config configs/ppo.yaml                        # beta_KL forks (eps=0.2/beta=0.1 fork shared)
```

### Task 3 - GRPO

```bash
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml --skip-train
```

### Task 4 - Safety calibration

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml   # blind sheet: 60 prompts x 4 policies
# fill results/task4_safety/manual_audit_sheet.csv (manual_label) without opening manual_audit_key.csv
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

### Task 5 - RLVR vs RLAIF

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

### Task 6 - tables, cross-task figures, qualitative candidates

```bash
python -m common.visualize_all          # report/tables/*.csv|*.tex + task6 figures
python -m scripts.extract_qualitative   # report/qualitative_candidates.md
python -m scripts.build_task_notebooks  # regenerate the per-step inspection notebooks
```

## 5b. Implementation decisions for this run

- **DPO sequence-length policy (TA clarification, 3 Oct).** For every DPO training and evaluation run the
  rendered prompt is kept intact and an over-length response is right-truncated (`common/data.py`,
  based on the starter's prompt-preservation patch). Unlike that patch, no EOS is appended to a response
  that was cut, so DPO is never trained on an artificial end-of-turn. Pairs whose prompt leaves fewer than
  `min_response_tokens: 64` response tokens are dropped (the patch raises an error on them). The effect per
  file and length stratum is in `results/task1_dpo/dpo_preprocessing_report.json` and
  `report/figures/task1_dpo_truncation_policy.png`.
- **Generation prompts** longer than the prompt cap are truncated from the left (TRL convention); the
  tokenizer default (right) removed the assistant header. Reward-model inputs are also left-truncated.
- **KL** everywhere = released sampled-response estimator on the policy's own generations, aggregated
  as one token-level mean (`common/policy_eval.py`); sequence-level sums are stored as well.
- **PPO/GRPO** run with LoRA dropout disabled (`disable_dropout: true`) so ratios/clip fractions measure
  policy change, not dropout noise. Held-out RL evaluation uses the first 64 prompts of the held-out pool
  for every condition (`eval_num_prompts`).
- **Deliberate defects fixed:** DPO logits used `+ref_margin` (`task1_dpo/dpo.py`); PPO surrogate used
  `max` instead of `min` (`task2_ppo/ppo.py`); GRPO advantages were normalised over the whole batch instead
  of within each prompt group (`task3_grpo/grpo.py`).

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.
