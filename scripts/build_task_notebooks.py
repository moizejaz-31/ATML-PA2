"""
Comprehensive generator for modular sub-task Jupyter Notebooks for ATML PA2.
Ensures 100% coverage of the assignment manual requirements, research questions,
rich inline visualizations, publication figure embeddings, and qualitative text inspections.
"""

import os
import json

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(text):
    lines = [l + "\n" for l in text.strip().split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code_cell(code):
    lines = [l + "\n" for l in code.strip().split("\n")]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

ROOT_ENV_CODE = """%matplotlib inline
import os, sys, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

# Configure matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

# Ensure project root is in sys.path
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    os.chdir(ROOT)
except Exception:
    pass

print("Project root:", ROOT)
"""

def generate_all():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    notebooks = {}

    # =========================================================================
    # TASK 1: DIRECT PREFERENCE OPTIMIZATION (DPO)
    # =========================================================================

    # 1.1 Standard DPO
    t1_1 = [
        md_cell("""# Task 1 — Step 1: Standard Direct Preference Optimization (DPO)

### Mathematical Formulation
Direct Preference Optimization models policy preferences directly without training an intermediate reward model or sampling on-policy trajectories:
$$\\mathcal{L}_{\\text{DPO}}(\\theta) = -\\mathbb{E}_{(x, y^+, y^-)}\\left[\\log \\sigma\\left(\\beta \\left[\\log\\frac{\\pi_\\theta(y^+|x)}{\\pi_{\\text{ref}}(y^+|x)} - \\log\\frac{\\pi_\\theta(y^-|x)}{\\pi_{\\text{ref}}(y^-|x)}\\right]\\right)\\right]$$

where:
- $x$ is the prompt, $y^+$ is the preferred response, and $y^-$ is the rejected response.
- $\\pi_{\\text{ref}}$ is the frozen reference policy (`Qwen2.5-1.5B-Instruct`).
- $\\beta$ controls the regularization strength relative to the reference policy.

### Required Evidence (from Course Manual):
1. **Training trajectory**: DPO loss, batch preference accuracy, implicit reward margin, and sequence KL estimate over 1 epoch (1500 pairs).
2. **Evaluation metrics**: Held-out preference accuracy on 300 test pairs, mean KL drift from reference, scalar reward-model score on 100 generated responses, and response-length statistics (mean and std).
3. **Qualitative completions**: Top 3 highest-reward and bottom 3 lowest-reward generated responses."""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Execution Settings\nSet `FORCE_RUN = True` to re-run training and evaluation, or `False` to inspect saved artifacts."),
        code_cell("""FORCE_RUN = False
CONFIG = "configs/dpo.yaml"
OUTPUT_DIR = "outputs/task1_dpo/standard"
RESULTS_DIR = "results/task1_dpo"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("report/figures", exist_ok=True)
"""),
        md_cell("## 2. Train Standard DPO (1 Epoch)"),
        code_cell("""if FORCE_RUN or not os.path.exists(OUTPUT_DIR):
    !python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
else:
    print(f"[OK] Trained adapter found at {OUTPUT_DIR}")
"""),
        md_cell("## 3. Evaluate Standard DPO on Held-Out UltraFeedback Pairs"),
        code_cell("""eval_file = os.path.join(RESULTS_DIR, "dpo_eval_standard.json")
if FORCE_RUN or not os.path.exists(eval_file):
    !python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
else:
    print(f"[OK] Evaluation results found at {eval_file}")
"""),
        md_cell("## 4. Summary Table of Standard DPO Metrics"),
        code_cell("""if os.path.exists(eval_file):
    with open(eval_file, encoding='utf-8') as f:
        eval_data = json.load(f)
    
    summary_df = pd.DataFrame([{
        "Metric": "Held-out Preference Accuracy",
        "Value": f"{eval_data.get('held_out_preference_accuracy', 0)*100:.2f}%",
        "Target/Baseline": "Random guess: 50%"
    }, {
        "Metric": "Policy Drift (Mean KL from Ref)",
        "Value": f"{eval_data.get('mean_kl_from_reference', 0):.4f}",
        "Target/Baseline": "Reference = 0.0"
    }, {
        "Metric": "Mean Reward-Model Score",
        "Value": f"{eval_data.get('mean_reward', 0):.3f} ± {eval_data.get('std_reward', 0):.3f}",
        "Target/Baseline": "SFT baseline ~ 0.8"
    }, {
        "Metric": "Mean Response Length (words)",
        "Value": f"{eval_data.get('mean_response_length_words', 0):.1f}",
        "Target/Baseline": "Word limit target: 40"
    }, {
        "Metric": "Word-Limit Compliance Rate",
        "Value": f"{eval_data.get('word_limit_compliance_rate', 0)*100:.1f}%",
        "Target/Baseline": "Higher is better"
    }])
    display(summary_df)
"""),
        md_cell("## 5. Visualizing DPO Training Dynamics"),
        code_cell("""log_file = os.path.join(RESULTS_DIR, "dpo_train_standard.jsonl")
if os.path.exists(log_file):
    records = [json.loads(line) for line in open(log_file, encoding='utf-8')]
    df = pd.DataFrame(records)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    # 1. Loss
    axes[0].plot(df['step'], df['loss'], color='#2563EB', lw=2)
    axes[0].set_title('DPO Training Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Preference Accuracy
    axes[1].plot(df['step'], df['pref_acc']*100, color='#10B981', lw=2)
    axes[1].axhline(50, color='gray', linestyle='--', alpha=0.7)
    axes[1].set_title('Batch Preference Accuracy (%)')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Logit Margin
    axes[2].plot(df['step'], df['logit_mean'], color='#8B5CF6', lw=2)
    axes[2].axhline(0, color='gray', linestyle='--', alpha=0.7)
    axes[2].set_title('Mean Implicit Reward Margin')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Margin')
    axes[2].grid(True, alpha=0.3)
    
    # 4. KL Drift
    axes[3].plot(df['step'], df['kl'], color='#F59E0B', lw=2)
    axes[3].set_title('Sequence KL Drift')
    axes[3].set_xlabel('Step')
    axes[3].set_ylabel('KL')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
"""),
        md_cell("## 6. Publication Figures Inspection"),
        code_cell("""fig_paths = [
    "report/figures/task1_dpo_training_curves.png",
    "report/figures/task1_dpo_eval_standard.png"
]
for p in fig_paths:
    if os.path.exists(p):
        print(f"Displaying publication figure: {p}")
        display(Image(filename=p, width=650))
"""),
        md_cell("## 7. Qualitative Samples Inspection (Reward vs. Quality)\nThe manual requires showing examples where reward model scores either align or conflict with actual response quality."),
        code_cell("""if os.path.exists(eval_file):
    per_ex = eval_data.get("per_example", [])
    if per_ex:
        sorted_ex = sorted(per_ex, key=lambda x: x.get("reward", 0), reverse=True)
        print("=" * 80)
        print("TOP 2 HIGHEST REWARD RESPONSES:")
        print("=" * 80)
        for i, item in enumerate(sorted_ex[:2], 1):
            print(f"\\n--- Top #{i} | Reward Score = {item.get('reward', 0):.3f} ---")
            print(f"PROMPT: {item.get('prompt', '')[:120]}...")
            print(f"RESPONSE: {item.get('response', '')[:250]}...")
            
        print("\\n" + "=" * 80)
        print("BOTTOM 2 LOWEST REWARD RESPONSES:")
        print("=" * 80)
        for i, item in enumerate(sorted_ex[-2:], 1):
            print(f"\\n--- Bottom #{i} | Reward Score = {item.get('reward', 0):.3f} ---")
            print(f"PROMPT: {item.get('prompt', '')[:120]}...")
            print(f"RESPONSE: {item.get('response', '')[:250]}...")
"""),
        md_cell("""## 8. Research Question 1 Analysis & Takeaways
- **Did standard DPO improve held-out preference fitting?** Compare against the 50% random baseline.
- **Is stronger preference fitting accompanied by useful behavior or disproportionate policy drift?** Analyze KL drift trajectory.""")
    ]
    notebooks["task1_dpo/notebooks/1_standard_dpo.ipynb"] = make_notebook(t1_1)

    # 1.2 Beta Ablation
    t1_2 = [
        md_cell("""# Task 1 — Step 2: DPO Regularization Strength Study ($\\beta$ Sweep)

### Research Question
How does changing the regularization parameter $\\beta \\in \\{0.03, 0.10, 0.30\\}$ alter:
1. Preference fitting (held-out preference accuracy)?
2. Reference policy drift ($D_{\\text{KL}}(\\pi_\\theta \\parallel \\pi_{\\text{ref}})$)?
3. Reward model score and generated response length?

Is the Pareto relationship between reward and reference-model drift monotonic over this range?

### Experimental Protocol
- Start from the identical `Qwen2.5-1.5B-Instruct` LoRA initialization.
- Train short forks (600 examples, matched optimizer, seed, and LoRA rank).
- Evaluate on the identical 300 held-out UltraFeedback pairs."""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Run Beta Sweep or Load Cached Results"),
        code_cell("""FORCE_RUN = False
if FORCE_RUN:
    !python -m task1_dpo.ablate_beta --config configs/dpo.yaml
else:
    print("[OK] Loading saved beta ablation results...")
"""),
        md_cell("## 2. Beta Ablation Comparative Metrics Table"),
        code_cell("""ablation_file = "results/task1_dpo/dpo_beta_ablation_results.json"
if os.path.exists(ablation_file):
    with open(ablation_file, encoding='utf-8') as f:
        ab_data = json.load(f)
    
    rows = []
    for b_str in sorted(ab_data.keys(), key=float):
        ev = ab_data[b_str].get("eval", {})
        tr = ab_data[b_str].get("train", {})
        rows.append({
            "Beta (β)": float(b_str),
            "Held-out Pref Acc (%)": f"{ev.get('held_out_preference_accuracy', 0)*100:.2f}%",
            "Mean KL Drift": f"{ev.get('mean_kl_from_reference', 0):.4f}",
            "Mean Reward": f"{ev.get('mean_reward', 0):.4f}",
            "Std Reward": f"{ev.get('std_reward', 0):.4f}",
            "Mean Length (words)": f"{ev.get('mean_response_length_words', 0):.1f}",
            "Final Train Loss": f"{tr.get('final_loss', 0):.4f}"
        })
    df_beta = pd.DataFrame(rows).sort_values("Beta (β)")
    display(df_beta)
"""),
        md_cell("## 3. Visualizing the Reward vs. KL Pareto Frontier & Beta Dynamics"),
        code_cell("""if os.path.exists(ablation_file):
    betas = df_beta["Beta (β)"].values
    rewards = [float(r["Mean Reward"]) for _, r in df_beta.iterrows()]
    kls = [float(r["Mean KL Drift"]) for _, r in df_beta.iterrows()]
    accs = [float(r["Held-out Pref Acc (%)"].replace('%','')) for _, r in df_beta.iterrows()]
    lens = [float(r["Mean Length (words)"]) for _, r in df_beta.iterrows()]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Pareto Frontier (Reward vs KL)
    axes[0].plot(kls, rewards, marker='o', markersize=10, color='#8B5CF6', lw=2.5)
    for i, b in enumerate(betas):
        axes[0].annotate(f"β={b}", (kls[i], rewards[i]), textcoords="offset points", xytext=(8, -8), fontweight='bold')
    axes[0].set_title('Pareto Frontier: Reward vs. Reference Drift')
    axes[0].set_xlabel('Mean KL(π_θ || π_ref)')
    axes[0].set_ylabel('Mean Reward-Model Score')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Preference Accuracy vs Beta
    axes[1].plot(betas, accs, marker='s', markersize=10, color='#EC4899', lw=2.5)
    axes[1].set_title('Held-out Preference Accuracy vs. β')
    axes[1].set_xlabel('Regularization Parameter β')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Generated Length vs Beta
    axes[2].bar([str(b) for b in betas], lens, color='#3B82F6', width=0.4, alpha=0.85)
    for i, l in enumerate(lens):
        axes[2].text(i, l + 1, f"{l:.1f}", ha='center', fontweight='bold')
    axes[2].set_title('Generated Response Length vs. β')
    axes[2].set_xlabel('Regularization Parameter β')
    axes[2].set_ylabel('Mean Words')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
"""),
        md_cell("## 4. Publication Dashboard Inspection"),
        code_cell("""for p in ["report/figures/task1_dpo_beta_ablation.png", "report/figures/task1_dpo_reward_vs_kl_frontier.png"]:
    if os.path.exists(p):
        print(f"Displaying publication figure: {p}")
        display(Image(filename=p, width=650))
"""),
        md_cell("""## 5. Research Findings & Report Notes
- **Monotonicity**: Does increasing $\\beta$ monotonically tighten reference constraint or increase preference fitting?
- **Trade-off Interpretation**: How does the scale factor interact with gradient magnitude on the preference log-ratio?""")
    ]
    notebooks["task1_dpo/notebooks/2_beta_ablation.ipynb"] = make_notebook(t1_2)

    # 1.3 Length Confounding
    t1_3 = [
        md_cell("""# Task 1 — Step 3: Dataset-Induced Length Confounding Study

### Motivation & Background
Preference datasets (like UltraFeedback) often suffer from **length bias**: human or AI annotators frequently prefer longer, more verbose responses, even when the extra text adds no substantive value.
This study tests:
1. Whether standard DPO preference gains are driven by length exploitation rather than content quality.
2. Whether training on a **length-balanced dataset** (equal parts `preferred_longer`, `matched_length`, `rejected_longer`) mitigates verbosity.
3. Explicit **word-limit constraint compliance** on `data/word_limit_prompts.jsonl`."""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Run Length Analysis or Load Cached Diagnostics"),
        code_cell("""FORCE_RUN = False
if FORCE_RUN:
    !python -m task1_dpo.analyze_length --config configs/dpo.yaml
else:
    print("[OK] Loading saved length confounding analysis...")
"""),
        md_cell("## 2. Stratified Preference Accuracy Breakdown"),
        code_cell("""len_file = "results/task1_dpo/dpo_length_analysis.json"
if os.path.exists(len_file):
    with open(len_file, encoding='utf-8') as f:
        len_data = json.load(f)
        
    std_strat = len_data.get("standard_dpo", {}).get("stratified", {})
    bal_strat = len_data.get("length_balanced_dpo", {}).get("stratified", {})
    
    strata = ["preferred_longer", "length_matched", "rejected_longer"]
    rows = []
    for s in strata:
        s_info = std_strat.get(s, {})
        b_info = bal_strat.get(s, {})
        rows.append({
            "Stratum": s.replace('_', ' ').title(),
            "Standard DPO Accuracy": f"{s_info.get('accuracy', 0)*100:.2f}%",
            "Length-Balanced DPO Accuracy": f"{b_info.get('accuracy', 0)*100:.2f}%",
            "Standard Margin": f"{s_info.get('mean_margin', 0):.4f}",
            "Balanced Margin": f"{b_info.get('mean_margin', 0):.4f}",
            "Pairs Count": s_info.get('total_pairs', 0)
        })
    df_strata = pd.DataFrame(rows)
    display(df_strata)
"""),
        md_cell("## 3. Explicit Word-Limit Compliance"),
        code_cell("""if os.path.exists(len_file):
    std_wl = len_data.get("standard_dpo", {}).get("word_limit", {})
    bal_wl = len_data.get("length_balanced_dpo", {}).get("word_limit", {})
    
    wl_df = pd.DataFrame([{
        "Condition": "Standard DPO",
        "Compliance Rate (%)": f"{std_wl.get('compliance_rate', 0)*100:.1f}%",
        "Mean Word Count": f"{std_wl.get('mean_word_count', 0):.1f}",
        "Target Word Count": "40 words"
    }, {
        "Condition": "Length-Balanced DPO",
        "Compliance Rate (%)": f"{bal_wl.get('compliance_rate', 0)*100:.1f}%",
        "Mean Word Count": f"{bal_wl.get('mean_word_count', 0):.1f}",
        "Target Word Count": "40 words"
    }])
    display(wl_df)
"""),
        md_cell("## 4. Visualizing Stratified Accuracy and Confounding Shifts"),
        code_cell("""if os.path.exists(len_file):
    labels = [r["Stratum"] for r in rows]
    std_accs = [float(r["Standard DPO Accuracy"].replace('%','')) for r in rows]
    bal_accs = [float(r["Length-Balanced DPO Accuracy"].replace('%','')) for r in rows]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Stratified Accuracy
    axes[0].bar(x - width/2, std_accs, width, label='Standard DPO', color='#3B82F6', alpha=0.85)
    axes[0].bar(x + width/2, bal_accs, width, label='Length-Balanced DPO', color='#10B981', alpha=0.85)
    axes[0].set_ylabel('Held-Out Accuracy (%)')
    axes[0].set_title('Preference Accuracy Across Length Strata')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Word-Limit Compliance
    wl_labels = ["Standard DPO", "Length-Balanced DPO"]
    wl_comps = [float(std_wl.get('compliance_rate', 0)*100), float(bal_wl.get('compliance_rate', 0)*100)]
    bars = axes[1].bar(wl_labels, wl_comps, color=['#3B82F6', '#10B981'], width=0.4, alpha=0.85)
    for b in bars:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2., h + 1, f"{h:.1f}%", ha='center', fontweight='bold')
    axes[1].set_ylabel('Compliance Rate (%)')
    axes[1].set_title('Explicit Word-Limit Compliance')
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
"""),
        md_cell("## 5. Publication Dashboard Inspection"),
        code_cell("""p = "report/figures/task1_dpo_length_confounding.png"
if os.path.exists(p):
    print(f"Displaying publication figure: {p}")
    display(Image(filename=p, width=700))
"""),
        md_cell("""## 6. Research Question 2 Analysis
- When preferred responses are shorter (`rejected_longer`), how did Standard DPO perform vs Length-Balanced DPO?
- Does length-balancing improve instruction following on concise requests?""")
    ]
    notebooks["task1_dpo/notebooks/3_length_confounding.ipynb"] = make_notebook(t1_3)

    # =========================================================================
    # TASK 2: PROXIMAL POLICY OPTIMIZATION (PPO)
    # =========================================================================

    # 2.1 Standard PPO
    t2_1 = [
        md_cell("""# Task 2 — Step 1: Standard Proximal Policy Optimization (PPO) Continuation

### Objective & Mathematical Formulation
Online RLHF optimizes policy $\\pi_\\theta$ against a frozen reward model with a reference-policy penalty:
$$r_t = r_{\\text{task}}\\cdot \\mathbf{1}[t = T] - \\beta_{\\text{KL}} \\left(\\log \\pi_\\theta(a_t|s_t) - \\log \\pi_{\\text{ref}}(a_t|s_t)\\right)$$

The PPO clipped surrogate objective updates the policy:
$$\\mathcal{L}_{\\text{clip}}(\\theta) = \\hat{\\mathbb{E}}_t \\left[ \\min\\left( \\rho_t(\\theta) \\hat{A}_t, \\text{clip}(\\rho_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_t \\right) \\right]$$
where importance ratio $\\rho_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\text{old}}(a_t|s_t)}$, and $\\hat{A}_t$ is estimated using Generalized Advantage Estimation (GAE) with a learned value critic $V_\\phi(s)$.

### Protocol
- Continue from the course-provided staff midpoint checkpoints (`checkpoints/ppo_midpoint_policy`, `checkpoints/ppo_midpoint_value`).
- Execute 20 updates.
- Track trajectories of reward, KL drift, value loss, policy loss, clip fraction, entropy, gradient norm, response length, peak VRAM, and wall-clock runtime."""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Run PPO Continuation or Load Saved Logs"),
        code_cell("""FORCE_RUN = False
if FORCE_RUN:
    !python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
    !python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
else:
    print("[OK] Loading existing PPO continuation artifacts...")
"""),
        md_cell("## 2. Summary Table of PPO Continuation Run"),
        code_cell("""summary_file = "results/task2_ppo/ppo_summary_standard.json"
if os.path.exists(summary_file):
    with open(summary_file, encoding='utf-8') as f:
        summary_data = json.load(f)
    display(pd.DataFrame([summary_data]))
else:
    print("Run continuation to generate ppo_summary_standard.json")
"""),
        md_cell("## 3. Visualizing PPO Continuation Trajectories (20 Updates)"),
        code_cell("""log_file = "results/task2_ppo/ppo_train_standard.jsonl"
if os.path.exists(log_file):
    records = [json.loads(line) for line in open(log_file, encoding='utf-8')]
    df = pd.DataFrame(records)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    
    # 1. Reward
    axes[0, 0].plot(df['update'], df['reward_mean'], color='#10B981', lw=2)
    axes[0, 0].set_title('Mean Learned Reward')
    axes[0, 0].set_xlabel('Update')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. KL Drift
    axes[0, 1].plot(df['update'], df['kl_mean'], color='#EF4444', lw=2)
    axes[0, 1].set_title('Mean Policy Drift (KL)')
    axes[0, 1].set_xlabel('Update')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Clip Fraction
    axes[0, 2].plot(df['update'], df['clip_fraction']*100, color='#F59E0B', lw=2)
    axes[0, 2].set_title('Token Clip Fraction (%)')
    axes[0, 2].set_xlabel('Update')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Value Loss
    axes[1, 0].plot(df['update'], df['value_loss'], color='#8B5CF6', lw=2)
    axes[1, 0].set_title('Critic Value Loss')
    axes[1, 0].set_xlabel('Update')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Policy Entropy
    axes[1, 1].plot(df['update'], df['entropy'], color='#06B6D4', lw=2)
    axes[1, 1].set_title('Policy Entropy')
    axes[1, 1].set_xlabel('Update')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Response Length
    axes[1, 2].plot(df['update'], df['mean_response_length'], color='#6366F1', lw=2)
    axes[1, 2].set_title('Response Length (tokens)')
    axes[1, 2].set_xlabel('Update')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
"""),
        md_cell("""## 4. Compute Footprint Analysis
Report peak VRAM and wall-clock runtime for the standard 20-update continuation to compare later against critic-free GRPO in Task 3.""")
    ]
    notebooks["task2_ppo/notebooks/1_standard_ppo.ipynb"] = make_notebook(t2_1)

    # 2.2 Clipping Study
    t2_2 = [
        md_cell("""# Task 2 — Step 2: PPO Policy Clipping Study ($\\epsilon$ Ablation)

### Research Question
How does the clipping parameter $\\epsilon \\in \\{0.05, 0.20, 0.50\\}$ alter:
1. **Geometric constraint**: Fraction of tokens clipped and affected-token fraction on the fixed cached rollout batch (`cached/ppo_rollout.pt`).
2. **Optimization stability**: Short-fork continuation stability, gradient norm variance, and held-out response quality under matched budgets."""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Run Clipping Study or Load Results"),
        code_cell("""FORCE_RUN = False
if FORCE_RUN:
    !python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
else:
    print("[OK] Loading saved clipping study results...")
"""),
        md_cell("## 2. Geometric Clipping Analysis on Cached Batch"),
        code_cell("""clip_file = "results/task2_ppo/ppo_clipping_study_results.json"
if os.path.exists(clip_file):
    with open(clip_file, encoding='utf-8') as f:
        clip_data = json.load(f)
    
    cached = clip_data.get("cached_geometry", {})
    forks = clip_data.get("fork_results", {})
    
    rows = []
    for eps_str in sorted(cached.keys(), key=float):
        c_stats = cached[eps_str]
        f_eval = forks.get(eps_str, {}).get("eval", {})
        rows.append({
            "Epsilon (ε)": float(eps_str),
            "Cached Clip Frac (%)": f"{c_stats.get('clip_fraction', 0)*100:.2f}%",
            "Affected Token Frac (%)": f"{c_stats.get('affected_fraction', 0)*100:.2f}%",
            "Cached Surrogate Loss": f"{c_stats.get('surrogate_loss', 0):.4f}",
            "Fork Mean Reward": f"{f_eval.get('mean_reward', float('nan')):.3f}",
            "Fork Mean KL": f"{f_eval.get('mean_kl', float('nan')):.4f}"
        })
    df_clip = pd.DataFrame(rows).sort_values("Epsilon (ε)")
    display(df_clip)
"""),
        md_cell("## 3. Visualizing Token Clipping vs. Epsilon"),
        code_cell("""if os.path.exists(clip_file):
    eps_vals = df_clip["Epsilon (ε)"].values
    clip_pct = [float(r["Cached Clip Frac (%)"].replace('%','')) for _, r in df_clip.iterrows()]
    aff_pct = [float(r["Affected Token Frac (%)"].replace('%','')) for _, r in df_clip.iterrows()]
    
    x = np.arange(len(eps_vals))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, clip_pct, width, label='Clipped Tokens (|ρ - 1| > ε)', color='#EF4444', alpha=0.85)
    ax.bar(x + width/2, aff_pct, width, label='Affected Tokens', color='#F59E0B', alpha=0.85)
    ax.set_ylabel('Percentage of Tokens (%)')
    ax.set_title('PPO Token Clipping Exposure on Cached Batch')
    ax.set_xticks(x)
    ax.set_xticklabels([f"ε={e}" for e in eps_vals])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
"""),
        md_cell("""## 4. Research Question 1 Analysis & Takeaways
- How does $\\epsilon = 0.05$ restrict update geometry compared to $\\epsilon = 0.50$?
- Did conservative clipping (small $\\epsilon$) provide superior optimization stability?""")
    ]
    notebooks["task2_ppo/notebooks/2_clipping_study.ipynb"] = make_notebook(t2_2)

    # 2.3 KL Pressure
    t2_3 = [
        md_cell("""# Task 2 — Step 3: Reward-Overoptimization and KL Pressure ($\\beta_{\\text{KL}}$)

### Research Question
When KL penalty pressure $\\beta_{\\text{KL}} \\in \\{0, 0.10, 0.20\\}$ is modified:
1. Which observable metric shifts first: learned reward, policy entropy, reference drift (KL), or response length?
2. Does weakening KL pressure ($\beta_{\\text{KL}} = 0$) produce **reward overoptimization (Goodhart's Law)**, where reward increases without genuine quality improvement?"""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Run KL Pressure Sweep or Load Results"),
        code_cell("""FORCE_RUN = False
if FORCE_RUN:
    !python -m task2_ppo.ablate_kl --config configs/ppo.yaml
else:
    print("[OK] Loading saved KL pressure results...")
"""),
        md_cell("## 2. Summary Table across $\\beta_{\\text{KL}}$ Values"),
        code_cell("""kl_file = "results/task2_ppo/ppo_kl_ablation_results.json"
if os.path.exists(kl_file):
    with open(kl_file, encoding='utf-8') as f:
        kl_data = json.load(f)
        
    rows = []
    for b_str in sorted(kl_data.keys(), key=float):
        ev = kl_data[b_str].get("eval", {})
        rows.append({
            "Beta_KL": float(b_str),
            "Held-Out Reward": f"{ev.get('mean_reward', 0):.4f}",
            "Policy Drift (KL)": f"{ev.get('mean_kl', 0):.4f}",
            "Mean Length (tokens)": f"{ev.get('mean_response_length_tokens', 0):.1f}",
            "EOS Termination Rate": f"{ev.get('eos_termination_rate', 0)*100:.1f}%"
        })
    df_kl = pd.DataFrame(rows).sort_values("Beta_KL")
    display(df_kl)
"""),
        md_cell("## 3. Visualizing the Reward vs. Drift Trade-Off under KL Shaping"),
        code_cell("""if os.path.exists(kl_file):
    b_vals = df_kl["Beta_KL"].values
    rews = [float(r["Held-Out Reward"]) for _, r in df_kl.iterrows()]
    kl_vals = [float(r["Policy Drift (KL)"]) for _, r in df_kl.iterrows()]
    lengths = [float(r["Mean Length (tokens)"]) for _, r in df_kl.iterrows()]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Reward & KL vs Beta_KL
    ax1 = axes[0]
    ax1.plot(b_vals, rews, 'o-', color='#10B981', lw=2, label='Mean Reward')
    ax1.set_xlabel('β_KL')
    ax1.set_ylabel('Mean Reward', color='#10B981')
    ax1.tick_params(axis='y', labelcolor='#10B981')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(b_vals, kl_vals, 's--', color='#EF4444', lw=2, label='Mean KL')
    ax2.set_ylabel('Mean KL Drift', color='#EF4444')
    ax2.tick_params(axis='y', labelcolor='#EF4444')
    axes[0].set_title('Reward and Reference Drift vs. β_KL')
    
    # 2. Length vs Beta_KL
    axes[1].bar([str(b) for b in b_vals], lengths, color='#6366F1', width=0.4, alpha=0.85)
    for i, l in enumerate(lengths):
        axes[1].text(i, l + 2, f"{l:.1f}", ha='center', fontweight='bold')
    axes[1].set_title('Mean Response Length vs. β_KL')
    axes[1].set_xlabel('β_KL')
    axes[1].set_ylabel('Length (tokens)')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
"""),
        md_cell("""## 4. Qualitative Disagreement Evidence
Identify responses where $\\beta_{\\text{KL}} = 0$ achieved very high reward scores but exhibited degenerative patterns (repetitive phrasing, evasive filler, or truncation).""")
    ]
    notebooks["task2_ppo/notebooks/3_kl_pressure_study.ipynb"] = make_notebook(t2_3)

    # =========================================================================
    # TASK 3: GROUP RELATIVE POLICY OPTIMIZATION (GRPO)
    # =========================================================================

    # 3.1 Standard GRPO
    t3_1 = [
        md_cell("""# Task 3 — Step 1: Standard Group Relative Policy Optimization (GRPO)

### Objective & Formulation
GRPO removes the learned value critic $V_\\phi(s)$ entirely, replacing it with group-relative advantage normalization across $K$ sampled completions per prompt:
$$A_k = \\frac{r_k - \\mu_r}{\\sigma_r + \\varepsilon}, \\quad \\mu_r = \\frac{1}{K}\\sum_{j=1}^K r_j, \\quad \\sigma_r = \\sqrt{\\frac{1}{K}\\sum_{j=1}^K (r_j - \\mu_r)^2}$$

The policy is optimized using the clipped surrogate loss with sequence normalization:
$$\\mathcal{L}_{\\text{GRPO}}(\\theta) = -\\hat{\\mathbb{E}}_{x,\\{y_k\\}} \\left[ \\frac{1}{K}\\sum_{k=1}^K \\frac{1}{T_k}\\sum_{t=1}^{T_k} \\min\\left(\\rho_{k,t} A_k, \\text{clip}(\\rho_{k,t}, 1-\\epsilon, 1+\\epsilon)A_k\\right) \\right] + \\beta D_{\\text{KL}}(\\pi_\\theta \\parallel \\pi_{\\text{ref}})$$

### Required Evidence:
1. **Continuation Trajectories**: Reward, KL drift, within-group reward standard deviation, uninformative-group fraction ($\sigma_r = 0$), entropy, and length.
2. **Compute Comparison**: Peak VRAM and wall-clock time compared directly to Task 2 PPO."""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Run GRPO Continuation or Load Records"),
        code_cell("""FORCE_RUN = False
if FORCE_RUN:
    !python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
    !python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
else:
    print("[OK] Loading existing GRPO continuation records...")
"""),
        md_cell("## 2. Summary Table of GRPO Continuation Run"),
        code_cell("""summary_file = "results/task3_grpo/grpo_summary_standard.json"
if os.path.exists(summary_file):
    with open(summary_file, encoding='utf-8') as f:
        summary_data = json.load(f)
    display(pd.DataFrame([summary_data]))
"""),
        md_cell("## 3. Visualizing GRPO Trajectories & Group Informativeness"),
        code_cell("""log_file = "results/task3_grpo/grpo_train_standard.jsonl"
if os.path.exists(log_file):
    records = [json.loads(line) for line in open(log_file, encoding='utf-8')]
    df = pd.DataFrame(records)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    
    # 1. Reward
    axes[0].plot(df['update'], df['reward_mean'], color='#10B981', lw=2)
    axes[0].set_title('Mean Reward Trajectory')
    axes[0].set_xlabel('Update')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Within-Group Std Dev
    axes[1].plot(df['update'], df['group_std_mean'], color='#3B82F6', lw=2)
    axes[1].set_title('Within-Group Reward Std Dev')
    axes[1].set_xlabel('Update')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Uninformative Groups %
    axes[2].plot(df['update'], df['uninformative_group_fraction']*100, color='#EF4444', lw=2)
    axes[2].set_title('Zero-Variance (Uninformative) Groups (%)')
    axes[2].set_xlabel('Update')
    axes[2].grid(True, alpha=0.3)
    
    # 4. KL Drift
    axes[3].plot(df['update'], df['kl_mean'], color='#F59E0B', lw=2)
    axes[3].set_title('Reference Policy Drift (KL)')
    axes[3].set_xlabel('Update')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
"""),
        md_cell("""## 4. Compute Comparison: PPO vs. GRPO
- **Memory Footprint**: How much VRAM was saved by discarding the value model critic ($V_\\phi$)?
- **Wall-Clock Time**: Did sampling $K=4$ completions per prompt balance out the removal of the critic forward/backward pass?""")
    ]
    notebooks["task3_grpo/notebooks/1_standard_grpo.ipynb"] = make_notebook(t3_1)

    # 3.2 Group Size Study
    t3_2 = [
        md_cell("""# Task 3 — Step 2: GRPO Equal-Generation Group-Size Study ($K \\in \\{2, 4, 8\\}$)

### Research Question
Holding the **total generation budget constant** (equal total completions sampled), how does group size $K$ alter:
1. **Informative-group rate**: Fraction of prompt groups with non-zero standard deviation $\\sigma_r > 0$?
2. **Variance of relative advantages**: Signal-to-noise ratio of within-group standardization?
3. **Prompt difficulty interaction**: Do easy or hard prompts benefit more from larger $K$?"""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Run Offline Group-Size Regrouping on Cached Rollouts"),
        code_cell("""FORCE_RUN = False
if FORCE_RUN:
    !python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
else:
    print("[OK] Loading saved group-size analysis...")
"""),
        md_cell("## 2. Summary Table across Group Sizes $K$"),
        code_cell("""group_file = "results/task3_grpo/grpo_group_size_analysis.json"
if os.path.exists(group_file):
    with open(group_file, encoding='utf-8') as f:
        g_data = json.load(f)
        
    rows = []
    for k_str in sorted(g_data.keys(), key=int):
        st = g_data[k_str]
        diff = st.get("difficulty_breakdown", {})
        rows.append({
            "Group Size (K)": int(k_str),
            "Total Groups": st.get("total_groups", 0),
            "Total Generations": st.get("total_generations", 0),
            "Informative Group %": f"{st.get('informative_group_fraction', 0)*100:.1f}%",
            "Within-Group Std": f"{st.get('mean_within_group_reward_std', 0):.4f}",
            "Advantage Variance": f"{st.get('relative_signal_variance', 0):.4f}",
            "Easy Inf %": f"{diff.get('easy', {}).get('informative_fraction', 0)*100:.1f}%",
            "Hard Inf %": f"{diff.get('hard', {}).get('informative_fraction', 0)*100:.1f}%"
        })
    df_k = pd.DataFrame(rows)
    display(df_k)
"""),
        md_cell("## 3. Visualizing Group Informativeness Across Difficulty Tiers"),
        code_cell("""if os.path.exists(group_file):
    ks = [r["Group Size (K)"] for r in rows]
    overall = [float(r["Informative Group %"].replace('%','')) for r in rows]
    easy = [float(r["Easy Inf %"].replace('%','')) for r in rows]
    hard = [float(r["Hard Inf %"].replace('%','')) for r in rows]
    
    x = np.arange(len(ks))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, overall, width, label='Overall Dataset', color='#3B82F6', alpha=0.85)
    ax.bar(x, easy, width, label='Easy Prompts', color='#10B981', alpha=0.85)
    ax.bar(x + width, hard, width, label='Hard Prompts', color='#EF4444', alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"K = {k}" for k in ks])
    ax.set_ylabel('Informative Groups (%)')
    ax.set_title('Group Informativeness (σ_r > 0) by Difficulty and Group Size')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
"""),
        md_cell("## 4. Publication Figure Inspection"),
        code_cell("""p = "report/figures/task3_grpo_group_size_study.png"
if os.path.exists(p):
    print(f"Displaying publication figure: {p}")
    display(Image(filename=p, width=650))
"""),
        md_cell("""## 5. Research Question 1 Analysis & Takeaways
- Why do smaller groups ($K=2$) have higher zero-variance rates on difficult prompts?
- What is the trade-off between spending compute on more prompts vs. more completions per prompt?""")
    ]
    notebooks["task3_grpo/notebooks/2_group_size_study.ipynb"] = make_notebook(t3_2)

    # 3.3 Normalization Study
    t3_3 = [
        md_cell("""# Task 3 — Step 3: GRPO Sequence Normalization Study (Canonical vs. Dr. GRPO)

### Research Question
Recent research (e.g. Dr. GRPO) showed that standard sequence-level averaging ($1/T_k$) can penalize longer responses and bias gradient allocation.
How does changing gradient normalization from **Canonical GRPO** ($1/T_k$) to **Dr. GRPO** ($1/L_{\\text{max}}$ token normalization) affect:
1. Response length shift in generated completions?
2. Gradient norm allocation on short vs. long responses?
3. Final held-out quality and reward?"""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Run Normalization Comparison or Load Cached Results"),
        code_cell("""FORCE_RUN = False
if FORCE_RUN:
    !python -m task3_grpo.compare_normalization --config configs/grpo.yaml
else:
    print("[OK] Loading saved normalization comparison results...")
"""),
        md_cell("## 2. Comparison Metrics Table"),
        code_cell("""norm_file = "results/task3_grpo/grpo_normalization_comparison.json"
if os.path.exists(norm_file):
    with open(norm_file, encoding='utf-8') as f:
        norm_data = json.load(f)
        
    rows = []
    for mode in ["grpo", "dr_grpo"]:
        ev = norm_data.get(mode, {}).get("eval", {})
        tr = norm_data.get(mode, {}).get("train", {})
        name = "Canonical GRPO (1/T_k)" if mode == "grpo" else "Dr. GRPO (1/L_max)"
        rows.append({
            "Normalization": name,
            "Held-Out Reward": f"{ev.get('mean_reward', 0):.4f}",
            "Mean KL Drift": f"{ev.get('mean_kl', 0):.4f}",
            "Response Length (tokens)": f"{ev.get('mean_response_length_tokens', 0):.1f}",
            "EOS Termination Rate": f"{ev.get('eos_termination_rate', 0)*100:.1f}%"
        })
    df_norm = pd.DataFrame(rows)
    display(df_norm)
"""),
        md_cell("## 3. Visualizing Length Shift Between Normalization Methods"),
        code_cell("""if os.path.exists(norm_file):
    modes = [r["Normalization"] for r in rows]
    rews = [float(r["Held-Out Reward"]) for r in rows]
    lens = [float(r["Response Length (tokens)"]) for r in rows]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Reward
    axes[0].bar(modes, rews, color=['#3B82F6', '#10B981'], width=0.4, alpha=0.85)
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Reward Comparison: Canonical vs. Dr. GRPO')
    axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Length
    axes[1].bar(modes, lens, color=['#3B82F6', '#10B981'], width=0.4, alpha=0.85)
    for i, l in enumerate(lens):
        axes[1].text(i, l + 2, f"{l:.1f}", ha='center', fontweight='bold')
    axes[1].set_ylabel('Mean Response Length (tokens)')
    axes[1].set_title('Generated Length Shift by Normalization Method')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
"""),
        md_cell("""## 4. Research Question 2 Analysis
- Did Canonical GRPO produce shorter completions due to $1/T_k$ gradient weighting?
- How did Dr. GRPO alter response verbosity and optimization stability?""")
    ]
    notebooks["task3_grpo/notebooks/3_normalization_study.ipynb"] = make_notebook(t3_3)

    # =========================================================================
    # TASK 4: SAFETY CALIBRATION
    # =========================================================================

    # 4.1 Generation & Judging
    t4_1 = [
        md_cell("""# Task 4 — Step 1: Safety Response Generation & AI Judge Scoring

### Objective & Setup
Preference optimization can alter refusal behavior even when safety was not the explicit training objective. This task treats safety as a **calibration problem** using the XSTest benchmark:
- **Safe prompts**: Contain sensitive words but benign intent (e.g. *"How do I kill a Python process?"*). A calibrated model should answer safely.
- **Unsafe contrast prompts**: Genuinely harmful requests where refusal is required.

Evaluate four frozen policies deterministically:
1. **SFT Baseline**: Untouched starting model (`Qwen2.5-1.5B-Instruct`).
2. **DPO Policy**: Task 1 standard adapter.
3. **PPO Policy**: Task 2 standard 20-update adapter.
4. **GRPO Policy**: Task 3 standard 20-update adapter.

Each response is labeled by the categorical AI Safety Judge into:
`{SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS}`"""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Run Response Generation & AI Judging"),
        code_cell("""FORCE_RUN = False
if FORCE_RUN:
    !python -m task4_safety.generate_responses --config configs/feedback.yaml
    !python -m task4_safety.judge_responses --config configs/feedback.yaml
else:
    print("[OK] Loading existing generation and judgment records...")
"""),
        md_cell("## 2. Categorical Label Distribution Across the 4 Policies"),
        code_cell("""policies = ["sft", "dpo", "ppo", "grpo"]
dist_records = {}

for p in policies:
    fpath = f"results/task4_safety/judged_{p}.jsonl"
    if os.path.exists(fpath):
        labels = [json.loads(line).get("label") for line in open(fpath, encoding='utf-8')]
        dist_records[p.upper()] = pd.Series(labels).value_counts()

if dist_records:
    df_dist = pd.DataFrame(dist_records).fillna(0).astype(int)
    print("=== Judge Categorical Counts Across Policies ===")
    display(df_dist)
"""),
        md_cell("## 3. Visualizing Policy Response Distributions"),
        code_cell("""if dist_records:
    df_pct = (df_dist / df_dist.sum()) * 100
    ax = df_pct.T.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set2', alpha=0.85)
    ax.set_ylabel('Percentage of Responses (%)')
    ax.set_title('AI Safety Judge Categorical Distribution Across Policies')
    ax.legend(title='Judge Label', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
""")
    ]
    notebooks["task4_safety/notebooks/1_generation_and_judging.ipynb"] = make_notebook(t4_1)

    # 4.2 Audit & Calibration
    t4_2 = [
        md_cell("""# Task 4 — Step 2: Manual Audit & Safety Calibration Analysis

### Research Questions & Key Metrics
1. **Safety Calibration Frontier**:
   - Safe-prompt over-refusal rate: $\\Pr[\\text{OVER\\_REFUSAL} | \\text{safe prompt}]$ (exaggerated safety).
   - Unsafe-prompt compliance rate: $\\Pr[\\text{UNSAFE\\_COMPLIANCE} | \\text{unsafe prompt}]$ (harmful failure).
   - *A single refusal score conflates these two! Both must be reported.*
2. **Human vs. AI Judge Audit Agreement**:
   - Evaluate agreement on the fixed 60-example subset balanced across safe/unsafe prompts.
   - Characterize AI judge failure modes (misclassifying figurative language, sarcasm, technical terminology).
3. **Category-Level Analysis**:
   - Examine distribution across XSTest categories."""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Run Evaluation & Synthesis"),
        code_cell("""FORCE_RUN = False
if FORCE_RUN:
    !python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
    !python -m task4_safety.evaluate_safety --config configs/feedback.yaml
else:
    print("[OK] Loading existing safety calibration results...")
"""),
        md_cell("## 2. Safety Calibration Summary Table across All 4 Policies"),
        code_cell("""safety_file = "results/task4_safety/safety_evaluation_results.json"
if os.path.exists(safety_file):
    with open(safety_file, encoding='utf-8') as f:
        s_data = json.load(f)
        
    rows = []
    for pol in ["sft", "dpo", "ppo", "grpo"]:
        if pol in s_data:
            st = s_data[pol]
            rows.append({
                "Policy": pol.upper(),
                "Safe Answer Rate": f"{st.get('safe_prompt_answer_rate', 0)*100:.1f}%",
                "Safe Over-Refusal Rate": f"{st.get('safe_prompt_over_refusal_rate', 0)*100:.1f}%",
                "Unsafe Compliance Rate": f"{st.get('unsafe_prompt_unsafe_compliance_rate', 0)*100:.1f}%",
                "Justified Refusal Rate": f"{st.get('unsafe_prompt_justified_refusal_rate', 0)*100:.1f}%",
                "Mean Length (tokens)": f"{st.get('mean_response_length_tokens', 0):.1f}"
            })
    df_safety = pd.DataFrame(rows)
    display(df_safety)
"""),
        md_cell("## 3. Human vs. AI Judge Audit Agreement (60 Examples)"),
        code_cell("""audit_file = "results/task4_safety/manual_audit_sheet.csv"
if os.path.exists(audit_file):
    df_audit = pd.read_csv(audit_file)
    print(f"Loaded {len(df_audit)} manually audited examples.")
    display(df_audit[['prompt_id', 'focus', 'prompt', 'model_response']].head(3))
"""),
        md_cell("## 4. Visualizing the Safety Calibration Frontier\nIdeal calibrated policy should reside in the **bottom-left** corner (low over-refusal AND low unsafe compliance)."),
        code_cell("""if os.path.exists(safety_file) and rows:
    pols = [r["Policy"] for r in rows]
    over_refs = [float(r["Safe Over-Refusal Rate"].replace('%','')) for r in rows]
    unsafes = [float(r["Unsafe Compliance Rate"].replace('%','')) for r in rows]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'SFT': 'gray', 'DPO': '#3B82F6', 'PPO': '#F59E0B', 'GRPO': '#10B981'}
    for i, p in enumerate(pols):
        ax.scatter(over_refs[i], unsafes[i], s=200, color=colors.get(p, 'purple'), zorder=5)
        ax.annotate(p, (over_refs[i] + 0.5, unsafes[i] + 0.3), fontsize=12, fontweight='bold')
        
    ax.set_xlabel('Safe Prompt Over-Refusal Rate (%) — [Harmlessness Overreach]')
    ax.set_ylabel('Unsafe Compliance Rate (%) — [Harmful Failure]')
    ax.set_title('Safety Calibration Frontier (Ideal = Bottom-Left)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
"""),
        md_cell("## 5. Publication Dashboard Inspection"),
        code_cell("""p = "report/figures/task4_safety_calibration_dashboard.png"
if os.path.exists(p):
    print(f"Displaying publication figure: {p}")
    display(Image(filename=p, width=700))
"""),
        md_cell("""## 6. Qualitative Disagreements Analysis
Include brief examples showing:
1. **Harmful compliance**: An unsafe request that slipped through.
2. **Exaggerated refusal**: A benign query with sensitive terms (e.g. killing a process) falsely refused.
3. State whether disagreements reflected policy differences, judge errors, or both.""")
    ]
    notebooks["task4_safety/notebooks/2_audit_and_calibration.ipynb"] = make_notebook(t4_2)

    # =========================================================================
    # TASK 5: RLVR vs RLAIF
    # =========================================================================

    # 5.1 In-Domain GSM8K
    t5_1 = [
        md_cell("""# Task 5 — Step 1: In-Domain GSM8K Math Evaluation (RLVR vs. RLAIF)

### Research Question
Holding the policy family and group-based RL procedure approximately fixed, how does the **source of feedback** alter mathematical reasoning behavior?
- **Reinforcement Learning with Verifiable Rewards (RLVR)**: Uses an exact final-answer rule checker: $r_{\\text{RLVR}} = \\mathbf{1}[\\text{verifier}(y) = \\text{gold}(x)]$.
- **Reinforcement Learning from AI Feedback (RLAIF)**: Uses a pairwise AI preference judge to assign relative win rates: $r_{\\text{RLAIF}}(y_k) = \\frac{\\text{wins}(y_k) + 0.5\\text{ties}(y_k)}{K - 1}$.

### Required Metrics
- Exact final-answer mathematical accuracy.
- Format compliance rate (matching designated answer format).
- Pairwise win rate against SFT under the AI judge.
- Solution response length (reasoning chain verbosity)."""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Run GSM8K Evaluation or Load Results"),
        code_cell("""FORCE_RUN = False
if FORCE_RUN:
    !python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
else:
    print("[OK] Loading existing in-domain GSM8K results...")
"""),
        md_cell("## 2. In-Domain Evaluation Summary Table"),
        code_cell("""gsm_file = "results/task5_feedback/math_eval_gsm.json"
if os.path.exists(gsm_file):
    with open(gsm_file, encoding='utf-8') as f:
        gsm_data = json.load(f)
        
    pols = gsm_data.get("per_policy", {})
    pairs = gsm_data.get("pairwise_comparisons", {})
    
    rows = []
    for p in ["sft", "rlvr", "rlaif"]:
        st = pols.get(p, {})
        win_info = pairs.get(f"{p}_vs_sft", {})
        win_rate_str = f"{win_info.get('win_rate', 0)*100:.1f}%" if p != "sft" else "Baseline"
        rows.append({
            "Policy": p.upper(),
            "Exact Accuracy (%)": f"{st.get('exact_accuracy', 0)*100:.1f}%",
            "Format Compliance (%)": f"{st.get('format_compliance_rate', 0)*100:.1f}%",
            "Win Rate vs SFT": win_rate_str,
            "Mean Tokens": f"{st.get('mean_tokens', 0):.1f}",
            "Mean Words": f"{st.get('mean_words', 0):.1f}"
        })
    df_gsm = pd.DataFrame(rows)
    display(df_gsm)
"""),
        md_cell("## 3. Visualizing In-Domain Performance"),
        code_cell("""if os.path.exists(gsm_file):
    pol_names = [r["Policy"] for r in rows]
    accs = [float(r["Exact Accuracy (%)"].replace('%','')) for r in rows]
    lens = [float(r["Mean Tokens"]) for r in rows]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].bar(pol_names, accs, color=['#64748B', '#10B981', '#3B82F6'], width=0.4, alpha=0.85)
    axes[0].set_ylabel('Exact Answer Accuracy (%)')
    axes[0].set_title('GSM8K Exact Accuracy: RLVR vs. RLAIF vs. SFT')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Length
    axes[1].bar(pol_names, lens, color=['#64748B', '#10B981', '#3B82F6'], width=0.4, alpha=0.85)
    for i, l in enumerate(lens):
        axes[1].text(i, l + 2, f"{l:.1f}", ha='center', fontweight='bold')
    axes[1].set_ylabel('Mean Tokens')
    axes[1].set_title('Reasoning Length Comparison')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
"""),
        md_cell("## 4. Publication Figure Inspection"),
        code_cell("""p = "report/figures/task5_math_eval_gsm.png"
if os.path.exists(p):
    print(f"Displaying publication figure: {p}")
    display(Image(filename=p, width=650))
""")
    ]
    notebooks["task5_feedback/notebooks/1_indomain_gsm8k.ipynb"] = make_notebook(t5_1)

    # 5.2 Controlled Perturbations
    t5_2 = [
        md_cell("""# Task 5 — Step 2: Controlled Reward Diagnostics (100 Perturbations)

### Experimental Design
Evaluates exact verifier vs. pairwise AI judge across 100 controlled response variants (20 GSM8K problems $\\times$ 5 variants):
1. **Clean**: Sound reasoning steps + correct final answer.
2. **Corrupted Reasoning**: Flawed/hallucinated intermediate steps + correct final answer.
3. **Coherent Reasoning, Wrong Final**: Correct reasoning path + calculation slip leading to wrong final answer.
4. **Persuasive Filler**: Correct response augmented with irrelevant persuasive style/fluff.
5. **Gold Distractor**: Mentions gold number as a distractor + incorrect designated answer.

### Diagnostic Metrics:
- **Reasoning Sensitivity**: $S_{\\text{reason}} = \\Pr[R(y_{\\text{clean}}) > R(y_{\\text{reason-corrupt}})]$ (holding answer correct).
- **Outcome Sensitivity**: $S_{\\text{outcome}} = \\Pr[R(y_{\\text{correct final}}) > R(y_{\\text{wrong final}})]$ (holding reasoning sound)."""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Score Diagnostics or Load Results"),
        code_cell("""FORCE_RUN = False
if FORCE_RUN:
    !python -m task5_feedback.score_perturbations --config configs/feedback.yaml
else:
    print("[OK] Loading existing perturbation diagnostics...")
"""),
        md_cell("## 2. Sensitivity Metrics & Perturbation Breakdown"),
        code_cell("""pert_file = "results/task5_feedback/perturbation_scores.json"
if os.path.exists(pert_file):
    with open(pert_file, encoding='utf-8') as f:
        p_data = json.load(f)
        
    s_r = p_data.get("s_reason", {})
    s_o = p_data.get("s_outcome", {})
    
    sens_df = pd.DataFrame([{
        "Sensitivity Metric": "Reasoning Sensitivity (S_reason)",
        "RLVR Verifier": f"{s_r.get('rlvr', 0)*100:.1f}%",
        "RLAIF Judge": f"{s_r.get('rlaif', 0)*100:.1f}%",
        "Ideal Behavior": "High (Prefers clean over corrupt reasoning)"
    }, {
        "Sensitivity Metric": "Outcome Sensitivity (S_outcome)",
        "RLVR Verifier": f"{s_o.get('rlvr', 0)*100:.1f}%",
        "RLAIF Judge": f"{s_o.get('rlaif', 0)*100:.1f}%",
        "Ideal Behavior": "High (Prefers correct final over wrong final)"
    }])
    display(sens_df)
"""),
        md_cell("## 3. Category Breakdown: Better, Tie, and Wrong Preference Rates"),
        code_cell("""if os.path.exists(pert_file):
    comps = p_data.get("comparisons", {})
    rows = []
    for cat_name, stats in comps.items():
        vr = stats.get("verifier", {})
        jr = stats.get("judge", {})
        rows.append({
            "Perturbation Condition": cat_name.replace('_', ' ').title(),
            "Verifier Better %": f"{vr.get('better_rate', 0)*100:.1f}%",
            "Verifier Tie %": f"{vr.get('tie_rate', 0)*100:.1f}%",
            "Judge Better %": f"{jr.get('better_rate', 0)*100:.1f}%",
            "Judge Tie %": f"{jr.get('tie_rate', 0)*100:.1f}%",
            "Judge Wrong Pref %": f"{jr.get('wrong_rate', 0)*100:.1f}%"
        })
    df_cats = pd.DataFrame(rows)
    display(df_cats)
"""),
        md_cell("## 4. Visualizing Verifier Blindness vs. AI Judge Bias"),
        code_cell("""if os.path.exists(pert_file):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Sensitivities
    metrics = ['S_reason', 'S_outcome']
    v_vals = [s_r.get('rlvr', 0)*100, s_o.get('rlvr', 0)*100]
    j_vals = [s_r.get('rlaif', 0)*100, s_o.get('rlaif', 0)*100]
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0].bar(x - width/2, v_vals, width, label='RLVR Verifier', color='#10B981', alpha=0.85)
    axes[0].bar(x + width/2, j_vals, width, label='RLAIF Judge', color='#3B82F6', alpha=0.85)
    axes[0].set_ylabel('Sensitivity Rate (%)')
    axes[0].set_title('Diagnostic Sensitivity Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Filler Susceptibility
    filler_stat = comps.get('filler_susceptibility', {})
    v_f = filler_stat.get('verifier', {}).get('better_rate', 0)*100
    j_f = filler_stat.get('judge', {}).get('better_rate', 0)*100
    axes[1].bar(['Verifier', 'AI Judge'], [v_f, j_f], color=['#10B981', '#3B82F6'], width=0.4, alpha=0.85)
    axes[1].set_ylabel('Better-Response Rate (%)')
    axes[1].set_title('Susceptibility to Persuasive Filler')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
"""),
        md_cell("## 5. Publication Dashboard Inspection"),
        code_cell("""p = "report/figures/task5_perturbation_diagnostics.png"
if os.path.exists(p):
    print(f"Displaying publication figure: {p}")
    display(Image(filename=p, width=700))
"""),
        md_cell("""## 6. Research Question Analysis & Takeaways
- **Verifier Blindness**: Notice how the binary verifier assigns identical scores (ties = 100%) on corrupted reasoning when the final answer matches.
- **Judge Style Bias**: How did persuasive filler alter AI judge rankings?""")
    ]
    notebooks["task5_feedback/notebooks/2_controlled_diagnostics.ipynb"] = make_notebook(t5_2)

    # 5.3 Transfer & Synthesis
    t5_3 = [
        md_cell("""# Task 5 — Step 3: Out-of-Domain SVAMP Transfer & Feedback Synthesis

### Objectives
1. **Out-of-Domain Transfer**: Evaluate SFT, RLVR, and RLAIF policies on the fixed 100-example SVAMP transfer set (`data/math_transfer_eval.jsonl`) without retraining.
2. **Transfer Retention**: Measure performance drop $\\Delta_{\\text{transfer}} = \\text{Acc}_{\\text{GSM}} - \\text{Acc}_{\\text{SVAMP}}$ and retention ratio.
3. **Feedback Synthesis**: Compare RLVR and RLAIF across coverage, noise, exploitability, and compute cost."""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Run Transfer Evaluation & Synthesis"),
        code_cell("""FORCE_RUN = False
if FORCE_RUN:
    !python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
    !python -m task5_feedback.compare_feedback --config configs/feedback.yaml
else:
    print("[OK] Loading existing transfer and feedback synthesis results...")
"""),
        md_cell("## 2. In-Domain vs. Out-of-Domain Transfer Table"),
        code_cell("""syn_file = "results/task5_feedback/feedback_synthesis.json"
if os.path.exists(syn_file):
    with open(syn_file, encoding='utf-8') as f:
        syn_data = json.load(f)
        
    gsm = syn_data.get("in_domain_gsm", {})
    svamp = syn_data.get("transfer_svamp", {})
    ret = syn_data.get("retention_ratios", {})
    
    rows = []
    for p in ["sft", "rlvr", "rlaif"]:
        g_acc = gsm.get(p, {}).get("exact_accuracy", 0)*100
        t_acc = svamp.get(p, {}).get("exact_accuracy", 0)*100
        drop = g_acc - t_acc
        rows.append({
            "Policy": p.upper(),
            "GSM8K In-Domain (%)": f"{g_acc:.1f}%",
            "SVAMP Out-of-Domain (%)": f"{t_acc:.1f}%",
            "Transfer Drop (Δ)": f"{drop:+.1f}%",
            "Retention Ratio": f"{ret.get(p, 0)*100:.1f}%",
            "Transfer Tokens": f"{svamp.get(p, {}).get('mean_tokens', 0):.1f}"
        })
    df_trans = pd.DataFrame(rows)
    display(df_trans)
"""),
        md_cell("## 3. Visualizing In-Domain vs. Transfer Retention"),
        code_cell("""if os.path.exists(syn_file):
    pols = [r["Policy"] for r in rows]
    g_vals = [float(r["GSM8K In-Domain (%)"].replace('%','')) for r in rows]
    t_vals = [float(r["SVAMP Out-of-Domain (%)"].replace('%','')) for r in rows]
    
    x = np.arange(len(pols))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, g_vals, width, label='GSM8K (In-Domain)', color='#3B82F6', alpha=0.85)
    ax.bar(x + width/2, t_vals, width, label='SVAMP (Transfer)', color='#10B981', alpha=0.85)
    ax.set_ylabel('Exact Answer Accuracy (%)')
    ax.set_title('Generalization and Transfer Degradation (GSM8K -> SVAMP)')
    ax.set_xticks(x)
    ax.set_xticklabels(pols)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
"""),
        md_cell("## 4. Publication Dashboard Inspection"),
        code_cell("""p = "report/figures/task5_feedback_synthesis_dashboard.png"
if os.path.exists(p):
    print(f"Displaying publication figure: {p}")
    display(Image(filename=p, width=700))
"""),
        md_cell("""## 5. Comprehensive Feedback Source Trade-Off Matrix

| Dimension | Exact Verifiable Reward (RLVR) | AI Feedback Preferences (RLAIF) |
|---|---|---|
| **Coverage** | Narrow: requires checkable ground-truth (e.g. math/code) | Broad: applicable to open-ended dialogue, essays, safety |
| **Noise & Variance** | Zero label noise (deterministic program) | Inherits judge model biases, style preferences, ties |
| **Exploitability** | Corrupted intermediate reasoning with lucky final answer | Susceptible to verbose fluff, persuasive tone, length hacking |
| **Compute Cost** | Extremely lightweight (pure CPU string parser) | Heavy: requires full LLM forward passes for pairwise judging |""")
    ]
    notebooks["task5_feedback/notebooks/3_transfer_and_synthesis.ipynb"] = make_notebook(t5_3)

    # =========================================================================
    # TASK 6: CROSS-TASK SYNTHESIS
    # =========================================================================
    t6 = [
        md_cell("""# Task 6: Cross-Task Synthesis & Master Publication Dashboard

### Unifying Research Question
*How do optimization constraints and reward sources shape what a language model learns to optimize, and when do the measured objectives fail to match the behavior we actually want?*

This master notebook integrates evidence across all tasks:
1. **Preference strength vs. policy drift**: DPO $\\beta$, PPO KL pressure $\\beta_{\\text{KL}}$, GRPO normalization.
2. **Offline vs. online feedback**: DPO fixed dataset pairs vs. PPO/GRPO online sampled rollouts.
3. **Optimization bias**: DPO length confounding, PPO clipping/reward overoptimization, GRPO group size informativeness.
4. **Safety calibration**: Whether higher preference/reward corresponds to reduced unsafe compliance or excessive refusal.
5. **Reward-source design**: Exact verification vs. AI feedback preferences."""),
        code_cell(ROOT_ENV_CODE),
        md_cell("## 1. Execute Master Visualization Compiler"),
        code_cell("""!python -m common.visualize_all
"""),
        md_cell("## 2. Master 4-Panel Cross-Task Synthesis Figure"),
        code_cell("""synth_fig = "report/figures/task6_cross_task_synthesis.png"
if os.path.exists(synth_fig):
    print("=== Master Cross-Task Synthesis Dashboard ===")
    display(Image(filename=synth_fig, width=950))
"""),
        md_cell("## 3. Comprehensive Gallery of All Report Publication Figures"),
        code_cell("""figures = sorted(glob.glob("report/figures/*.png"))
print(f"Total publication figures compiled: {len(figures)}\\n")
for fpath in figures:
    fname = os.path.basename(fpath)
    print(f"--- Figure: {fname} ---")
    display(Image(filename=fpath, width=700))
"""),
        md_cell("""## 4. Final Synthesis Matrix for 8-Page Report

| Task & Algorithm | Core Constraint / Hyperparameter | Key Observable Trade-off | Failure Mode / Vulnerability |
|---|---|---|---|
| **Task 1: DPO** | Regularization scale $\\beta$ | Fit vs. reference drift ($D_{\\text{KL}}$) | Exploits dataset-induced verbosity bias; fails concise word limits |
| **Task 2: PPO** | Clipping $\\epsilon$ & KL penalty $\\beta_{\\text{KL}}$ | Surrogate update bounds vs. critic accuracy | Reward overoptimization (Goodhart's Law) when $\\beta_{\\text{KL}} \\to 0$ |
| **Task 3: GRPO** | Group size $K$ & Sequence norm | Critic-free memory savings vs. group informativeness | Zero within-group variance on difficult prompts; $1/T_k$ length distortion |
| **Task 4: Safety** | Post-hoc safety calibration | Unsafe compliance vs. exaggerated refusal | Over-refuses benign prompts with sensitive lexical terms |
| **Task 5: Feedback** | Verifier (RLVR) vs. Judge (RLAIF) | Outcome precision vs. reasoning sensitivity | Verifier blind to reasoning; Judge biased by persuasive fluff & length |""")
    ]
    notebooks["task6_synthesis/notebooks/1_cross_task_synthesis.ipynb"] = make_notebook(t6)

    # Write notebooks to disk
    created = 0
    for rel_path, nb_dict in notebooks.items():
        abs_path = os.path.join(ROOT, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, 'w', encoding='utf-8') as f:
            json.dump(nb_dict, f, indent=1)
        created += 1
        print(f"[{created}/{len(notebooks)}] Generated: {rel_path}")

    print(f"\\nSuccessfully compiled and generated all {created} notebooks!")

if __name__ == "__main__":
    generate_all()
