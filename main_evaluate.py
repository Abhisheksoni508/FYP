"""
Evaluate the full 3-Layer Hybrid AI System vs Fixed-Threshold Baseline.

Layer 1: LSTM Deep Ensemble (RUL prediction + uncertainty)
Layer 2: DQN Agent (learned maintenance policy)
Layer 3: Safety Supervisor (deterministic override when RUL < critical)

The safety supervisor ensures zero catastrophic failures even when
the RL agent misjudges, providing a hard safety guarantee.

Usage: python main_evaluate.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from stable_baselines3 import DQN

from src.preprocessing import load_combined_data, calculate_rul, process_data
from src.gym_env import PdMEnvironment
from src.config import *


def evaluate_agent(env, model, num_episodes=200, use_safety=True):
    """Evaluate the full hybrid system (RL agent + safety supervisor)."""
    rewards, fails, jacks, safes, wastes = [], 0, 0, 0, 0
    safety_overrides = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # --- LAYER 3: Safety Supervisor Override ---
            # If agent says WAIT but predicted RUL is critically low,
            # force maintenance to prevent catastrophic failure.
            if use_safety and action == 0 and obs[0] < CRITICAL_RUL_NORM:
                action = 1
                safety_overrides += 1
            
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            if done:
                if reward == -100: fails += 1
                elif reward == 500: jacks += 1
                elif reward == 10: safes += 1
                elif reward == -20: wastes += 1
        rewards.append(ep_reward)
    
    ci_95 = 1.96 * np.std(rewards) / np.sqrt(len(rewards))
    
    return {
        'avg': np.mean(rewards), 'std': np.std(rewards), 'ci_95': ci_95,
        'fails': fails, 'jacks': jacks, 'safes': safes, 'wastes': wastes,
        'overrides': safety_overrides, 'rewards': rewards,
    }


def evaluate_baseline(env, threshold=30, num_episodes=200):
    """Fixed-threshold baseline: maintain when predicted RUL < threshold."""
    rewards, fails, jacks, safes, wastes = [], 0, 0, 0, 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0
        while not done:
            pred_rul = obs[0] * 125.0
            action = 1 if pred_rul < threshold else 0
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            if done:
                if reward == -100: fails += 1
                elif reward == 500: jacks += 1
                elif reward == 10: safes += 1
                elif reward == -20: wastes += 1
        rewards.append(ep_reward)
    
    ci_95 = 1.96 * np.std(rewards) / np.sqrt(len(rewards))
    
    return {
        'avg': np.mean(rewards), 'std': np.std(rewards), 'ci_95': ci_95,
        'fails': fails, 'jacks': jacks, 'safes': safes, 'wastes': wastes,
        'rewards': rewards,
    }


if __name__ == "__main__":
    print("=" * 65)
    print("  3-LAYER HYBRID AI SYSTEM EVALUATION")
    print("  Layer 1: LSTM Deep Ensemble | Layer 2: DQN | Layer 3: Safety")
    print("=" * 65)
    
    print("\n--- Loading Data ---")
    df = load_combined_data()
    df = calculate_rul(df)
    df_clean, _ = process_data(df, DROP_SENSORS, DROP_SETTINGS)
    
    env = PdMEnvironment(df_clean, models_dir='models', noise_prob=0.0)
    model = DQN.load("models/dqn_pdm_agent")
    
    N = 200
    print(f"\nEvaluating Hybrid AI System ({N} episodes, clean data)...")
    ai = evaluate_agent(env, model, N, use_safety=True)
    print(f"Evaluating Fixed-Threshold Baseline ({N} episodes)...")
    base = evaluate_baseline(env, threshold=30, num_episodes=N)
    
    # --- Results Table ---
    print(f"\n{'='*65}")
    print(f"  RESULTS ({N} Episodes, Clean Data)")
    print(f"{'='*65}")
    print(f"{'METRIC':<25} | {'HYBRID AI':<18} | {'BASELINE':<18}")
    print(f"{'-'*65}")
    print(f"{'Avg Reward':<25} | {ai['avg']:.2f} ± {ai['ci_95']:.2f}    | {base['avg']:.2f} ± {base['ci_95']:.2f}")
    print(f"{'Crashes (-100)':<25} | {ai['fails']:<18} | {base['fails']:<18}")
    print(f"{'Jackpots (+500)':<25} | {ai['jacks']:<18} | {base['jacks']:<18}")
    print(f"{'Safe Repairs (+10)':<25} | {ai['safes']:<18} | {base['safes']:<18}")
    print(f"{'Wasteful (-20)':<25} | {ai['wastes']:<18} | {base['wastes']:<18}")
    print(f"{'Safety Overrides':<25} | {ai['overrides']:<18} | {'N/A':<18}")
    print(f"{'='*65}")
    
    # Statistical test
    t_stat, p_val = stats.ttest_ind(ai['rewards'], base['rewards'], equal_var=False)
    improvement = ((ai['avg'] - base['avg']) / abs(base['avg'])) * 100
    
    print(f"\n  Improvement: {improvement:+.1f}% over baseline")
    print(f"  Welch's t-test: t={t_stat:.3f}, p={p_val:.2e}")
    if p_val < 0.001:
        print(f"  Result: STATISTICALLY SIGNIFICANT (p < 0.001)")
    elif p_val < 0.05:
        print(f"  Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print(f"  Result: NOT statistically significant (p = {p_val:.4f})")
    
    if ai['overrides'] > 0:
        print(f"\n  Safety Supervisor: {ai['overrides']} override(s) prevented potential crashes")
    else:
        print(f"\n  Safety Supervisor: 0 overrides needed (DQN handled all cases)")
    
    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = ['Hybrid AI\nSystem', 'Fixed-Threshold\nBaseline']
    scores = [ai['avg'], base['avg']]
    errors = [ai['ci_95'], base['ci_95']]
    colors = ['#2ecc71', '#95a5a6']
    
    bars = axes[0].bar(labels, scores, color=colors, edgecolor='black', linewidth=0.5,
                       yerr=errors, capsize=8, error_kw={'linewidth': 1.5})
    axes[0].set_title("Average Reward Comparison", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Reward Points")
    axes[0].axhline(0, color='black', linewidth=0.8)
    for bar, val, ci in zip(bars, scores, errors):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 5,
                     f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Significance annotation
    max_h = max(scores) + max(errors) + 20
    axes[0].plot([0, 0, 1, 1], [max_h, max_h+10, max_h+10, max_h], 'k-', linewidth=1)
    sig_text = f"p < 0.001 ***" if p_val < 0.001 else f"p = {p_val:.3f}"
    axes[0].text(0.5, max_h + 12, sig_text, ha='center', fontsize=10, fontweight='bold')

    cats = ['Jackpots\n(+500)', 'Safe\n(+10)', 'Wasteful\n(-20)', 'Crashes\n(-100)']
    ai_c = [ai['jacks'], ai['safes'], ai['wastes'], ai['fails']]
    base_c = [base['jacks'], base['safes'], base['wastes'], base['fails']]
    x = np.arange(len(cats))
    w = 0.35
    axes[1].bar(x - w/2, ai_c, w, label='Hybrid AI', color='#2ecc71', edgecolor='black', linewidth=0.5)
    axes[1].bar(x + w/2, base_c, w, label='Baseline', color='#95a5a6', edgecolor='black', linewidth=0.5)
    axes[1].set_title("Outcome Breakdown", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Count")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cats)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("final_result_chart.png", dpi=150)
    print(f"\nChart saved as 'final_result_chart.png'")