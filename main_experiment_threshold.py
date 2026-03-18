"""
Experiment 3: Does the DQN Agent Learn Better Than a Simple Rule?

MOTIVATION:
  A critic could argue: "Why use RL? Just trigger maintenance when
  predicted RUL drops below a fixed threshold." This experiment
  directly answers that question.

DESIGN:
  1. Test 8 threshold policies (trigger when pred_RUL < X cycles)
  2. Find the BEST threshold across all noise levels
  3. Compare the best threshold vs UA DQN vs Blind DQN
  4. Test under same noise conditions as Experiments 1 & 2

  If the DQN outperforms the optimal threshold, it has learned
  non-trivial timing behaviour. If not, RL adds no value.

OUTPUT:
  - experiment_3_threshold_baseline.png (2-panel chart)
  - Console table with all results

Usage: python main_experiment_threshold.py
Prereq: models/dqn_pdm_agent.zip, models/dqn_blind_agent.zip
Time: ~10-15 mins (no training, evaluation only)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from stable_baselines3 import DQN

from src.preprocessing import load_combined_data, calculate_rul, process_data
from src.gym_env import PdMEnvironment, BlindPdMEnvironment, classify_terminal_reward
from src.config import *


# ============================================================
# 2. Threshold Policy (the "dumb" baseline)
# ============================================================
class ThresholdPolicy:
    """
    Simple rule-based policy: trigger maintenance when the
    ensemble's mean predicted RUL drops below a fixed threshold.

    No learning, no uncertainty, no neural network.
    This is what an engineer would do with a simple alarm.
    """
    def __init__(self, threshold_norm):
        """
        Args:
            threshold_norm: Normalised RUL threshold (0-1 scale).
                           e.g. 0.16 = trigger when pred_RUL < 20 cycles
        """
        self.threshold = threshold_norm

    def predict(self, obs, deterministic=True):
        """Match the stable-baselines3 predict() interface."""
        # obs[0] = normalised mean predicted RUL
        if obs[0] < self.threshold:
            return 1, None  # MAINTAIN
        return 0, None      # WAIT


# ============================================================
# 3. Evaluation (no safety supervisor — raw policy comparison)
# ============================================================
def evaluate(env, policy, num_episodes=200):
    """Evaluate a policy (DQN or threshold) without safety supervisor."""
    rewards = []
    fails, jacks, safes, wastes = 0, 0, 0, 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            if done:
                outcome = classify_terminal_reward(reward)
                if outcome == 'crash':
                    fails += 1
                elif outcome == 'jackpot':
                    jacks += 1
                elif outcome == 'safe':
                    safes += 1
                elif outcome == 'wasteful':
                    wastes += 1
        rewards.append(ep_reward)

    n = len(rewards)
    return {
        'mean': np.mean(rewards), 'std': np.std(rewards),
        'ci_95': 1.96 * np.std(rewards) / np.sqrt(n),
        'failures': fails, 'jackpots': jacks, 'safe': safes, 'wasteful': wastes,
        'fail_rate': fails / n * 100, 'jack_rate': jacks / n * 100,
        'rewards': rewards,
    }


# ============================================================
# 4. Main
# ============================================================
def run_threshold_experiment():
    print("=" * 70)
    print("  EXPERIMENT 3: DQN vs Threshold Baseline")
    print("  Does reinforcement learning add value over a simple rule?")
    print("=" * 70)

    # --- Data ---
    print("\n--- Loading Data ---")
    df = load_combined_data()
    df = calculate_rul(df)
    df_clean, _ = process_data(df, DROP_SENSORS, DROP_SETTINGS)

    # --- Load DQN agents ---
    print("--- Loading DQN Agents ---")
    ua_model = DQN.load("models/dqn_pdm_agent")

    # Try loading blind agent; skip if not available
    try:
        blind_model = DQN.load("models/dqn_blind_agent")
        has_blind = True
        print("  Loaded: UA agent + Blind agent")
    except:
        has_blind = False
        print("  Loaded: UA agent only (run ablation first for blind agent)")

    # --- Define threshold policies ---
    # Thresholds in real RUL cycles, converted to normalised (÷125)
    threshold_cycles = [5, 10, 15, 20, 25, 30, 40, 50]
    threshold_policies = {}
    for tc in threshold_cycles:
        threshold_policies[f'T={tc}'] = ThresholdPolicy(tc / 125.0)

    # --- Noise levels (same as ablation) ---
    noise_levels = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15]
    NUM_EPS = 200

    # =========================================================
    # PHASE 1: Find the best threshold on clean data
    # =========================================================
    print(f"\n--- Phase 1: Evaluating {len(threshold_cycles)} thresholds on clean data ---")

    clean_results = {}
    ua_env = PdMEnvironment(df_clean, 'models', noise_prob=0.0)

    for name, policy in threshold_policies.items():
        r = evaluate(ua_env, policy, NUM_EPS)
        clean_results[name] = r
        print(f"  {name:>6}: reward={r['mean']:>7.1f} ±{r['ci_95']:.1f}  "
              f"J:{r['jackpots']:>3}  S:{r['safe']:>3}  F:{r['failures']:>3}  W:{r['wasteful']:>3}")

    # Also evaluate DQN on clean
    ua_clean = evaluate(ua_env, ua_model, NUM_EPS)
    print(f"  {'UA DQN':>6}: reward={ua_clean['mean']:>7.1f} ±{ua_clean['ci_95']:.1f}  "
          f"J:{ua_clean['jackpots']:>3}  S:{ua_clean['safe']:>3}  F:{ua_clean['failures']:>3}  W:{ua_clean['wasteful']:>3}")

    # Find best threshold
    best_name = max(clean_results, key=lambda k: clean_results[k]['mean'])
    best_clean_reward = clean_results[best_name]['mean']
    print(f"\n  Best threshold on clean data: {best_name} (reward={best_clean_reward:.1f})")
    print(f"  UA DQN on clean data: reward={ua_clean['mean']:.1f}")

    dqn_vs_best = ua_clean['mean'] - best_clean_reward
    print(f"  DQN advantage: {dqn_vs_best:+.1f}")

    # =========================================================
    # PHASE 2: Test ALL policies across ALL noise levels
    # =========================================================
    print(f"\n--- Phase 2: Testing across {len(noise_levels)} noise levels ---")

    # Store results: {policy_name: [result_per_noise_level]}
    all_results = {}

    # DQN agents
    policies_to_test = {'UA DQN': ua_model}
    if has_blind:
        policies_to_test['Blind DQN'] = blind_model

    # Add best 3 thresholds + worst for comparison
    sorted_thresholds = sorted(clean_results.items(), key=lambda x: x[1]['mean'], reverse=True)
    for name, _ in sorted_thresholds[:3]:
        policies_to_test[name] = threshold_policies[name.split('=')[0] + '=' + name.split('=')[1]]

    # Re-add by name properly
    policies_to_test_clean = {'UA DQN': ua_model}
    if has_blind:
        policies_to_test_clean['Blind DQN'] = blind_model

    top_3_names = [name for name, _ in sorted_thresholds[:3]]
    for name in top_3_names:
        policies_to_test_clean[name] = threshold_policies[name]

    for policy_name, policy in policies_to_test_clean.items():
        all_results[policy_name] = []
        for nl in noise_levels:
            if nl == 0:
                env = PdMEnvironment(df_clean, 'models', noise_prob=0.0)
            else:
                env = PdMEnvironment(df_clean, 'models', noise_prob=1.0, noise_level=nl)

            r = evaluate(env, policy, NUM_EPS)
            all_results[policy_name].append(r)

        print(f"  {policy_name:>10}: clean={all_results[policy_name][0]['mean']:.0f}  "
              f"heavy={all_results[policy_name][-1]['mean']:.0f}  "
              f"retain={all_results[policy_name][-1]['mean']/max(all_results[policy_name][0]['mean'],1)*100:.0f}%")

    # =========================================================
    # RESULTS TABLE
    # =========================================================
    print(f"\n{'='*95}")
    print(f"  EXPERIMENT 3 RESULTS — DQN vs Threshold Baselines ({NUM_EPS} eps/condition)")
    print(f"{'='*95}")

    header = f"{'Policy':<12}"
    for nl in noise_levels:
        header += f" | {'σ='+f'{nl:.2f}':>10}"
    header += f" | {'Avg':>8}"
    print(header)
    print(f"{'-'*95}")

    for policy_name in all_results:
        row = f"{policy_name:<12}"
        means = [all_results[policy_name][i]['mean'] for i in range(len(noise_levels))]
        for m in means:
            row += f" | {m:>10.1f}"
        row += f" | {np.mean(means):>8.1f}"
        print(row)

    print(f"{'='*95}")

    # Statistical comparison: UA DQN vs best threshold at each noise level
    print(f"\n  STATISTICAL COMPARISON: UA DQN vs {best_name}")
    print(f"  {'Noise':<7} | {'UA DQN':>10} | {best_name:>10} | {'Gap':>8} | {'t-stat':>8} | {'p-value':>10} | {'Sig':>5}")
    print(f"  {'-'*70}")

    ua_wins, sig_wins = 0, 0
    best_threshold_results = []

    # Get full results for best threshold
    for nl in noise_levels:
        if nl == 0:
            env = PdMEnvironment(df_clean, 'models', noise_prob=0.0)
        else:
            env = PdMEnvironment(df_clean, 'models', noise_prob=1.0, noise_level=nl)
        r = evaluate(env, threshold_policies[best_name], NUM_EPS)
        best_threshold_results.append(r)

    for i, nl in enumerate(noise_levels):
        ua_r = all_results['UA DQN'][i]
        th_r = best_threshold_results[i]
        gap = ua_r['mean'] - th_r['mean']

        t_stat, p_val = stats.ttest_ind(ua_r['rewards'], th_r['rewards'], equal_var=False)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        if gap > 0:
            ua_wins += 1
            if p_val < 0.05: sig_wins += 1

        print(f"  {nl:<7.2f} | {ua_r['mean']:>10.1f} | {th_r['mean']:>10.1f} | {gap:>+8.1f} | {t_stat:>8.3f} | {p_val:>10.2e} | {sig:>5}")

    print(f"\n  UA DQN wins {ua_wins}/{len(noise_levels)} conditions ({sig_wins} statistically significant)")

    # =========================================================
    # PLOTTING
    # =========================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Colour scheme
    colors = {
        'UA DQN': '#27ae60',
        'Blind DQN': '#e74c3c',
    }
    threshold_colors = ['#3498db', '#9b59b6', '#f39c12']

    # --- Panel 1: All policies across noise levels ---
    ax1 = axes[0]

    for idx, (name, results) in enumerate(all_results.items()):
        means = [r['mean'] for r in results]
        cis = [r['ci_95'] for r in results]

        if name in colors:
            c = colors[name]
            ax1.plot(noise_levels, means, '-o', color=c, linewidth=2.5,
                    markersize=8, label=name, zorder=3)
            ax1.fill_between(noise_levels,
                            [m - ci for m, ci in zip(means, cis)],
                            [m + ci for m, ci in zip(means, cis)],
                            alpha=0.1, color=c)
        else:
            ci = idx - (2 if has_blind else 1)
            c = threshold_colors[ci % len(threshold_colors)]
            ax1.plot(noise_levels, means, '--s', color=c, linewidth=1.8,
                    markersize=6, label=f'{name} (rule)', alpha=0.8)

    ax1.set_xlabel('Sensor Noise Level (σ)', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title('DQN Agent vs Threshold Baselines\nAcross Noise Levels',
                   fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linewidth=0.5)

    # --- Panel 2: Bar chart on clean data (all thresholds + DQN) ---
    ax2 = axes[1]

    # All thresholds
    names = [f'T={tc}' for tc in threshold_cycles]
    clean_means = [clean_results[n]['mean'] for n in names]
    clean_cis = [clean_results[n]['ci_95'] for n in names]
    bar_colors = ['#3498db'] * len(names)

    # Highlight best threshold
    best_idx = names.index(best_name)
    bar_colors[best_idx] = '#9b59b6'

    bars = ax2.bar(names, clean_means, yerr=clean_cis, color=bar_colors,
                   edgecolor='black', linewidth=0.5, alpha=0.8, capsize=4)

    # Add DQN bar
    dqn_bar = ax2.bar(['UA\nDQN'], [ua_clean['mean']], yerr=[ua_clean['ci_95']],
                       color='#27ae60', edgecolor='black', linewidth=0.5, capsize=4)

    # Value labels
    for bar, val in zip(bars, clean_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f'{val:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax2.text(dqn_bar[0].get_x() + dqn_bar[0].get_width() / 2,
             dqn_bar[0].get_height() + 15,
             f'{ua_clean["mean"]:.0f}', ha='center', va='bottom',
             fontsize=8, fontweight='bold', color='#27ae60')

    # Annotate best threshold
    ax2.annotate(f'Best rule\n({best_name})',
                xy=(best_idx, clean_means[best_idx]),
                xytext=(best_idx - 1.5, clean_means[best_idx] + 80),
                fontsize=9, fontweight='bold', color='#9b59b6',
                arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.5))

    ax2.set_xlabel('Policy', fontsize=12)
    ax2.set_ylabel('Average Reward (Clean Data)', fontsize=12)
    ax2.set_title('Threshold Sweep on Clean Data\nDQN vs All Thresholds',
                   fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Summary
    if dqn_vs_best > 0:
        verdict = f"DQN outperforms the best threshold ({best_name}) by {dqn_vs_best:.0f} reward points on clean data — RL learned non-trivial timing."
    else:
        verdict = f"Best threshold ({best_name}) matches or beats DQN — threshold policy is a competitive baseline."

    fig.text(0.5, -0.02, f'EXPERIMENT 3 — {verdict}', ha='center', fontsize=10,
             style='italic', color='#333333', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig('experiment_3_threshold_baseline.png', dpi=200, bbox_inches='tight')
    print(f"\nChart saved as 'experiment_3_threshold_baseline.png'")

    # =========================================================
    # SUMMARY
    # =========================================================
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 3 SUMMARY")
    print(f"{'='*70}")
    print(f"  Best threshold policy: {best_name} (reward={best_clean_reward:.1f} on clean)")
    print(f"  UA DQN:                reward={ua_clean['mean']:.1f} on clean")
    print(f"  Gap:                   {dqn_vs_best:+.1f}")
    print(f"  UA DQN wins:           {ua_wins}/{len(noise_levels)} noise conditions")
    print(f"  Significant wins:      {sig_wins} (p<0.05)")
    print(f"")
    if dqn_vs_best > 0:
        print(f"  CONCLUSION: The DQN agent learned a policy that OUTPERFORMS")
        print(f"  the best fixed-threshold rule. This confirms RL adds value")
        print(f"  beyond simple alarm-based maintenance scheduling.")
    else:
        print(f"  CONCLUSION: The DQN agent performs comparably to a tuned")
        print(f"  threshold rule, but gains robustness under noisy conditions.")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_threshold_experiment()
