"""
PPO vs DQN Algorithm Comparison — Supplementary Experiment.

Trains a PPO agent on the same environment and compares against
the existing DQN agent across noise levels. This validates that
DQN was an appropriate algorithm choice for the binary action space.

Usage:  python main_experiment_ppo.py
Prereq: models/dqn_pdm_agent.zip, ensemble models in models/
Output: report_ppo_vs_dqn.png, report_ppo_comparison.csv
Time:   ~30-40 mins (PPO training + evaluation)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from src.preprocessing import load_combined_data, calculate_rul, process_data
from src.gym_env import PdMEnvironment, safety_override, classify_terminal_reward
from src.config import *

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Config ───────────────────────────────────────────────────
PPO_TIMESTEPS = 200_000
NUM_EPISODES = 300
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.15]

# ── Plot Style ───────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
DQN_COLOR = '#2e86de'
PPO_COLOR = '#10ac84'


def evaluate_agent(env, model, num_episodes, use_safety=False):
    """Evaluate an agent with optional safety supervisor."""
    rewards = []
    fails, jacks, safes, wastes = 0, 0, 0, 0
    overrides = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if use_safety:
                action, was_overridden = safety_override(action, obs)
                if was_overridden:
                    overrides += 1
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            if done:
                outcome = classify_terminal_reward(reward)
                if outcome == 'crash':    fails += 1
                elif outcome == 'jackpot': jacks += 1
                elif outcome == 'safe':    safes += 1
                elif outcome == 'wasteful': wastes += 1
        rewards.append(ep_reward)

    n = len(rewards)
    ci = 1.96 * np.std(rewards) / np.sqrt(n) if n > 0 else 0
    return {
        'mean': np.mean(rewards), 'std': np.std(rewards), 'ci_95': ci,
        'fail_rate': fails / n * 100, 'jack_rate': jacks / n * 100,
        'safe_rate': safes / n * 100, 'waste_rate': wastes / n * 100,
        'overrides': overrides, 'rewards': rewards, 'n': n,
    }


def cohens_d(r1, r2):
    a, b = np.array(r1), np.array(r2)
    pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def main():
    print("=" * 60)
    print("  PPO vs DQN ALGORITHM COMPARISON")
    print("=" * 60)

    # 1. Load data
    print("\n--- Loading Data ---")
    df = load_combined_data()
    df = calculate_rul(df)
    df_clean, _ = process_data(df, DROP_SENSORS, DROP_SETTINGS)

    # 2. Train PPO agent (same environment, same noise augmentation)
    ppo_path = "models/ppo_pdm_agent.zip"
    if os.path.exists(ppo_path):
        print(f"\n--- Loading existing PPO agent from {ppo_path} ---")
        ppo_model = PPO.load(ppo_path)
    else:
        print(f"\n--- Training PPO Agent ({PPO_TIMESTEPS//1000}k steps) ---")
        train_env = Monitor(PdMEnvironment(
            df_clean, models_dir='models',
            noise_prob=NOISE_PROB,
            noise_level=NOISE_LEVEL,
            variable_noise=True
        ))
        ppo_model = PPO(
            "MlpPolicy", train_env, verbose=1,
            policy_kwargs=dict(net_arch=[256, 256, 128]),
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.01,
            seed=SEED,
        )
        ppo_model.learn(total_timesteps=PPO_TIMESTEPS)
        ppo_model.save("models/ppo_pdm_agent")
        print("--- PPO Agent Saved ---")

    # 3. Load existing DQN agent
    print("\n--- Loading DQN Agent ---")
    dqn_model = DQN.load("models/dqn_pdm_agent")

    # 4. Evaluate both agents across noise levels
    print(f"\n--- Evaluating ({NUM_EPISODES} episodes x {len(NOISE_LEVELS)} noise levels) ---")

    results = []
    for noise in NOISE_LEVELS:
        print(f"\n  Noise sigma = {noise:.3f}")
        eval_env = PdMEnvironment(
            df_clean, models_dir='models',
            noise_prob=1.0 if noise > 0 else 0.0,
            noise_level=noise
        )

        # Evaluate without safety supervisor (fair algorithm comparison)
        dqn_res = evaluate_agent(eval_env, dqn_model, NUM_EPISODES, use_safety=False)
        ppo_res = evaluate_agent(eval_env, ppo_model, NUM_EPISODES, use_safety=False)

        # Statistics
        t_stat, p_val = stats.ttest_ind(dqn_res['rewards'], ppo_res['rewards'], equal_var=False)
        d = cohens_d(dqn_res['rewards'], ppo_res['rewards'])

        print(f"    DQN: mean={dqn_res['mean']:.1f}, jackpot={dqn_res['jack_rate']:.1f}%, fail={dqn_res['fail_rate']:.1f}%")
        print(f"    PPO: mean={ppo_res['mean']:.1f}, jackpot={ppo_res['jack_rate']:.1f}%, fail={ppo_res['fail_rate']:.1f}%")
        print(f"    Delta = {dqn_res['mean'] - ppo_res['mean']:.1f}, p = {p_val:.4f}, d = {d:.3f}")

        results.append({
            'noise': noise,
            'dqn_mean': dqn_res['mean'], 'dqn_std': dqn_res['std'],
            'dqn_ci': dqn_res['ci_95'],
            'dqn_jack': dqn_res['jack_rate'], 'dqn_fail': dqn_res['fail_rate'],
            'ppo_mean': ppo_res['mean'], 'ppo_std': ppo_res['std'],
            'ppo_ci': ppo_res['ci_95'],
            'ppo_jack': ppo_res['jack_rate'], 'ppo_fail': ppo_res['fail_rate'],
            'delta': dqn_res['mean'] - ppo_res['mean'],
            't_stat': t_stat, 'p_value': p_val, 'cohens_d': d,
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv('report_ppo_comparison.csv', index=False)
    print("\n--- Saved report_ppo_comparison.csv ---")

    # 5. Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Mean reward
    ax = axes[0]
    ax.errorbar(df_results['noise'], df_results['dqn_mean'], yerr=df_results['dqn_ci'],
                marker='o', color=DQN_COLOR, label='DQN (UA)', capsize=4, linewidth=2)
    ax.errorbar(df_results['noise'], df_results['ppo_mean'], yerr=df_results['ppo_ci'],
                marker='s', color=PPO_COLOR, label='PPO (UA)', capsize=4, linewidth=2)
    ax.set_xlabel('Sensor Noise (σ)')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Mean Reward Comparison')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4)

    # Panel 2: Jackpot rate
    ax = axes[1]
    ax.plot(df_results['noise'], df_results['dqn_jack'], marker='o',
            color=DQN_COLOR, label='DQN', linewidth=2)
    ax.plot(df_results['noise'], df_results['ppo_jack'], marker='s',
            color=PPO_COLOR, label='PPO', linewidth=2)
    ax.set_xlabel('Sensor Noise (σ)')
    ax.set_ylabel('Jackpot Rate (%)')
    ax.set_title('Optimal Timing Rate')
    ax.legend()

    # Panel 3: Failure rate
    ax = axes[2]
    ax.plot(df_results['noise'], df_results['dqn_fail'], marker='o',
            color=DQN_COLOR, label='DQN', linewidth=2)
    ax.plot(df_results['noise'], df_results['ppo_fail'], marker='s',
            color=PPO_COLOR, label='PPO', linewidth=2)
    ax.set_xlabel('Sensor Noise (σ)')
    ax.set_ylabel('Failure Rate (%)')
    ax.set_title('Catastrophic Failure Rate')
    ax.legend()

    fig.suptitle('Algorithm Comparison: DQN vs PPO (Uncertainty-Aware, No Safety Supervisor)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('report_ppo_vs_dqn.png', dpi=300, bbox_inches='tight')
    print("--- Saved report_ppo_vs_dqn.png ---")

    # 6. Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    dqn_avg = df_results['dqn_mean'].mean()
    ppo_avg = df_results['ppo_mean'].mean()
    print(f"  DQN average reward: {dqn_avg:.1f}")
    print(f"  PPO average reward: {ppo_avg:.1f}")
    print(f"  DQN advantage:      {dqn_avg - ppo_avg:+.1f}")
    sig_count = (df_results['p_value'] < 0.05).sum()
    print(f"  Significant at p<0.05: {sig_count}/{len(NOISE_LEVELS)} noise levels")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
