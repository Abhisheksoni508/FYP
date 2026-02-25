"""
Train the Uncertainty-Aware DQN Agent with Noise Augmentation.

CRITICAL: This trains with NOISE_PROB=50% noisy episodes so the agent
learns what high uncertainty (sigma) means. Without this, the agent
has no reason to use the uncertainty signal.

Usage: python main_train_rl.py
Prereq: Ensemble models must exist in models/ (run main_train_ensemble.py first)
Output: models/dqn_pdm_agent.zip
Time: ~30-45 mins on CPU
"""

import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from src.preprocessing import load_combined_data, calculate_rul, process_data
from src.gym_env import PdMEnvironment
from src.config import *


def train_rl_agent():
    print("=" * 60)
    print("  UNCERTAINTY-AWARE DQN TRAINING")
    print("=" * 60)
    print(f"\n  Device:         {DEVICE}")
    print(f"  Timesteps:      {RL_TIMESTEPS:,}")
    print(f"  Network:        {RL_NET_ARCH}")
    print(f"  Noise Prob:     {NOISE_PROB} ({int(NOISE_PROB*100)}% of episodes)")
    print(f"  Noise Range:    {NOISE_LEVEL_MIN} – {NOISE_LEVEL} (sampled per episode)")
    print(f"  Uncertainty 5x: {UNCERTAINTY_SCALE}")
    print(f"  Obs Space:      4D [mean_rul, sigma_now, sigma_rolling, trend]")
    
    # 1. Load Data
    print("\n--- Loading Data ---")
    df = load_combined_data()
    df = calculate_rul(df)
    df_clean, _ = process_data(df, DROP_SENSORS, DROP_SETTINGS)
    
    # 2. Create Environment WITH noise augmentation
    # This is the key: 50% of episodes inject noise into sensors,
    # causing ensemble sigma to spike. The agent learns to detect this.
    print(f"\n--- Creating Environment (noise_prob={NOISE_PROB}, variable_noise=True) ---")
    env = Monitor(PdMEnvironment(
        df_clean, 
        models_dir='models',
        noise_prob=NOISE_PROB,       # FROM CONFIG (now 0.7)
        noise_level=NOISE_LEVEL,     # FROM CONFIG (max 0.15)
        variable_noise=True          # Option 2: sample noise level per episode
    ))
    
    # 3. Create DQN
    model = DQN(
        "MlpPolicy", env, verbose=1,
        policy_kwargs=dict(net_arch=RL_NET_ARCH),
        learning_rate=RL_LR,
        batch_size=RL_BATCH,
        gamma=RL_GAMMA,
        buffer_size=RL_BUFFER,
        learning_starts=RL_LEARNING_STARTS,
        exploration_fraction=RL_EXPLORE_FRAC,
        exploration_initial_eps=1.0,
        exploration_final_eps=RL_EXPLORE_FINAL,
        target_update_interval=RL_TARGET_UPDATE,
        train_freq=RL_TRAIN_FREQ,
        gradient_steps=1,
    )

    # 4. Train
    print(f"\n--- Training ({RL_TIMESTEPS//1000}k steps) ---")
    model.learn(total_timesteps=RL_TIMESTEPS)
    
    # 5. Save
    model.save("models/dqn_pdm_agent")
    print("\n--- Agent Saved to models/dqn_pdm_agent.zip ---")

    # 6. Quick eval on CLEAN data
    print("\n--- Quick Eval (30 episodes, clean data) ---")
    clean_env = PdMEnvironment(df_clean, models_dir='models', noise_prob=0.0)
    
    rewards, jackpots, failures = [], 0, 0
    for _ in range(30):
        obs, _ = clean_env.reset()
        done, ep_reward = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = clean_env.step(action)
            ep_reward += reward
            if done:
                if reward == 500: jackpots += 1
                if reward == -100: failures += 1
        rewards.append(ep_reward)
    
    print(f"\n{'='*45}")
    print(f"  CLEAN DATA EVAL (30 eps)")
    print(f"{'='*45}")
    print(f"  Mean Reward:  {np.mean(rewards):.2f}")
    print(f"  Std Reward:   {np.std(rewards):.2f}")
    print(f"  Jackpots:     {jackpots}/30")
    print(f"  Failures:     {failures}/30")
    print(f"{'='*45}")
    
    # 7. Quick eval on NOISY data
    print("\n--- Quick Eval (30 episodes, noisy data) ---")
    noisy_env = PdMEnvironment(df_clean, models_dir='models', noise_prob=1.0, noise_level=0.15)
    
    rewards_n, jackpots_n, failures_n = [], 0, 0
    for _ in range(30):
        obs, _ = noisy_env.reset()
        done, ep_reward = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = noisy_env.step(action)
            ep_reward += reward
            if done:
                if reward == 500: jackpots_n += 1
                if reward == -100: failures_n += 1
        rewards_n.append(ep_reward)
    
    print(f"\n{'='*45}")
    print(f"  NOISY DATA EVAL (30 eps, noise=0.15)")
    print(f"{'='*45}")
    print(f"  Mean Reward:  {np.mean(rewards_n):.2f}")
    print(f"  Jackpots:     {jackpots_n}/30")
    print(f"  Failures:     {failures_n}/30")
    print(f"{'='*45}")
    
    print(f"\n  Noise robustness: {np.mean(rewards_n)/max(np.mean(rewards),1)*100:.1f}% of clean performance retained")


if __name__ == "__main__":
    train_rl_agent()