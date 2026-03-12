"""Diagnose: what reward values is UA agent getting that aren't being categorised?"""
import numpy as np
from stable_baselines3 import DQN
from src.preprocessing import load_combined_data, calculate_rul, process_data
from src.gym_env import PdMEnvironment, BlindPdMEnvironment, safety_override
from src.config import *

df = load_combined_data()
df = calculate_rul(df)
df_clean, _ = process_data(df, DROP_SENSORS, DROP_SETTINGS)
ua_model = DQN.load('models/dqn_pdm_agent')

N = 100
for nl in [0.10, 0.15]:
    env = PdMEnvironment(df_clean, 'models', noise_prob=1.0, noise_level=nl)
    odd_rewards = []
    for _ in range(N):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True) if False else (0, None)
            action, _ = ua_model.predict(obs, deterministic=True)
            action, _ = safety_override(action, obs)
            obs, reward, done, _, _ = env.step(action)
            if done:
                if reward not in [500, 10, -20] and reward > -100:
                    odd_rewards.append(round(reward, 2))
    print(f"s={nl:.2f}: {len(odd_rewards)}/{N} episodes with non-standard rewards")
    if odd_rewards:
        unique = sorted(set(odd_rewards))
        print(f"  Unique values: {unique[:20]}")
        # Check if these could be jackpots with proactive bonus
        print(f"  Range: {min(odd_rewards)} to {max(odd_rewards)}")
