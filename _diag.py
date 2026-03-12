"""Quick diagnostic: outcome distributions at different noise levels."""
import numpy as np
from stable_baselines3 import DQN
from src.preprocessing import load_combined_data, calculate_rul, process_data
from src.gym_env import PdMEnvironment, BlindPdMEnvironment, safety_override
from src.config import *

df = load_combined_data()
df = calculate_rul(df)
df_clean, _ = process_data(df, DROP_SENSORS, DROP_SETTINGS)
ua_model = DQN.load('models/dqn_pdm_agent')
blind_model = DQN.load('models/dqn_blind_agent')

N = 300
for nl in [0.0, 0.05, 0.10, 0.15]:
    ua_env = PdMEnvironment(df_clean, 'models', noise_prob=0.0 if nl == 0 else 1.0, noise_level=nl)
    bl_env = BlindPdMEnvironment(df_clean, 'models', noise_prob=0.0 if nl == 0 else 1.0, noise_level=nl)
    for name, env, model in [('UA', ua_env, ua_model), ('Blind', bl_env, blind_model)]:
        counts = {'jackpot': 0, 'safe': 0, 'wasteful': 0, 'crash': 0, 'other': 0}
        overrides = 0
        for _ in range(N):
            obs, _ = env.reset()
            done = False
            ep_ovr = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action, was_ovr = safety_override(action, obs)
                if was_ovr:
                    ep_ovr = True
                obs, reward, done, _, _ = env.step(action)
                if done:
                    if reward <= -100:
                        counts['crash'] += 1
                    elif reward >= 500:
                        counts['jackpot'] += 1
                    elif reward >= 10 and reward < 100:
                        counts['safe'] += 1
                    elif reward == -20:
                        counts['wasteful'] += 1
                    else:
                        counts['other'] += 1
            if ep_ovr:
                overrides += 1
        pcts = {k: v / N * 100 for k, v in counts.items()}
        print(f"s={nl:.2f} {name:5s}  J={pcts['jackpot']:5.1f}%  S={pcts['safe']:5.1f}%  "
              f"W={pcts['wasteful']:5.1f}%  C={pcts['crash']:5.1f}%  Ovr={overrides/N*100:.0f}%")
    print()
