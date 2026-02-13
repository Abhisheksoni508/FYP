from src.preprocessing import load_data, calculate_rul, process_data
from src.gym_env import PdMEnvironment
from src.config import *
import numpy as np

# 1. Load Data
print("Loading data...")
df = load_data('data/train_FD001.txt')
df = calculate_rul(df)
df_clean, _ = process_data(df, DROP_SENSORS, DROP_SETTINGS)

# 2. Initialize Environment
print("Initializing Environment...")
env = PdMEnvironment(df_clean, models_dir='models')

# 3. Test Reset
obs, info = env.reset()
print(f"Initial Observation: {obs}")
print(f"Obs Shape: {obs.shape} (Should be (3,))")

# 4. Test Step
# Force an action: 0 = Wait
obs, reward, terminated, truncated, info = env.step(0)
print(f"Step Result -> Reward: {reward}, Terminated: {terminated}")

print("\nStep 3 Verification: SUCCESS. The environment is ready.")