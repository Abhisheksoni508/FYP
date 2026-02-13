import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import pandas as pd
from src.config import *
from src.lstm_model import RUL_LSTM


class PdMEnvironment(gym.Env):
    """
    Predictive Maintenance Gymnasium Environment.
    
    The agent observes [mean_rul, uncertainty, sensor_trend] and decides
    whether to WAIT (0) or MAINTAIN (1).
    
    Noise augmentation (noise_prob > 0) injects Gaussian noise into sensor
    readings before the ensemble sees them. This causes ensemble disagreement
    (higher sigma), teaching the agent that high sigma = unreliable predictions.
    
    Args:
        df: Preprocessed DataFrame with sensor data and RUL
        models_dir: Directory containing ensemble_model_*.pth files
        noise_prob: Fraction of episodes with sensor noise (0.0 to 1.0)
        noise_level: Std of Gaussian noise when active
    """
    
    def __init__(self, df, models_dir, noise_prob=0.0, noise_level=0.15):
        super().__init__()
        
        self.df = df
        self.units = df['unit'].unique()
        self.models = self._load_ensemble(models_dir)
        
        # Noise settings
        self.noise_prob = noise_prob
        self.noise_level = noise_level
        self.episode_noisy = False
        
        # Spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # State
        self.current_unit = None
        self.current_time = 0
        self.max_time = 0
        
    def _load_ensemble(self, models_dir):
        models = []
        for i in range(ENSEMBLE_SIZE):
            path = f"{models_dir}/ensemble_model_{i}.pth"
            m = RUL_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT)
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.to(DEVICE)
            m.eval()
            models.append(m)
        return models

    def reset(self, seed=None, options=None):
        self.current_unit = np.random.choice(self.units)
        unit_data = self.df[self.df['unit'] == self.current_unit]
        
        max_len = len(unit_data)
        if max_len <= WINDOW_SIZE + 10:
            self.current_time = WINDOW_SIZE
        else:
            self.current_time = np.random.randint(WINDOW_SIZE, max_len - 10)
        self.max_time = max_len
        
        # Decide noise for this entire episode
        self.episode_noisy = (np.random.random() < self.noise_prob)
        
        return self._get_observation(), {}

    def _get_observation(self):
        unit_data = self.df[self.df['unit'] == self.current_unit]
        
        start_idx = self.current_time - WINDOW_SIZE
        end_idx = self.current_time
        
        features = [c for c in unit_data.columns if c not in ['unit', 'time', 'RUL']]
        seq = unit_data.iloc[start_idx:end_idx][features].values.copy()
        
        # Noise injection (if this episode is noisy)
        if self.episode_noisy:
            noise = np.random.normal(0, self.noise_level, seq.shape)
            seq = np.clip(seq + noise, 0, 1)
        
        # Ensemble inference
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
        preds = []
        with torch.no_grad():
            for model in self.models:
                preds.append(model(seq_tensor).item())
        
        mean_pred = np.mean(preds)
        std_pred = np.std(preds)
        
        # Build observation
        norm_mean = np.clip(mean_pred, 0, 1)
        norm_std = np.clip(std_pred * UNCERTAINTY_SCALE, 0, 1)
        sensor_trend = np.clip(np.mean(seq[-1, :]), 0, 1)
        
        return np.array([norm_mean, norm_std, sensor_trend], dtype=np.float32)

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0
        
        unit_data = self.df[self.df['unit'] == self.current_unit]
        true_failure_time = unit_data['time'].max()
        current_true_rul = true_failure_time - unit_data.iloc[self.current_time]['time']
        current_norm_rul = current_true_rul / 125.0
        
        NORM_REPAIR = 20 / 125.0    # Jackpot window
        NORM_BUFFER = 50 / 125.0    # Acceptable window

        if action == 1:  # MAINTAIN
            if current_norm_rul < NORM_REPAIR:
                reward = 500     # Jackpot
            elif current_norm_rul < NORM_BUFFER:
                reward = 10      # Acceptable
            else:
                reward = -20     # Wasteful
            terminated = True
            
        else:  # WAIT
            self.current_time += 1
            if self.current_time >= self.max_time:
                reward = -100    # Crash
                terminated = True
            else:
                reward = 1       # Survived
                
        return self._get_observation(), reward, terminated, truncated, {}