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
    
    def __init__(self, df, models_dir, noise_prob=0.0, noise_level=0.15, variable_noise=False):
        super().__init__()
        
        self.df = df
        self.units = df['unit'].unique()
        self.models = self._load_ensemble(models_dir)
        
        # Noise settings
        self.noise_prob = noise_prob
        self.noise_level = noise_level            # Max noise std
        self.variable_noise = variable_noise      # Option 2: sample per episode
        self.episode_noisy = False
        self.episode_noise_level = noise_level    # Set each episode in reset()
        
        # Spaces — 4D obs: [mean_rul, sigma_now, sigma_rolling_avg, sensor_trend]
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        # State
        self.current_unit = None
        self.current_time = 0
        self.max_time = 0
        self.sigma_history = [0.0, 0.0, 0.0]     # Option 3: rolling sigma window
        
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
            # End-of-life bias: EOL_EPISODE_PROB fraction of episodes start
            # in the last EOL_WINDOW cycles so the agent sees many more
            # maintenance timing decisions during training.
            # Without this, only ~13% of starts land in the critical zone
            # and the agent learns "always WAIT" as the dominant strategy.
            if np.random.random() < EOL_EPISODE_PROB:
                start_min = max(WINDOW_SIZE, max_len - EOL_WINDOW)
                self.current_time = np.random.randint(start_min, max_len - 5)
            else:
                self.current_time = np.random.randint(WINDOW_SIZE, max_len - 10)
        self.max_time = max_len
        
        # Reset sigma history at start of each new episode
        self.sigma_history = [0.0, 0.0, 0.0]

        # Decide noise for this entire episode
        self.episode_noisy = (np.random.random() < self.noise_prob)

        # Option 2: Sample noise level per episode so agent learns all intensities
        if self.episode_noisy and self.variable_noise:
            self.episode_noise_level = np.random.uniform(NOISE_LEVEL_MIN, self.noise_level)
        else:
            self.episode_noise_level = self.noise_level
        
        return self._get_observation(), {}

    def _get_observation(self):
        unit_data = self.df[self.df['unit'] == self.current_unit]
        
        start_idx = self.current_time - WINDOW_SIZE
        end_idx = self.current_time
        
        features = [c for c in unit_data.columns if c not in ['unit', 'time', 'RUL']]
        seq = unit_data.iloc[start_idx:end_idx][features].values.copy()
        
        # Noise injection — uses per-episode level (Option 2: variable noise)
        if self.episode_noisy:
            noise = np.random.normal(0, self.episode_noise_level, seq.shape)
            seq = np.clip(seq + noise, 0, 1)
        
        # Ensemble inference
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
        preds = []
        with torch.no_grad():
            for model in self.models:
                preds.append(model(seq_tensor).item())
        
        mean_pred = np.mean(preds)
        std_pred = np.std(preds)
        
        # Build 4D observation: [mean_rul, sigma_now, sigma_rolling_avg, sensor_trend]
        norm_mean = np.clip(mean_pred, 0, 1)
        norm_std = np.clip(std_pred * UNCERTAINTY_SCALE, 0, 1)
        sensor_trend = np.clip(np.mean(seq[-1, :]), 0, 1)

        # Option 3: Rolling sigma average (last 3 steps)
        self.sigma_history.pop(0)
        self.sigma_history.append(float(norm_std))
        rolling_sigma = float(np.mean(self.sigma_history))

        # Store sigma for uncertainty-aware reward shaping
        self._current_sigma = float(norm_std)

        return np.array([norm_mean, norm_std, rolling_sigma, sensor_trend], dtype=np.float32)

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
            # Proactive maintenance bonus: if sigma is high (uncertainty > 0.5)
            # AND we're in the buffer zone, give a small bonus to reward
            # risk-aware decision-making. This teaches UA to maintain under
            # uncertainty rather than gambling on waiting.
            sigma = getattr(self, '_current_sigma', 0.0)
            proactive_bonus = 0
            if sigma > 0.5 and current_norm_rul < NORM_BUFFER:
                proactive_bonus = 30  # Bonus for being cautious under uncertainty
            
            if current_norm_rul < NORM_REPAIR:
                reward = 500 + proactive_bonus     # Jackpot (bonus doesn't matter here)
            elif current_norm_rul < NORM_BUFFER:
                reward = 10 + proactive_bonus      # Acceptable, possibly with uncertainty bonus
            else:
                reward = -20     # Wasteful (still penalize too early)
            terminated = True
            
        else:  # WAIT
            self.current_time += 1
            if self.current_time >= self.max_time:
                reward = CRASH_PENALTY
                terminated = True
            else:
                # Time-pressure shaping: WAIT reward decays as engine nears end of life.
                # Above TIME_PRESSURE_START cycles remaining → full +1 reward.
                # Below that → reward drops linearly, going negative near failure.
                time_pressure_penalty = max(0.0, (TIME_PRESSURE_START - current_true_rul) / 10.0)

                # Uncertainty urgency: when sigma is high AND near failure,
                # waiting is EXTRA costly.  UA agent sees real sigma → learns
                # “uncertain + near failure = maintain NOW”.  Blind agent has
                # sigma=0 → no urgency → keeps waiting → crashes more.
                # STRENGTHENED: multiplier increased from /8.0 to /3.0, making
                # uncertainty avoidance 2.7x stronger to force risk-averse behavior
                sigma = getattr(self, '_current_sigma', 0.0)
                uncertainty_urgency = sigma * max(0.0, (TIME_PRESSURE_START - current_true_rul) / 3.0)
                reward = 1.0 - time_pressure_penalty - uncertainty_urgency
                
        return self._get_observation(), reward, terminated, truncated, {}


# ============================================================
# Blind Environment (for ablation studies)
# ============================================================

class BlindPdMEnvironment(PdMEnvironment):
    """
    Identical to PdMEnvironment but the uncertainty signal (sigma)
    is always zero. Used in ablation studies to test whether
    uncertainty awareness improves decision-making.
    """
    def _get_observation(self):
        obs = super()._get_observation()
        obs[1] = 0.0  # Zero out current sigma
        obs[2] = 0.0  # Zero out rolling sigma avg — blind agent sees neither
        self._current_sigma = 0.0  # No uncertainty urgency for blind agent
        return obs


# ============================================================
# Safety Supervisor (Layer 3)
# ============================================================

def safety_override(action, obs):
    """
    Layer 3 Safety Supervisor: overrides the DQN agent's decision
    when predicted RUL falls below a critical threshold AND the
    ensemble is confident in that prediction (low rolling sigma).

    Two-tier design:
      Tier 1 — Hard critical fallback (HARD_CRITICAL_RUL_NORM):
        At extreme sensor noise the rolling sigma can persistently exceed
        SUPERVISOR_SIGMA_THRESHOLD, silencing the supervisor entirely and
        leaving no safety net. This tier fires regardless of uncertainty —
        if the predicted RUL is catastrophically close to failure (< ~5 cycles)
        there is no justified reason to keep waiting.

      Tier 2 — Uncertainty-aware override (CRITICAL_RUL_NORM):
        Normal operating range. Fires only when the ensemble is confident
        (rolling sigma < threshold), which is what separates UA from Blind:
          - Blind agent: rolling sigma always 0 → always "confident"
            → supervisor fires on every low prediction (catches false alarms too)
          - UA agent: rolling sigma reflects real disagreement
            → ignores noise-driven dips, fires only on trustworthy signals

    Args:
        action (int): DQN's chosen action (0=WAIT, 1=MAINTAIN)
        obs (np.array): [mean_rul_norm, sigma_now, rolling_sigma, sensor_trend]

    Returns:
        final_action (int): Possibly overridden action
        was_overridden (bool): True if safety supervisor intervened
    """
    if action == 0:
        # Tier 1: hard fallback — always fire at extreme proximity to failure
        if obs[0] < HARD_CRITICAL_RUL_NORM:
            return 1, True

        # Tier 2: uncertainty-aware override — only fire when confident
        # rolling_sigma (obs[2]) is the 3-cycle average of sigma,
        # more stable than single-cycle sigma, harder to fool with one spike
        is_confident = obs[2] < SUPERVISOR_SIGMA_THRESHOLD
        if obs[0] < CRITICAL_RUL_NORM and is_confident:
            return 1, True

    return action, False


def classify_terminal_reward(reward):
    """
    Map terminal rewards to a stable outcome label.

    The environment can emit proactive-bonus variants of the nominal
    jackpot/safe rewards (for example 530 or 40). Centralising the
    classification keeps evaluation scripts consistent with the current
    reward design.
    """
    if reward <= CRASH_PENALTY:
        return 'crash'
    if reward >= 500:
        return 'jackpot'
    if reward >= 10:
        return 'safe'
    if reward == -20:
        return 'wasteful'
    return 'other'
