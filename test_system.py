"""
Automated Test Suite for the Predictive Maintenance Framework.

Covers unit tests (data pipeline, LSTM, environment, safety supervisor)
and integration tests (ensemble-environment interaction, agent evaluation).

Usage:  pytest test_system.py -v
Prereq: Trained models in models/ directory
"""

import numpy as np
import torch
import pytest

from src.config import (
    INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, ENSEMBLE_SIZE,
    WINDOW_SIZE, UNCERTAINTY_SCALE, CRITICAL_RUL_NORM,
    HARD_CRITICAL_RUL_NORM, SUPERVISOR_SIGMA_THRESHOLD,
    CRASH_PENALTY, DEVICE
)
from src.lstm_model import RUL_LSTM
from src.preprocessing import load_combined_data, calculate_rul, process_data
from src.gym_env import (
    PdMEnvironment, BlindPdMEnvironment,
    safety_override, classify_terminal_reward
)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def data():
    """Load and preprocess C-MAPSS data once for all tests."""
    df = load_combined_data()
    df = calculate_rul(df)
    df_clean, _ = process_data(df, [], [])
    return df_clean


@pytest.fixture(scope="session")
def env(data):
    """Create a clean (no noise) environment."""
    return PdMEnvironment(data, models_dir='models', noise_prob=0.0)


@pytest.fixture(scope="session")
def blind_env(data):
    """Create a blind ablation environment."""
    return BlindPdMEnvironment(data, models_dir='models', noise_prob=0.0)


@pytest.fixture(scope="session")
def noisy_env(data):
    """Create a noisy environment."""
    return PdMEnvironment(data, models_dir='models', noise_prob=1.0, noise_level=0.15)


# ── Data Pipeline Tests ──────────────────────────────────────

class TestDataPipeline:
    def test_combined_data_loads(self):
        df = load_combined_data()
        assert len(df) > 0, "DataFrame should not be empty"

    def test_column_count(self):
        df = load_combined_data()
        # unit, time, 3 settings, 21 sensors = 26 columns minimum
        assert len(df.columns) >= 26

    def test_rul_cap(self):
        df = load_combined_data()
        df = calculate_rul(df)
        assert df['RUL'].max() <= 125, "RUL should be capped at 125"

    def test_no_nan_after_processing(self):
        df = load_combined_data()
        df = calculate_rul(df)
        df_clean, _ = process_data(df, [], [])
        features = [c for c in df_clean.columns if c not in ['unit', 'time', 'RUL']]
        assert not df_clean[features].isnull().any().any(), "No NaN values after normalisation"

    def test_fd002_unit_offset(self):
        df = load_combined_data()
        units = df['unit'].unique()
        # FD001 has units 1-100, FD002 should be offset to 101+
        assert units.max() > 100, "FD002 units should be offset beyond 100"
        assert len(units) == 360, "Should have 360 unique engines (100 + 260)"


# ── LSTM Model Tests ─────────────────────────────────────────

class TestLSTMModel:
    def test_output_shape(self):
        model = RUL_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT)
        x = torch.randn(4, WINDOW_SIZE, INPUT_DIM)  # batch=4
        out = model(x)
        assert out.numel() == 4, f"Expected 4 scalar outputs, got shape {out.shape}"

    def test_output_range_on_normalised_input(self):
        model = RUL_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT)
        model.eval()
        x = torch.rand(1, WINDOW_SIZE, INPUT_DIM)  # values in [0,1]
        with torch.no_grad():
            out = model(x).item()
        # Output should be finite (not NaN or Inf)
        assert np.isfinite(out), f"Output should be finite, got {out}"

    def test_ensemble_models_load(self):
        for i in range(ENSEMBLE_SIZE):
            path = f"models/ensemble_model_{i}.pth"
            m = RUL_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT)
            m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            assert m is not None

    def test_ensemble_diversity(self):
        """Ensemble members should produce different predictions (they are diverse)."""
        models = []
        for i in range(ENSEMBLE_SIZE):
            m = RUL_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT)
            m.load_state_dict(torch.load(f"models/ensemble_model_{i}.pth", map_location=DEVICE, weights_only=True))
            m.eval()
            models.append(m)

        x = torch.rand(1, WINDOW_SIZE, INPUT_DIM).to(DEVICE)
        preds = []
        with torch.no_grad():
            for m in models:
                m.to(DEVICE)
                preds.append(m(x).item())
        assert np.std(preds) > 0, "Ensemble members should produce different predictions"


# ── Environment Tests ────────────────────────────────────────

class TestEnvironment:
    def test_reset_returns_4d_observation(self, env):
        obs, info = env.reset()
        assert obs.shape == (4,), f"Expected 4D obs, got {obs.shape}"

    def test_observation_in_range(self, env):
        obs, _ = env.reset()
        assert np.all(obs >= 0) and np.all(obs <= 1), f"Obs should be in [0,1], got {obs}"

    def test_wait_advances_time(self, env):
        env.reset()
        t_before = env.current_time
        env.step(0)  # WAIT
        assert env.current_time == t_before + 1

    def test_maintain_terminates(self, env):
        env.reset()
        _, _, terminated, _, _ = env.step(1)  # MAINTAIN
        assert terminated, "MAINTAIN should terminate the episode"

    def test_noise_increases_sigma(self, data):
        """Noisy env should produce higher uncertainty than clean env."""
        clean_env = PdMEnvironment(data, models_dir='models', noise_prob=0.0)
        noisy_env = PdMEnvironment(data, models_dir='models', noise_prob=1.0, noise_level=0.15)

        np.random.seed(42)
        clean_sigmas, noisy_sigmas = [], []
        for _ in range(20):
            obs, _ = clean_env.reset()
            clean_sigmas.append(obs[1])
            obs, _ = noisy_env.reset()
            noisy_sigmas.append(obs[1])

        assert np.mean(noisy_sigmas) > np.mean(clean_sigmas), \
            "Noisy env should have higher mean sigma"

    def test_blind_env_zeros_sigma(self, blind_env):
        obs, _ = blind_env.reset()
        assert obs[1] == 0.0, f"Blind sigma_now should be 0, got {obs[1]}"
        assert obs[2] == 0.0, f"Blind sigma_rolling should be 0, got {obs[2]}"


# ── Reward Function Tests ────────────────────────────────────

class TestRewardFunction:
    def test_classify_jackpot(self):
        assert classify_terminal_reward(500) == 'jackpot'
        assert classify_terminal_reward(530) == 'jackpot'  # with proactive bonus

    def test_classify_crash(self):
        assert classify_terminal_reward(CRASH_PENALTY) == 'crash'

    def test_classify_safe(self):
        assert classify_terminal_reward(10) == 'safe'
        assert classify_terminal_reward(40) == 'safe'  # with proactive bonus

    def test_classify_wasteful(self):
        assert classify_terminal_reward(-20) == 'wasteful'


# ── Safety Supervisor Tests ──────────────────────────────────

class TestSafetySupervisor:
    def test_tier1_fires_at_hard_critical(self):
        """Tier 1 should fire regardless of sigma when RUL is critically low."""
        obs = np.array([0.03, 0.9, 0.9, 0.5])  # very low RUL, very high sigma
        action, overridden = safety_override(0, obs)  # agent says WAIT
        assert action == 1 and overridden, "Tier 1 should override to MAINTAIN"

    def test_tier2_fires_when_confident(self):
        """Tier 2 should fire when RUL is low AND sigma is low (confident)."""
        obs = np.array([0.10, 0.2, 0.2, 0.5])  # low RUL, low sigma
        action, overridden = safety_override(0, obs)
        assert action == 1 and overridden, "Tier 2 should override when confident"

    def test_tier2_backs_off_when_uncertain(self):
        """Tier 2 should NOT fire when sigma is high (uncertain prediction)."""
        obs = np.array([0.10, 0.8, 0.8, 0.5])  # low RUL, HIGH sigma
        action, overridden = safety_override(0, obs)
        assert action == 0 and not overridden, "Tier 2 should back off under high uncertainty"

    def test_no_override_when_rul_safe(self):
        """Supervisor should not fire when RUL is above critical threshold."""
        obs = np.array([0.5, 0.1, 0.1, 0.5])  # healthy RUL
        action, overridden = safety_override(0, obs)
        assert action == 0 and not overridden, "Should not override when RUL is safe"

    def test_no_override_on_maintain(self):
        """Supervisor should never override a MAINTAIN decision."""
        obs = np.array([0.03, 0.9, 0.9, 0.5])  # critical state
        action, overridden = safety_override(1, obs)  # agent already says MAINTAIN
        assert action == 1 and not overridden, "Should not override MAINTAIN"

    def test_blind_always_confident(self):
        """Blind agent has sigma=0, so Tier 2 should always consider it confident."""
        obs = np.array([0.10, 0.0, 0.0, 0.5])  # low RUL, zero sigma (blind)
        action, overridden = safety_override(0, obs)
        assert action == 1 and overridden, "Blind agent should always trigger Tier 2"


# ── Integration Tests ────────────────────────────────────────

class TestIntegration:
    def test_full_episode_completes(self, env):
        """A full episode should terminate within a reasonable number of steps."""
        obs, _ = env.reset()
        for step in range(500):
            obs, reward, done, _, _ = env.step(1)  # always MAINTAIN
            if done:
                break
        assert done, "Episode should terminate when MAINTAIN is selected"

    def test_reproducibility(self, data):
        """Same seed should produce identical trajectories."""
        env1 = PdMEnvironment(data, models_dir='models', noise_prob=0.0)
        env2 = PdMEnvironment(data, models_dir='models', noise_prob=0.0)

        np.random.seed(123)
        obs1, _ = env1.reset()
        np.random.seed(123)
        obs2, _ = env2.reset()

        np.testing.assert_array_equal(obs1, obs2)

    def test_dqn_agent_loads_and_predicts(self, env):
        """The trained DQN agent should load and produce valid actions."""
        from stable_baselines3 import DQN
        model = DQN.load("models/dqn_pdm_agent")
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action in [0, 1], f"Action should be 0 or 1, got {action}"

    def test_config_values_positive(self):
        """All key config values should be positive numbers."""
        assert INPUT_DIM > 0
        assert HIDDEN_DIM > 0
        assert WINDOW_SIZE > 0
        assert ENSEMBLE_SIZE > 0
        assert UNCERTAINTY_SCALE > 0
        assert CRITICAL_RUL_NORM > 0
        assert HARD_CRITICAL_RUL_NORM > 0
        assert SUPERVISOR_SIGMA_THRESHOLD > 0
        assert CRASH_PENALTY < 0  # penalty should be negative
