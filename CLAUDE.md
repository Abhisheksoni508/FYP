# CLAUDE.md — FYP Project Reference

## Project Identity

| Field | Value |
|-------|-------|
| **Title** | A Robust Predictive Maintenance Framework for IoT Systems via Uncertainty-Aware Reinforcement Learning |
| **Author** | Abhishek Soni |
| **Module** | COMP1682 — Final Year Project |
| **Degree** | BSc(Hons) Computer Science |
| **University** | University of Greenwich |
| **Supervisor** | Yasmine Arafa |

---

## Architecture — 3-Layer Hybrid AI System

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: LSTM Deep Ensemble (Prediction + Uncertainty)     │
│  5 × LSTM models → μ_RUL, σ (ensemble disagreement)        │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: DQN Agent (Uncertainty-Aware Decision-Making)     │
│  Observes [μ, σ_now, σ_rolling, trend] → WAIT or MAINTAIN  │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Safety Supervisor (Hard Safety Override)          │
│  σ-gated 2-tier override: confident low-RUL → force MAINTAIN│
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: LSTM Deep Ensemble (`src/lstm_model.py`)

- **Architecture:** 2-layer LSTM, 100 hidden units, dropout=0.2
- **Input:** (batch, window=30, features=24) — 21 sensors + 3 operational settings
- **Output:** Normalized RUL ∈ [0,1], scaled to cycles via ×125
- **Ensemble:** 5 independent models trained with bootstrap sampling (80% data each, different subsets)
- **Diversity:** Deterministic seeds (42 + model_idx × 17) for reproducibility
- **Uncertainty:** σ = std(pred₁, …, pred₅), scaled by UNCERTAINTY_SCALE=15

### Layer 2: DQN Agent (`src/gym_env.py` — `PdMEnvironment`)

**Observation Space (4D continuous ∈ [0,1]⁴):**

| Index | Feature | Purpose |
|-------|---------|---------|
| 0 | `mean_rul` | Normalized predicted RUL (trajectory) |
| 1 | `sigma_now` | Current cycle ensemble disagreement (spike detection) |
| 2 | `sigma_rolling_avg` | 3-cycle rolling average of σ (robust trend) |
| 3 | `sensor_trend` | Mean of recent sensor readings (engine health) |

**Action Space:** Binary discrete — 0=WAIT, 1=MAINTAIN

**DQN Network:** 3-layer MLP [256, 256, 128] via Stable-Baselines3

**Reward Function:**

| Condition | Reward | Label |
|-----------|--------|-------|
| MAINTAIN when RUL < 20 | +500 | JACKPOT (optimal timing) |
| MAINTAIN when RUL < 50 | +10 | SAFE (acceptable, slightly early) |
| MAINTAIN when RUL ≥ 50 | -20 | WASTEFUL (too early) |
| WAIT when RUL ≤ 0 | -500 | CRASH (catastrophic failure) |
| WAIT otherwise | 1.0 - time_pressure - uncertainty_urgency | Shaped step reward |

**Step reward shaping:**
- `time_pressure = max(0, (25 - true_RUL) / 10.0)` — linear decay in final 25 cycles
- `uncertainty_urgency = σ × max(0, (25 - true_RUL) / 3.0)` — 2.7× stronger than time pressure; penalizes waiting under high σ near failure
- Proactive bonus (+30) when UA maintains in buffer zone under high uncertainty

### Layer 3: Safety Supervisor (`src/gym_env.py` — `safety_override`)

Two-tier σ-gated override (only activates when DQN chooses WAIT):

| Tier | Condition | Behavior |
|------|-----------|----------|
| **Tier 1 (Emergency)** | RUL_norm < 0.04 (~5 cycles) | Force MAINTAIN regardless of σ |
| **Tier 2 (Confident)** | RUL_norm < 0.12 (~15 cycles) AND rolling_σ < 0.55 | Force MAINTAIN only when confident |

- **UA agent:** σ reflects real uncertainty → supervisor fires only when truly needed (7 overrides per 500 episodes on clean data)
- **Blind agent:** σ always 0 → supervisor fires on every low-RUL prediction (215+ overrides per 500 episodes)

---

## Dataset — NASA C-MAPSS Turbofan Engine Degradation

**Source:** NASA Prognostics Center of Excellence

| Subset | Engines | Operating Conditions | Fault Mode | Cycles |
|--------|---------|---------------------|------------|--------|
| FD001 | 100 | 1 (single) | HPC degradation | 20,631 |
| FD002 | 260 | 6 (realistic variety) | HPC degradation | 61,249 |
| **Combined** | **360** | **Mixed** | **Single** | **81,880 sequences** |

- **Features per cycle:** 3 operational settings + 21 sensor readings = 24 total
- **Normalization:** MinMaxScaler to [0,1] (saved as `models/scaler.pkl`)
- **RUL capping:** min(RUL, 125) — standard practice for C-MAPSS
- **Window size:** 30 cycles (fixed temporal context per prediction)

---

## MDP Formulation

| Element | Definition |
|---------|-----------|
| **State** | s = [μ_RUL, σ_now, σ_rolling, trend] ∈ [0,1]⁴ |
| **Actions** | A = {0: WAIT, 1: MAINTAIN} |
| **Reward** | See reward table above |
| **Transitions** | Deterministic: advance one cycle in recorded engine lifecycle |
| **Episode termination** | action=MAINTAIN OR RUL reaches 0 (crash) |
| **Discount factor** | γ = 0.99 |
| **Stochasticity** | Aleatoric (noise injection), Epistemic (ensemble disagreement) |

---

## Key Innovations

### 1. Noise Augmentation for Uncertainty Learning
70% of training episodes inject Gaussian sensor noise (σ ∈ [0.02, 0.20]). This causes ensemble disagreement to spike, teaching the DQN that high σ = unreliable predictions. Without this, the uncertainty signal is meaningless.

### 2. Uncertainty-Aware Reward Shaping
The `uncertainty_urgency` term in the reward penalizes waiting under high σ near failure — 2.7× stronger than time pressure alone. The blind agent (σ zeroed out) has no urgency signal, learns "always WAIT," and crashes.

### 3. σ-Gated Safety Supervisor
Two-tier design: confident predictions → supervisor intervenes; uncertain predictions → supervisor backs off (trusts UA agent). Emergency tier fires regardless at RUL < 5 cycles.

### 4. End-of-Life Episode Bias
60% of training episodes start in the final 50 cycles (EOL zone). Without this, only ~13% of random starts land in the critical decision zone, and the agent learns "always WAIT."

### 5. Multi-Component Observation Design
Four signals capture both prediction and confidence: mean RUL (trajectory), instantaneous σ (spikes), rolling σ (trend), sensor trend (health). Each serves a distinct role in decision-making.

---

## Hyperparameters (`src/config.py`)

### Data & Preprocessing
| Parameter | Value | Purpose |
|-----------|-------|---------|
| WINDOW_SIZE | 30 | Temporal context per LSTM input |
| BATCH_SIZE | 64 | LSTM training batch |
| INPUT_DIM | 24 | 21 sensors + 3 settings |

### LSTM Ensemble
| Parameter | Value | Purpose |
|-----------|-------|---------|
| HIDDEN_DIM | 100 | LSTM hidden state size |
| NUM_LAYERS | 2 | Stacked LSTM layers |
| DROPOUT | 0.2 | Regularization |
| ENSEMBLE_SIZE | 5 | Number of models |
| EPOCHS | 80 | Max training epochs |
| LEARNING_RATE | 0.0005 | Adam optimizer LR |
| PATIENCE | 15 | Early stopping patience |
| GRAD_CLIP | 1.0 | Gradient clipping threshold |
| BOOTSTRAP_RATIO | 0.8 | Data fraction per model |

### DQN Agent
| Parameter | Value | Purpose |
|-----------|-------|---------|
| RL_TIMESTEPS | 1,500,000 | Total environment steps |
| RL_NET_ARCH | [256, 256, 128] | 3-layer MLP policy |
| RL_LR | 0.0003 | DQN learning rate |
| RL_BATCH | 128 | Replay buffer batch |
| RL_BUFFER | 150,000 | Replay buffer capacity |
| RL_LEARNING_STARTS | 10,000 | Steps before training begins |
| RL_EXPLORE_FRAC | 0.4 | ε-greedy exploration fraction |
| RL_EXPLORE_FINAL | 0.03 | Final exploration rate |
| RL_TARGET_UPDATE | 1,000 | Target network sync interval |
| RL_TRAIN_FREQ | 4 | Train every N steps |
| RL_GAMMA | 0.99 | Discount factor |

### Noise Augmentation
| Parameter | Value | Purpose |
|-----------|-------|---------|
| NOISE_PROB | 0.7 | 70% of episodes inject noise |
| NOISE_LEVEL | 0.20 | Max noise std |
| NOISE_LEVEL_MIN | 0.02 | Min noise std |
| UNCERTAINTY_SCALE | 15.0 | Scale raw σ (~0.01-0.10) → [0,1] |

### Safety Supervisor
| Parameter | Value | Purpose |
|-----------|-------|---------|
| CRITICAL_RUL_NORM | 0.12 | ~15 cycles, Tier 2 threshold |
| HARD_CRITICAL_RUL_NORM | 0.04 | ~5 cycles, Tier 1 (emergency) |
| SUPERVISOR_SIGMA_THRESHOLD | 0.55 | σ gate for Tier 2 override |

### Training Bias
| Parameter | Value | Purpose |
|-----------|-------|---------|
| EOL_EPISODE_PROB | 0.6 | 60% episodes start in EOL zone |
| EOL_WINDOW | 50 | Last 50 cycles = end-of-life |
| CRASH_PENALTY | -500 | Engine failure penalty |
| TIME_PRESSURE_START | 25 | Cycles from end where WAIT decays |

---

## File Structure

### Core Source (`src/`)
| File | Lines | Role |
|------|-------|------|
| `config.py` | 118 | All hyperparameters and device config |
| `lstm_model.py` | 39 | `RUL_LSTM` class — 2-layer LSTM for RUL prediction |
| `gym_env.py` | 276 | `PdMEnvironment` (Gymnasium env), reward shaping, `safety_override` |
| `preprocessing.py` | 108 | Data loading, MinMaxScaler, windowing for FD001+FD002 |

### Entry Points (root)
| File | Role |
|------|------|
| `main_train_ensemble.py` | Train 5 LSTM models with bootstrap sampling |
| `main_train_rl.py` | Train DQN agent with noise augmentation (outputs `dqn_pdm_agent.zip`) |
| `main_evaluate.py` | Quick comparison: hybrid system vs fixed-threshold baseline |
| `main_evaluate_ensemble.py` | LSTM prediction quality metrics (RMSE, C-MAPSS score) |
| `main_visualize.py` | 3-panel engine lifecycle figure |
| `main_experiment_final.py` | Publication-quality ablation (3 experiments × 8 noise levels × 500 episodes) |
| `main_experiment_ablation.py` | Two-experiment architecture validation |
| `main_experiment_threshold.py` | DQN vs rule-based threshold baselines |
| `main_cost_analysis.py` | Financial impact analysis (50-engine fleet) |
| `dashboard.py` | Streamlit interactive demo (883 lines) |

### Data (`data/`)
- `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` — 100 engines, 1 operating condition
- `train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt` — 260 engines, 6 operating conditions

### Trained Models (`models/`)
- `ensemble_model_0.pth` … `ensemble_model_4.pth` — 5 LSTM weights
- `dqn_pdm_agent.zip` — Uncertainty-aware DQN agent
- `dqn_blind_agent.zip` — Blind DQN agent (ablation baseline)
- `scaler.pkl` — MinMaxScaler fitted on training data

### Generated Outputs (root)
- `report_exp1_layer2_isolation.png` — Experiment 1: UA vs Blind without supervisor
- `report_exp2_full_system.png` — Experiment 2: Full 3-layer system
- `report_exp3_safety_contribution.png` — Experiment 3: Supervisor contribution
- `report_summary_dashboard.png` — Composite 2×3 overview
- `report_cost_analysis.png` — Financial impact visualization
- `report_experiment_results.csv` — All experiment statistics
- `report_cost_summary.csv` — Cost breakdown table

---

## Execution Pipeline

### Step 1: Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Train LSTM Ensemble (Layer 1)
```bash
python main_train_ensemble.py
```
Loads FD001+FD002 (360 engines, 81,880 sequences). Trains 5 LSTM models with bootstrap sampling. Outputs: `models/ensemble_model_{0-4}.pth`, `models/scaler.pkl`. Time: ~5-10 min (GPU) / ~30 min (CPU).

### Step 3: Train DQN Agent (Layer 2)
```bash
python main_train_rl.py
```
Creates environment with 70% noise augmentation. Trains DQN for 1.5M timesteps. Outputs: `models/dqn_pdm_agent.zip`. Quick eval: 30 episodes clean + 30 noisy. Time: ~30-45 min (CPU).

### Step 4: Quick Evaluation
```bash
python main_evaluate.py
```
Compares hybrid system vs fixed-threshold baseline. 200 episodes, clean data.

### Step 5: Publication Experiments
```bash
python main_experiment_final.py
```
3 experiments × 8 noise levels × 500 episodes = 12,000 evaluations. Generates all `report_exp*.png` figures and `report_experiment_results.csv`. Time: ~60-90 min.

### Step 6: Cost Analysis
```bash
python main_cost_analysis.py
```
50-engine fleet, 200 decisions/year. Generates `report_cost_analysis.png` and `report_cost_summary.csv`.

### Step 7: Visualization
```bash
python main_visualize.py
```
3-panel figure: RUL prediction + uncertainty bands, ensemble σ over time, decision timeline.

### Step 8: Interactive Dashboard
```bash
streamlit run dashboard.py
```
Real-time simulation: pick engine, toggle UA/Blind, enable/disable supervisor, inject noise. Shows all 3 layers working together.

---

## Results Summary

### Experiment 1: Layer 2 Isolation (No Safety Supervisor)
- **500 episodes × 8 noise levels** — UA vs Blind agent without supervisor
- At noise σ=0.15: UA scores +136 reward, Blind scores -113 (difference: +249, 9.5× better)
- Cohen's d = 0.65 at σ=0.175 (p < 0.001)
- UA retains 70% jackpot rate even at high noise

### Experiment 2: Full 3-Layer System
- **Zero catastrophic failures** for both agents (supervisor guarantees safety)
- UA autonomous success: 70% jackpots without supervisor help
- Blind autonomous success: only 3% jackpots, 43% of episodes need supervisor
- Blind requires 15× more supervisor overrides than UA

### Experiment 3: Safety Supervisor Contribution

| Metric | UA (Layer 2) | UA (Full) | Blind (Layer 2) | Blind (Full) |
|--------|-------------|-----------|-----------------|-------------|
| Avg Reward (σ=0.15) | +136 | +210 | -113 | +280 |
| Crash Rate | 8% | 0% | 45% | 0% |
| Jackpot Rate | 70% | 88% | 3% | 45% |
| Supervisor Overrides (per 500 eps) | 7 | 25 | 215 | 450 |

### Cost Analysis (50-Engine Fleet, σ=0.15)

| Cost Item | Value (GBP) |
|-----------|-------------|
| Jackpot maintenance | £18,000 |
| Safe maintenance | £35,000 |
| Wasteful maintenance | £52,000 |
| Unplanned failure (crash) | £275,000 |
| Supervisor override premium | +£12,000 |

| System | Annual Cost | Notes |
|--------|-------------|-------|
| UA system | £3.8M | 70% jackpots, minimal overrides |
| Blind system | £5.2M | 45% jackpots, massive override load |
| Fixed-threshold | £6.1M | High crash risk |
| **UA savings vs Blind** | **£1.4M/year (27%)** | |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch (LSTM, model training) |
| **Reinforcement Learning** | Stable-Baselines3 (DQN), Gymnasium (environment) |
| **Data Processing** | NumPy, Pandas, scikit-learn (MinMaxScaler) |
| **Statistics** | SciPy (Welch's t-test, Cohen's d) |
| **Visualization** | Matplotlib (publication figures, 300 DPI), Plotly (interactive charts) |
| **Dashboard** | Streamlit (real-time web interface) |
| **Serialization** | joblib (scaler), PyTorch (model weights), SB3 (agent zip) |
| **Hardware** | CUDA support with CPU fallback |
