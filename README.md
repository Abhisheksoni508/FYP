# Uncertainty-Aware Reinforcement Learning for Predictive Maintenance

A 3-layer hybrid AI system that combines deep learning, reinforcement learning, and rule-based safety to optimise turbofan engine maintenance scheduling under sensor uncertainty.

## Architecture

| Layer | Component | Purpose |
|-------|-----------|---------|
| **Layer 1** | LSTM Deep Ensemble (×5) | Predicts Remaining Useful Life (RUL) and quantifies prediction uncertainty via ensemble disagreement (σ) |
| **Layer 2** | DQN RL Agent | Observes `[mean_rul, σ_now, σ_rolling, sensor_trend]` and learns when to trigger maintenance |
| **Layer 3** | Safety Supervisor | Deterministic override that forces maintenance when the ensemble is confident the engine is near failure |

### Key Innovation
The DQN agent is trained with **noise augmentation** (70% of training episodes inject Gaussian sensor noise). This teaches the agent that high ensemble disagreement (σ) means unreliable predictions, enabling uncertainty-aware decision-making. A **Blind** ablation agent (σ zeroed out) serves as the control.

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Simulation:
- **FD001**: 100 engines, single operating condition, single fault mode
- **FD002**: 260 engines, six operating conditions, single fault mode
- Combined: 360 engines, 21 sensors + 3 operational settings per cycle
- RUL capped at 125 cycles (standard practice)

## MDP Formulation

| Element | Definition |
|---------|-----------|
| **State** | `s = [μ_RUL, σ_now, σ_rolling, sensor_trend]` — normalised LSTM ensemble mean, instantaneous and rolling uncertainty, average sensor reading |
| **Actions** | `A = {WAIT, MAINTAIN}` — binary decision at each cycle |
| **Reward** | +500 (optimal timing, RUL < 20), +10 (acceptable, RUL < 50), −20 (wasteful, too early), −500 (crash) with time-pressure and uncertainty-urgency shaping |
| **Transition** | Deterministic — advances one cycle in the engine's recorded lifecycle |
| **Discount** | γ = 0.99 |

## Results

### Experiment 1 — Layer 2 Isolation (No Safety Supervisor)
UA agent outperforms Blind under sensor noise with **medium effect size** (Cohen's d = 0.65 at σ=0.175, p < 0.001). At heavy noise: UA retains positive reward (+136) while Blind collapses to −113.

### Experiment 2 — Full 3-Layer System
Safety supervisor eliminates **all** catastrophic failures. UA agent achieves **70% autonomous jackpot rate** vs Blind's 43% — Blind's higher overall score at σ=0 is inflated by supervisor-assisted jackpots (179 vs 7).

### Experiment 3 — Safety Supervisor Contribution
Blind agent requires **15× more** supervisor overrides than UA on clean data, demonstrating UA's self-sufficiency and Blind's dependency on external safety mechanisms.

## Project Structure

```
├── src/
│   ├── config.py           # All hyperparameters and constants
│   ├── preprocessing.py    # C-MAPSS data loading, RUL calculation, windowing
│   ├── lstm_model.py       # LSTM architecture (Layer 1)
│   └── gym_env.py          # Gymnasium RL environment + Safety Supervisor (Layers 2 & 3)
├── main_train_ensemble.py      # Train 5 LSTM models with bootstrap sampling
├── main_train_rl.py            # Train uncertainty-aware DQN agent
├── main_evaluate.py            # Hybrid system vs fixed-threshold baseline
├── main_evaluate_ensemble.py   # LSTM prediction quality (RMSE, C-MAPSS score)
├── main_visualize.py           # 3-panel engine lifecycle visualisation
├── main_experiment_final.py    # Full ablation study (3 experiments, publication figures)
├── main_experiment_threshold.py# DQN vs rule-based threshold baselines
├── main_cost_analysis.py       # Financial impact analysis (50-engine fleet)
├── dashboard.py                # Streamlit real-time demo dashboard
├── data/                       # NASA C-MAPSS datasets (FD001 + FD002)
└── models/                     # Trained model weights
```

## Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train LSTM ensemble (Layer 1)
python main_train_ensemble.py

# 3. Train DQN agent (Layer 2)
python main_train_rl.py

# 4. Evaluate
python main_evaluate.py              # Quick comparison
python main_evaluate_ensemble.py     # LSTM prediction charts
python main_experiment_final.py      # Full ablation (trains Blind agent if needed)
python main_cost_analysis.py         # Financial analysis

# 5. Visualise
python main_visualize.py             # Static engine lifecycle figures
streamlit run dashboard.py           # Interactive demo
```

## Requirements

Python 3.9+, PyTorch, Stable-Baselines3, Gymnasium, Streamlit, Plotly, scikit-learn, pandas, matplotlib, scipy.
