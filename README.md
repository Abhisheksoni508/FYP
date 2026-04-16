# Uncertainty-Aware RL for Predictive Maintenance

> A three-layer predictive-maintenance framework that combines LSTM deep ensembles, a DQN maintenance policy, and a deterministic safety supervisor.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch&logoColor=white)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-DQN%20%7C%20PPO-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-29%2F29%20passing-brightgreen)

## Overview

This repository contains a final-year project on predictive maintenance for turbofan engines using the NASA C-MAPSS benchmark.

The system has three layers:

1. Layer 1: a five-member LSTM ensemble predicts remaining useful life (RUL) and produces a disagreement-based uncertainty signal.
2. Layer 2: a DQN agent decides whether to `WAIT` or `MAINTAIN` using both prediction and uncertainty.
3. Layer 3: a deterministic safety supervisor can override the agent when predicted failure risk is sufficiently high.

The central idea is simple: maintenance policies should not treat noisy predictions as if they were equally reliable.

## Key Result

Two findings matter most:

- In the no-supervisor comparison, the uncertainty-aware agent stayed positive at every tested noise level, while the blind baseline collapsed into negative reward under heavy noise.
- In the full three-layer system, both agents were kept near zero failure, but the uncertainty-aware agent was far more autonomous and much less dependent on supervisor intervention.

Clean-data headline numbers:

| Metric | Uncertainty-Aware | Blind Baseline |
|---|---:|---:|
| Mean reward (full system) | 396.6 | 432.1 |
| Supervisor overrides | 13 | 201 |
| Autonomy | 97.4% | 59.8% |
| Autonomous jackpots | 353 / 500 | 214 / 500 |

Financial headline:

- Average annual saving: GBP 556,950 for a representative 50-engine fleet.

## Dataset

The project uses NASA C-MAPSS FD001 and FD002.

| Subset | Engines | Rows |
|---|---:|---:|
| FD001 | 100 | 20,631 |
| FD002 | 260 | 53,759 |
| Combined | 360 | 74,390 |

After 30-cycle windowing, the combined dataset yields 63,590 windows.

Preprocessing choices:

- 24 features per cycle (21 sensors + 3 operating settings)
- MinMax scaling
- RUL capped at 125
- 30-cycle sliding windows

## Observation and Action Space

The DQN observation is:

```text
[mu_RUL, sigma_now, sigma_rolling, sensor_health]
```

where:

- `mu_RUL` is the ensemble mean prediction
- `sigma_now` is the current scaled ensemble disagreement
- `sigma_rolling` is a short rolling average of disagreement
- `sensor_health` is an aggregate term derived from the current normalized sensor vector

The action space is binary:

```text
WAIT or MAINTAIN
```

## Repository Structure

```text
src/
  config.py
  preprocessing.py
  lstm_model.py
  gym_env.py

dashboard.py
main_train_ensemble.py
main_train_rl.py
main_experiment_final.py
main_experiment_ppo.py
main_cost_analysis.py
main_visualize.py
generate_charts.py
test_system.py
main.tex
references.bib
figures/
data/
models/
```

## Installation

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Quick Start

Run the tests:

```bash
pytest test_system.py -v
```

Launch the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

## Reproducing the Core Pipeline

```bash
# Train the five-model ensemble
python main_train_ensemble.py

# Train the uncertainty-aware DQN
python main_train_rl.py

# Run the main experiments
python main_experiment_final.py
python main_experiment_ppo.py
python main_cost_analysis.py

# Regenerate evaluation figures
python main_evaluate_ensemble.py
python main_visualize.py
python generate_charts.py
```

## Interactive Dashboard

The dashboard is meant for viva/demo use as well as inspection:

- step through an engine lifecycle cycle by cycle
- compare uncertainty-aware and blind agents
- toggle the safety supervisor
- inject live sensor noise
- inspect RUL prediction, uncertainty, Q-values, and event logs

Recommended demo cases:

1. Engine 134 without extra noise
2. Engine 200 with moderate noise
3. Side-by-side uncertainty-aware vs blind comparison

## Testing

The repository includes a 29-test `pytest` suite covering:

- data loading and preprocessing
- LSTM model loading and diversity
- Gymnasium environment behavior
- reward classification
- safety supervisor logic
- end-to-end integration checks

## Dissertation

The audited and rebuilt dissertation PDF is here:

- [main.pdf](main.pdf)

The main source file is:

- [main.tex](main.tex)

## Notes on Claims

This is a research prototype, not a deployment-ready aerospace product.

Important limitations stated in the report:

- calibration is imperfect (PICP 0.61 at nominal 0.95)
- validation is on simulated C-MAPSS data, not live operator data
- the action space is binary
- DQN is the primary RL algorithm, with PPO included as a supplementary comparison

## Author

Abhishek Soni  
BSc Final Year Project  
University of Greenwich
