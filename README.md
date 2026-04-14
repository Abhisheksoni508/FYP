# 🛠️ Uncertainty-Aware RL for Predictive Maintenance

> **A Robust Predictive Maintenance Framework for IoT Systems via Uncertainty-Aware Reinforcement Learning**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch&logoColor=white)
![Stable--Baselines3](https://img.shields.io/badge/Stable--Baselines3-DQN%20%7C%20PPO-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-29%2F29%20passing-brightgreen)
![Status](https://img.shields.io/badge/status-FYP%20submission%20ready-success)

A **hybrid 3-layer AI framework** for turbofan predictive maintenance that combines:
1. **Deep ensemble LSTM forecasting** for Remaining Useful Life (RUL),
2. **Uncertainty-aware DQN decision-making** for maintenance timing,
3. **Safety-rule supervision** for hard-fail protection.

The system is designed to remain robust under noisy sensor conditions and demonstrates improved maintenance timing, lower avoidable failures, and lower annual fleet cost versus blind and fixed-threshold baselines.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Results](#-results)
- [Dataset](#-dataset)
- [MDP Formulation](#-mdp-formulation)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Reproducibility](#-reproducibility)
- [Interactive Dashboard](#-interactive-dashboard)
- [Testing](#-testing)
- [Tech Stack](#-tech-stack)
- [Dissertation](#-dissertation)
- [Citation](#-citation)
- [License & Attribution](#-license--attribution)

---

## 🔍 Overview

This project addresses predictive maintenance for turbofan engines using uncertainty-aware reinforcement learning.

Instead of relying only on a single RUL forecast, the framework explicitly models uncertainty and feeds it into the RL policy. This allows the agent to adapt behavior under noisy conditions and defer to a safety supervisor when the system is confidently close to failure.

**Author:** Abhishek Soni  
**Module:** COMP1682 (BSc Final Year Project)  
**Institution:** University of Greenwich  
**Supervisor:** Yasmine Arafa

---

## ✨ Key Features

- **Uncertainty-aware control:** RL agent observes both forecast and uncertainty.
- **Deep ensemble forecasting:** 5 bootstrap-trained LSTMs provide mean prediction and disagreement-based uncertainty.
- **Noise-robust policy learning:** 70% of RL training episodes include Gaussian sensor noise (σ ∈ [0.02, 0.20]).
- **Safety override layer:** Two-tier σ-gated supervisor can force emergency maintenance under high-confidence failure risk.
- **End-to-end evaluation:** Includes ablation, PPO comparison, statistical testing, and fleet-level cost analysis.
- **Interactive dashboard:** Streamlit app for per-engine policy inspection and behavior comparison.

---

## 📐 Architecture

```text
Layer 1: LSTM Deep Ensemble (x5)
  Input  -> sensor windows
  Output -> μ_RUL and uncertainty σ (inter-model disagreement)

Layer 2: DQN Agent
  Observation -> [μ_RUL, σ_now, σ_rolling, sensor_trend]
  Action      -> WAIT or MAINTAIN

Layer 3: Safety Supervisor
  Rule-based uncertainty-gated override
  Can force MAINTAIN under confident imminent-failure conditions
```

| Layer | Component | Role |
|---|---|---|
| **L1** | LSTM Deep Ensemble (×5) | Predicts RUL and quantifies uncertainty via model disagreement (σ) |
| **L2** | DQN Agent | Learns maintenance timing from `[μ_RUL, σ_now, σ_rolling, sensor_trend]` |
| **L3** | Safety Supervisor | Applies hard override when uncertainty and risk satisfy safety criteria |

---

## 📊 Results

| Metric | UA Agent | Blind Agent | Fixed Threshold |
|---|---:|---:|---:|
| **Crash rate (full system, σ=0.15)** | **0%** | 0% | — |
| **Autonomous JACKPOT rate** | **70%** | 3% | — |
| **Reward at σ=0.15 (Layer 2 isolated)** | **+136** | −113 | — |
| **Supervisor overrides per 500 episodes** | **7** | 215 | — |
| **Cohen's d (UA vs Blind)** | **0.65** *(p < 0.001)* | — | — |
| **Annual cost (50-engine fleet)** | **£3.8M** | £5.2M | £6.1M |
| **Savings vs Blind baseline** | **£1.4M/year (27%)** | — | — |

**Evaluation notes**
- Results are produced from the repository experiment scripts and summarised in `main.pdf`.
- “Layer 2 isolated” refers to RL policy evaluation without safety-layer intervention.
- “Full system” includes all 3 layers (forecasting + RL + safety supervisor).

---

## 🧠 Dataset

NASA **C-MAPSS** turbofan degradation subsets used:

| Subset | Engines | Operating Conditions | Cycles |
|---|---:|---:|---:|
| FD001 | 100 | 1 | 20,631 |
| FD002 | 260 | 6 | 61,249 |
| **Combined** | **360** | **Mixed** | **81,880** |

Preprocessing:
- 24 features/cycle (21 sensors + 3 operational settings)
- MinMaxScaler normalization
- RUL capped at 125
- 30-cycle sliding windows

---

## 🧮 MDP Formulation

| Element | Definition |
|---|---|
| **State** | `s = [μ_RUL, σ_now, σ_rolling, sensor_trend] ∈ [0,1]^4` |
| **Actions** | `A = {WAIT, MAINTAIN}` |
| **Reward** | `+500` JACKPOT (RUL < 20), `+10` SAFE (RUL < 50), `−20` WASTEFUL, `−500` CRASH, plus shaping terms |
| **Transition** | Deterministic environment step (advance one cycle) |
| **Termination** | MAINTAIN action or true RUL = 0 |
| **Discount** | γ = 0.99 |

---

## 📁 Repository Structure

```text
FYP/
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── lstm_model.py
│   └── gym_env.py
├── main_train_ensemble.py
├── main_train_rl.py
├── main_evaluate.py
├── main_evaluate_ensemble.py
├── main_experiment_final.py
├── main_experiment_ablation.py
├── main_experiment_threshold.py
├── main_experiment_ppo.py
├── main_cost_analysis.py
├── main_visualize.py
├── generate_charts.py
├── dashboard.py
├── test_system.py
├── main.tex / main.pdf
├── references.bib
├── figures/
├── data/
├── models/
└── report_*.csv
```

---

## ⚙️ Installation

```bash
# 1) Clone repository
git clone https://github.com/Abhisheksoni508/FYP.git
cd FYP

# 2) (Recommended) create virtual environment
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```bash
# Run tests first (expected: 29 passed)
pytest test_system.py -v

# Launch interactive dashboard
streamlit run dashboard.py
```

---

## 🔁 Reproducibility

To reproduce the full experimental pipeline from scratch:

```bash
# Layer 1: train 5-model LSTM ensemble (~25 min on GPU)
python main_train_ensemble.py

# Layer 2: train UA DQN (~40 min on CPU)
python main_train_rl.py

# Reproduce core experiments and outputs
python main_experiment_final.py
python main_experiment_ppo.py
python main_cost_analysis.py
python main_evaluate_ensemble.py
python main_visualize.py
python generate_charts.py
```

---

## 🖥️ Interactive Dashboard

The Streamlit dashboard enables step-by-step inspection of system behavior on individual engines:

- Engine selector (FD001/FD002)
- UA vs Blind policy comparison
- Safety supervisor toggle
- Sensor-noise injection slider (σ ∈ [0, 0.20])
- Side-by-side policy simulation
- Live plots for RUL ±σ, uncertainty trace, Q-values, and event logs

Run with:

```bash
streamlit run dashboard.py
```

---

## 🧪 Testing

A 29-test `pytest` suite validates core components:

| Test Class | Coverage |
|---|---|
| `TestDataPipeline` | Data loading, RUL capping, windowing, scaling |
| `TestLSTMEnsemble` | Model architecture, forward pass, uncertainty behavior |
| `TestRLEnvironment` | Observation/action space, reward correctness, termination |
| `TestSafetySupervisor` | Tiered override logic and σ-gating |
| `TestTrainedModels` | Model loading and inference shape checks |
| `TestEndToEnd` | Full pipeline from prediction to decision |

```bash
pytest test_system.py -v
```

---

## 🛠 Tech Stack

| Area | Technology |
|---|---|
| Deep Learning | PyTorch (LSTM ensemble) |
| Reinforcement Learning | Stable-Baselines3 (DQN, PPO), Gymnasium |
| Data | NumPy, pandas, scikit-learn |
| Statistics | SciPy (Welch’s t-test, Cohen’s d) |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Testing | pytest |
| Documentation | LaTeX, BibTeX |

---

## 📚 Dissertation

The full report is available as `main.pdf` (119 pages), including:
- Methodology and implementation details
- Statistical analysis and ablation studies
- PPO comparison and cost modeling
- Validation, ethics, and risk documentation

Build from source:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@misc{soni2026uncertaintyaware,
  title        = {Uncertainty-Aware Reinforcement Learning for Predictive Maintenance},
  author       = {Soni, Abhishek},
  year         = {2026},
  howpublished = {BSc Final Year Project, University of Greenwich},
  note         = {COMP1682 dissertation and code repository}
}
```

---

## 📜 License & Attribution

Academic work submitted in partial fulfilment of the BSc(Hons) Computer Science degree at the University of Greenwich.

NASA C-MAPSS dataset © NASA Prognostics Center of Excellence.

> If you want this repository to be open for reuse, add an explicit `LICENSE` file (e.g., MIT). Without a license, reuse rights are restricted by default.