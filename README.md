# 🛠️ Uncertainty-Aware RL for Predictive Maintenance

> **A Robust Predictive Maintenance Framework for IoT Systems via Uncertainty-Aware Reinforcement Learning**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch&logoColor=white)
![Stable--Baselines3](https://img.shields.io/badge/Stable--Baselines3-DQN%20%7C%20PPO-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-29%2F29%20passing-brightgreen)
![Status](https://img.shields.io/badge/status-FYP%20submission%20ready-success)

A **3-layer hybrid AI system** that combines deep ensemble uncertainty quantification, reinforcement learning, and rule-based safety to optimise turbofan engine maintenance scheduling under noisy sensor conditions.

> **Author** Abhishek Soni · **Module** COMP1682 (BSc Final Year Project) · **University of Greenwich** · **Supervisor** Yasmine Arafa

---

## 📐 The 3-Layer Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 1 — LSTM Deep Ensemble        Prediction + Uncertainty (σ)    │
│  5 × LSTMs (bootstrap)  ──►  μ_RUL ± σ                               │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 2 — DQN Agent                 Uncertainty-Aware Decisions     │
│  Obs: [μ, σ_now, σ_roll, trend]  ──►  WAIT  /  MAINTAIN              │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 3 — Safety Supervisor         Hard Safety Override            │
│  σ-gated 2-tier override  ──►  emergency MAINTAIN if confident       │
└──────────────────────────────────────────────────────────────────────┘
```

| Layer | Component | Role |
|---|---|---|
| **L1** | LSTM Deep Ensemble (×5) | Predicts Remaining Useful Life and quantifies uncertainty via inter-model disagreement (σ) |
| **L2** | DQN RL Agent | Observes `[μ_RUL, σ_now, σ_rolling, sensor_trend]` and learns the optimal maintenance moment |
| **L3** | Safety Supervisor | Two-tier σ-gated rule that forces MAINTAIN when the ensemble is confidently predicting imminent failure |

### 🎯 Key Innovation
The DQN is trained with **noise augmentation** — 70% of training episodes inject Gaussian sensor noise (σ ∈ [0.02, 0.20]). This forces ensemble disagreement to spike, teaching the agent that **high σ ⇒ unreliable predictions**. A **Blind** ablation agent (with σ zeroed out) is trained as the control to isolate the contribution of uncertainty awareness.

---

## 📊 Headline Results

| Metric | UA Agent | Blind Agent | Fixed Threshold |
|---|---|---|---|
| **Crash rate (full system, σ=0.15)** | **0%** | 0% | — |
| **Autonomous JACKPOT rate** | **70%** | 3% | — |
| **Reward at σ=0.15 (Layer 2 isolated)** | **+136** | −113 | — |
| **Supervisor overrides per 500 eps** | **7** | 215 | — |
| **Cohen's d (UA vs Blind)** | **0.65** *(p < 0.001)* | — | — |
| **Annual cost (50-engine fleet)** | **£3.8M** | £5.2M | £6.1M |
| **UA savings vs Blind baseline** | **£1.4M / year (27%)** | — | — |

> 📑 Full statistical analysis, ablation results, and PPO comparison available in the dissertation (`main.pdf`).

---

## 🧠 Dataset — NASA C-MAPSS Turbofan

| Subset | Engines | Op. Conditions | Cycles |
|---|---|---|---|
| FD001 | 100 | 1 | 20,631 |
| FD002 | 260 | 6 | 61,249 |
| **Combined** | **360** | **Mixed** | **81,880** |

24 features per cycle (21 sensors + 3 operational settings) · MinMaxScaler normalisation · RUL capped at 125 · 30-cycle sliding windows.

---

## 🧮 MDP Formulation

| Element | Definition |
|---|---|
| **State** | `s = [μ_RUL, σ_now, σ_rolling, sensor_trend] ∈ [0,1]⁴` |
| **Actions** | `A = {WAIT, MAINTAIN}` (binary) |
| **Reward** | `+500` JACKPOT (RUL<20) · `+10` SAFE (RUL<50) · `−20` WASTEFUL · `−500` CRASH · shaped step reward with time-pressure and uncertainty-urgency terms |
| **Transition** | Deterministic — advance one cycle |
| **Termination** | MAINTAIN action OR true RUL = 0 |
| **Discount** | γ = 0.99 |

---

## 📁 Project Structure

```
FINAL_FYP/
├── src/
│   ├── config.py                    # All hyperparameters
│   ├── preprocessing.py             # C-MAPSS loader, RUL calc, windowing
│   ├── lstm_model.py                # LSTM architecture (Layer 1)
│   └── gym_env.py                   # Gymnasium env + Safety Supervisor (L2+L3)
│
├── main_train_ensemble.py           # Train 5 LSTM models (bootstrap)
├── main_train_rl.py                 # Train UA DQN agent
├── main_evaluate.py                 # Hybrid vs fixed-threshold baseline
├── main_evaluate_ensemble.py        # LSTM prediction quality (RMSE, C-MAPSS)
├── main_experiment_final.py         # Publication-grade ablation (3 × 8 × 500)
├── main_experiment_ablation.py      # Architecture validation
├── main_experiment_threshold.py     # DQN vs rule-based baselines
├── main_experiment_ppo.py           # DQN vs PPO algorithm comparison
├── main_cost_analysis.py            # 50-engine fleet financial impact
├── main_visualize.py                # 3-panel engine lifecycle figures
├── generate_charts.py               # PPO + test result charts
├── dashboard.py                     # Streamlit interactive demo
├── test_system.py                   # Pytest suite (29 tests, 6 classes)
│
├── main.tex / main.pdf              # Dissertation source + compiled report
├── references.bib                   # Bibliography
├── figures/                         # All 29 figures used in main.pdf
├── data/                            # NASA C-MAPSS FD001 + FD002
├── models/                          # Trained LSTM ensemble + DQN/Blind agents
└── report_*.csv                     # Experiment + cost result tables
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the test suite (sanity check — should be 29/29 ✓)
pytest test_system.py -v

# 3. Launch the interactive dashboard (no training required — uses bundled models)
streamlit run dashboard.py
```

That's it for inspecting the work. To **reproduce the full results** from scratch:

```bash
# Train Layer 1 (5 LSTM models, ~25 min on GPU)
python main_train_ensemble.py

# Train Layer 2 (UA DQN agent, ~40 min on CPU)
python main_train_rl.py

# Reproduce all dissertation experiments + figures
python main_experiment_final.py        # Full ablation (3 × 8 × 500 episodes)
python main_experiment_ppo.py          # DQN vs PPO comparison
python main_cost_analysis.py           # Financial impact analysis
python main_evaluate_ensemble.py       # LSTM prediction quality
python main_visualize.py               # Engine lifecycle figures
python generate_charts.py              # PPO + test result charts
```

---

## 🖥️ Interactive Dashboard

A real-time Streamlit dashboard visualises every layer of the system as it makes decisions on individual engines:

- **Engine selector** for any FD001 or FD002 unit
- **UA / Blind toggle** to live-compare uncertainty-aware vs ablation agents
- **Safety supervisor toggle** to show / hide Layer 3 interventions
- **Sensor noise injection slider** (σ ∈ [0, 0.20]) — watch ensemble disagreement react in real time
- **Side-by-side mode** runs both agents on the same engine simultaneously
- Live plots: RUL prediction with ±σ band, ensemble uncertainty trace, DQN Q-value pressure, event log, decision badges (`JACKPOT` / `SAFE` / `EARLY` / `OVERRIDE`)

```bash
streamlit run dashboard.py
```

---

## 🧪 Test Suite

A 29-test pytest suite (`test_system.py`) verifies the integrity of every layer:

| Test class | Coverage |
|---|---|
| `TestDataPipeline` | Data loading, RUL capping, windowing, scaler integrity |
| `TestLSTMEnsemble` | Architecture, forward pass, ensemble disagreement |
| `TestRLEnvironment` | Observation/action space, reward correctness, episode termination |
| `TestSafetySupervisor` | Tier 1/Tier 2 override logic, σ-gating |
| `TestTrainedModels` | Model loading, inference shapes |
| `TestEndToEnd` | Full prediction → decision pipeline |

```bash
pytest test_system.py -v   # Expected: 29 passed in ~15s
```

---

## 📚 Dissertation

The full report (`main.pdf`, **119 pages**) contains:

1. **Chapters 1–7** — Introduction, literature review, methodology, implementation, results, discussion, conclusion (~12,500 words)
2. **Appendix A** — Original project proposal
3. **Appendix B** — Extended methodology (incl. PPO comparison study)
4. **Appendix C** — Ethics
5. **Appendix D** — Risk register
6. **Appendix E** — Statistical methods
7. **Appendix F** — Testing & validation (incl. dashboard usability evidence)
8. **Appendix G** — Extended implementation details

To rebuild from source:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch (LSTM ensemble) |
| Reinforcement Learning | Stable-Baselines3 (DQN, PPO) · Gymnasium |
| Data | NumPy · pandas · scikit-learn (MinMaxScaler) |
| Statistics | SciPy (Welch's t-test, Cohen's d) |
| Visualisation | Matplotlib (300 DPI figures) · Plotly (interactive) |
| Dashboard | Streamlit |
| Testing | pytest |
| Report | LaTeX (TikZ diagrams, BibTeX) |

---

## 📋 Requirements

Python 3.9+ · See [`requirements.txt`](requirements.txt) for the full pinned dependency list. CUDA-capable GPU recommended for LSTM training (CPU works but slower); DQN training is CPU-friendly.

---

## 📜 Licence & Attribution

Academic work submitted in partial fulfilment of the BSc(Hons) Computer Science degree at the University of Greenwich. NASA C-MAPSS dataset © NASA Prognostics Center of Excellence.
