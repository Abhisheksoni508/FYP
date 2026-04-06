"""
=================================================================
  ENGINE HEALTH MONITORING DASHBOARD v2
  Uncertainty-Aware RL for Predictive Maintenance
=================================================================

DEMO for FYP presentation. Interactive web dashboard that simulates
what a real maintenance engineer would see if this system was deployed.

HOW IT WORKS:
  1. Loads your trained models (LSTM ensemble + DQN agent)
  2. Lets you pick an engine from the C-MAPSS dataset
  3. Steps through the engine's lifecycle cycle-by-cycle
  4. At each cycle, runs the FULL 3-layer pipeline live:
     - Layer 1: LSTM ensemble predicts RUL + uncertainty
     - Layer 2: DQN agent decides WAIT or MAINTAIN
     - Layer 3: Safety supervisor can override
  5. Shows everything with live charts

USAGE:
  streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os

from src.config import *
from src.preprocessing import load_combined_data, calculate_rul, process_data
from src.lstm_model import RUL_LSTM
from src.gym_env import safety_override


# =================================================================
# 1. MODEL LOADING (cached)
# =================================================================

@st.cache_resource
def load_all_models():
    lstm_models = []
    for i in range(ENSEMBLE_SIZE):
        path = f"models/ensemble_model_{i}.pth"
        if os.path.exists(path):
            model = RUL_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            lstm_models.append(model)

    from stable_baselines3 import DQN
    dqn_agent = DQN.load("models/dqn_pdm_agent")

    blind_agent = None
    if os.path.exists("models/dqn_blind_agent.zip"):
        blind_agent = DQN.load("models/dqn_blind_agent")

    df = load_combined_data()
    df = calculate_rul(df)
    df_raw = df.copy()
    df_clean, _ = process_data(df, DROP_SENSORS, DROP_SETTINGS)

    return lstm_models, dqn_agent, blind_agent, df_clean, df_raw


def predict_ensemble(models, seq_tensor):
    preds = []
    with torch.no_grad():
        for model in models:
            pred = model(seq_tensor).item()
            preds.append(pred)
    return np.mean(preds), np.std(preds), preds


def get_q_values(agent, obs):
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_values = agent.q_net(obs_tensor).cpu().numpy()[0]
    return float(q_values[0]), float(q_values[1])


# =================================================================
# 2. SIMULATION STEP
# =================================================================

def run_one_cycle(engine_data, engine_raw, cycle_idx, lstm_models, agent,
                  inject_noise=False, noise_level=0.15, use_safety=True,
                  sigma_history=None, is_blind=False):
    features = [c for c in engine_data.columns if c not in ['unit', 'time', 'RUL']]
    seq = engine_data.iloc[cycle_idx - WINDOW_SIZE:cycle_idx][features].values.copy()

    if inject_noise:
        noise = np.random.normal(0, noise_level, seq.shape)
        seq = np.clip(seq + noise, 0, 1)

    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
    mean_pred, std_pred, individual_preds = predict_ensemble(lstm_models, seq_tensor)

    norm_mean = np.clip(mean_pred, 0, 1)
    norm_std = np.clip(std_pred * UNCERTAINTY_SCALE, 0, 1)
    sensor_trend = np.clip(np.mean(seq[-1, :]), 0, 1)

    if sigma_history is None:
        sigma_history = [0.0, 0.0, 0.0]
    rolling_sigma = float(np.mean(sigma_history))

    obs = np.array([norm_mean, norm_std, rolling_sigma, sensor_trend], dtype=np.float32)

    if is_blind:
        obs_for_agent = obs.copy()
        obs_for_agent[1] = 0.0
        obs_for_agent[2] = 0.0
    else:
        obs_for_agent = obs

    dqn_action, _ = agent.predict(obs_for_agent, deterministic=True)
    dqn_action = int(dqn_action)
    q_wait, q_maintain = get_q_values(agent, obs_for_agent)

    final_action = dqn_action
    was_overridden = False
    if use_safety:
        final_action, was_overridden = safety_override(dqn_action, obs_for_agent)

    true_rul = engine_raw.iloc[cycle_idx]['RUL']

    return {
        'true_rul': true_rul,
        'pred_rul': max(0, mean_pred * 125.0),
        'uncertainty': std_pred * 125.0,
        'norm_std': norm_std,
        'rolling_sigma': rolling_sigma,
        'individual_preds': [max(0, p * 125.0) for p in individual_preds],
        'sensor_trend': sensor_trend,
        'dqn_action': dqn_action,
        'final_action': final_action,
        'was_overridden': was_overridden,
        'q_wait': q_wait,
        'q_maintain': q_maintain,
        'obs': obs,
    }


# =================================================================
# 3. DASHBOARD UI
# =================================================================

def main():
    st.set_page_config(
        page_title="Engine Health Monitor",
        page_icon="⚙️",
        layout="wide"
    )

    # ── CSS ──
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        /* Global */
        .stApp { background: #0a0e1a; }
        .block-container { padding-top: 1.5rem; }

        /* Header */
        .dash-header {
            background: linear-gradient(135deg, #0f1629 0%, #1a1f3a 50%, #0d1117 100%);
            border: 1px solid rgba(99, 140, 255, 0.15);
            border-radius: 16px;
            padding: 20px 28px;
            margin-bottom: 18px;
            position: relative;
            overflow: hidden;
        }
        .dash-header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
        }
        .dash-title {
            font-family: 'Inter', sans-serif;
            font-size: 1.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .dash-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            color: #64748b;
            margin-top: 2px;
        }

        /* Layer badges */
        .layer-strip {
            display: flex;
            gap: 8px;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        .layer-chip {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 20px;
            letter-spacing: 0.3px;
        }
        .chip-l1 { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
        .chip-l2 { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
        .chip-l3 { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
        .chip-active { box-shadow: 0 0 8px rgba(99,140,255,0.4); }

        /* Status cards */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 10px;
            margin-bottom: 16px;
        }
        @media (max-width: 900px) {
            .metric-grid { grid-template-columns: repeat(3, 1fr); }
        }
        .metric-card {
            background: linear-gradient(135deg, #131829, #1a2040);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 14px 12px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
        }
        .mc-safe::before     { background: #22c55e; }
        .mc-warning::before  { background: #eab308; }
        .mc-danger::before   { background: #ef4444; }
        .mc-info::before     { background: #3b82f6; }
        .mc-purple::before   { background: #a855f7; }
        .mc-cyan::before     { background: #06b6d4; }

        .mc-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.65rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .mc-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.3rem;
            font-weight: 800;
            margin: 4px 0 2px 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .mc-value-safe    { color: #4ade80; }
        .mc-value-warning { color: #fbbf24; }
        .mc-value-danger  { color: #f87171; }
        .mc-value-info    { color: #60a5fa; }
        .mc-value-purple  { color: #c084fc; }
        .mc-value-cyan    { color: #22d3ee; }
        .mc-sub {
            font-family: 'Inter', sans-serif;
            font-size: 0.68rem;
            color: #475569;
        }

        /* Pipeline flow */
        .pipeline-flow {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0;
            margin: 12px 0 16px 0;
            padding: 12px 20px;
            background: linear-gradient(135deg, #0f1629, #151d35);
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 12px;
        }
        .pipe-node {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 8px;
            text-align: center;
            min-width: 120px;
        }
        .pipe-arrow {
            color: #334155;
            font-size: 1.2rem;
            margin: 0 6px;
        }
        .pn-active { box-shadow: 0 0 12px rgba(99,140,255,0.3); }
        .pn-l1 { background: rgba(59,130,246,0.12); color: #60a5fa; border: 1px solid rgba(59,130,246,0.25); }
        .pn-l2 { background: rgba(34,197,94,0.12); color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
        .pn-l3 { background: rgba(239,68,68,0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
        .pn-out { background: rgba(168,85,247,0.12); color: #c084fc; border: 1px solid rgba(168,85,247,0.25); }

        /* Event log */
        .event-log {
            background: #080c18;
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 12px 16px;
            max-height: 200px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            line-height: 1.85;
        }
        .ev-jackpot  { color: #4ade80; }
        .ev-safe     { color: #60a5fa; }
        .ev-override { color: #c084fc; }
        .ev-crash    { color: #f87171; }
        .ev-early    { color: #fbbf24; }
        .ev-info     { color: #64748b; }

        /* Outcome banner */
        .outcome-banner {
            border-radius: 12px;
            padding: 16px 20px;
            margin-top: 10px;
            text-align: center;
            font-family: 'Inter', sans-serif;
        }
        .ob-jackpot  { background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05)); border: 1px solid rgba(34,197,94,0.3); }
        .ob-safe     { background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(59,130,246,0.05)); border: 1px solid rgba(59,130,246,0.3); }
        .ob-override { background: linear-gradient(135deg, rgba(168,85,247,0.15), rgba(168,85,247,0.05)); border: 1px solid rgba(168,85,247,0.3); }
        .ob-crash    { background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05)); border: 1px solid rgba(239,68,68,0.3); }
        .ob-early    { background: linear-gradient(135deg, rgba(234,179,8,0.15), rgba(234,179,8,0.05)); border: 1px solid rgba(234,179,8,0.3); }
        .ob-title { font-size: 1.3rem; font-weight: 800; margin-bottom: 4px; }
        .ob-detail { font-size: 0.82rem; color: #94a3b8; }

        /* Progress bar */
        .progress-container {
            background: #131829;
            border-radius: 8px;
            padding: 8px 14px;
            margin-bottom: 14px;
            border: 1px solid rgba(255,255,255,0.04);
        }
        .progress-bar-bg {
            background: #1e293b;
            border-radius: 6px;
            height: 8px;
            overflow: hidden;
            margin-top: 4px;
        }
        .progress-bar-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.3s ease;
        }
        .progress-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: #64748b;
            display: flex;
            justify-content: space-between;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #0d1117;
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #e2e8f0;
        }

        /* Summary stats grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }
        .stat-item {
            background: #131829;
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 8px;
            padding: 10px 12px;
            text-align: center;
        }
        .stat-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.62rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
            font-weight: 700;
            color: #e2e8f0;
            margin-top: 2px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <div class="dash-header">
        <p class="dash-title">⚙️ Engine Health Monitoring Dashboard</p>
        <p class="dash-subtitle">Uncertainty-Aware Reinforcement Learning for Predictive Maintenance &nbsp;|&nbsp; 3-Layer Hybrid AI System</p>
        <div class="layer-strip">
            <span class="layer-chip chip-l1">L1 · LSTM Ensemble</span>
            <span class="layer-chip chip-l2">L2 · DQN Agent</span>
            <span class="layer-chip chip-l3">L3 · Safety Supervisor</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Models ──
    with st.spinner("Loading trained models..."):
        lstm_models, dqn_agent, blind_agent, df_clean, df_raw = load_all_models()

    # =============================================================
    # SIDEBAR
    # =============================================================
    st.sidebar.markdown("### Simulation Controls")

    all_units = sorted(df_clean['unit'].unique())
    unit_lengths = {u: len(df_clean[df_clean['unit'] == u]) for u in all_units}

    st.sidebar.markdown("**Engine Selection**")
    engine_id = st.sidebar.selectbox(
        "Engine Unit",
        all_units,
        index=all_units.index(134) if 134 in all_units else 0,
        format_func=lambda u: f"Engine {u}  ({unit_lengths[u]} cyc, {'FD001' if u <= 100 else 'FD002'})"
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Layer Controls**")

    agent_type = st.sidebar.radio(
        "Layer 2: Agent Type",
        ["Uncertainty-Aware (UA)", "Blind (No σ)"],
        help="UA agent uses uncertainty in decisions. Blind agent ignores it."
    )

    use_safety = st.sidebar.checkbox(
        "Layer 3: Safety Supervisor",
        value=True,
        help="Forces maintenance when predicted RUL is critically low and ensemble is confident."
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Sensor Noise**")
    inject_noise = st.sidebar.checkbox("Inject Sensor Noise", value=False)
    noise_level = st.sidebar.slider(
        "Noise Level (σ)", 0.01, 0.10, 0.05, 0.01,
        disabled=not inject_noise,
        help="Simulates sensor degradation. Watch uncertainty spike!"
    )

    st.sidebar.divider()
    speed = st.sidebar.slider("Simulation Speed (sec/cycle)", 0.05, 1.0, 0.25, 0.05)

    # Comparison mode
    st.sidebar.divider()
    st.sidebar.markdown("**Advanced**")
    comparison_mode = st.sidebar.checkbox(
        "Side-by-Side Comparison",
        value=False,
        help="Run UA and Blind agents simultaneously on the same engine"
    )

    # Select agent
    is_blind = agent_type == "Blind (No σ)"
    if is_blind and blind_agent is not None:
        active_agent = blind_agent
    else:
        active_agent = dqn_agent
        if is_blind and blind_agent is None:
            st.sidebar.warning("Blind agent not found.")

    # =============================================================
    # SESSION STATE
    # =============================================================
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'history_blind' not in st.session_state:
        st.session_state.history_blind = []
    if 'events' not in st.session_state:
        st.session_state.events = []
    if 'events_blind' not in st.session_state:
        st.session_state.events_blind = []
    if 'outcome' not in st.session_state:
        st.session_state.outcome = None
    if 'outcome_blind' not in st.session_state:
        st.session_state.outcome_blind = None
    if 'current_engine' not in st.session_state:
        st.session_state.current_engine = None
    if 'sigma_history' not in st.session_state:
        st.session_state.sigma_history = [0.0, 0.0, 0.0]
    if 'sigma_history_blind' not in st.session_state:
        st.session_state.sigma_history_blind = [0.0, 0.0, 0.0]
    if 'noise_seed' not in st.session_state:
        st.session_state.noise_seed = None

    # Reset if engine changed
    if st.session_state.current_engine != engine_id:
        for key in ['history', 'history_blind', 'events', 'events_blind',
                     'outcome', 'outcome_blind']:
            st.session_state[key] = [] if 'history' in key or 'events' in key else None
        st.session_state.running = False
        st.session_state.current_engine = engine_id
        st.session_state.sigma_history = [0.0, 0.0, 0.0]
        st.session_state.sigma_history_blind = [0.0, 0.0, 0.0]
        st.session_state.noise_seed = None

    # ── Buttons ──
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        start = st.button("▶  Start Simulation", use_container_width=True, type="primary")
    with col_b2:
        stop = st.button("⏹  Stop", use_container_width=True)
    with col_b3:
        reset = st.button("↺  Reset", use_container_width=True)

    if start:
        st.session_state.running = True
        if st.session_state.noise_seed is None:
            st.session_state.noise_seed = np.random.randint(0, 100000)
    if stop:
        st.session_state.running = False
    if reset:
        for key in ['history', 'history_blind', 'events', 'events_blind']:
            st.session_state[key] = []
        st.session_state.outcome = None
        st.session_state.outcome_blind = None
        st.session_state.running = False
        st.session_state.sigma_history = [0.0, 0.0, 0.0]
        st.session_state.sigma_history_blind = [0.0, 0.0, 0.0]
        st.session_state.noise_seed = None
        st.rerun()

    # ── Placeholders ──
    progress_ph = st.empty()
    pipeline_ph = st.empty()
    metric_ph = st.empty()
    chart_ph = st.empty()
    bottom_ph = st.empty()

    # ── Engine data ──
    engine_data = df_clean[df_clean['unit'] == engine_id].reset_index(drop=True)
    engine_raw = df_raw[df_raw['unit'] == engine_id].reset_index(drop=True)
    max_cycles = len(engine_data)

    # =============================================================
    # SIMULATION LOOP
    # =============================================================
    rendered_in_loop = False

    # Determine if we should keep running
    # In comparison mode, keep going until BOTH agents have outcomes
    use_comparison = comparison_mode and blind_agent is not None
    primary_done = st.session_state.outcome is not None
    blind_done = st.session_state.outcome_blind is not None
    should_run = st.session_state.running and (
        not primary_done or (use_comparison and not blind_done)
    )

    if should_run:
        start_idx = WINDOW_SIZE + len(st.session_state.history)

        for cycle_idx in range(start_idx, max_cycles):
            if not st.session_state.running:
                break

            # Set noise seed for reproducibility in comparison mode
            if inject_noise and st.session_state.noise_seed is not None:
                np.random.seed(st.session_state.noise_seed + cycle_idx)

            # ── Primary agent (skip if already done) ──
            if st.session_state.outcome is None:
                result = run_one_cycle(
                    engine_data, engine_raw, cycle_idx,
                    lstm_models, active_agent,
                    inject_noise=inject_noise,
                    noise_level=noise_level,
                    use_safety=use_safety,
                    sigma_history=list(st.session_state.sigma_history),
                    is_blind=is_blind
                )
                st.session_state.sigma_history.pop(0)
                st.session_state.sigma_history.append(result['norm_std'])
                st.session_state.history.append(result)
            else:
                # Primary is done but we're still running for blind agent
                # Append a copy of the last result to keep cycle counts aligned
                result = st.session_state.history[-1]

            # ── Comparison: Blind agent (if enabled) ──
            if use_comparison and st.session_state.outcome_blind is None:
                if inject_noise and st.session_state.noise_seed is not None:
                    np.random.seed(st.session_state.noise_seed + cycle_idx)

                result_b = run_one_cycle(
                    engine_data, engine_raw, cycle_idx,
                    lstm_models, blind_agent,
                    inject_noise=inject_noise,
                    noise_level=noise_level,
                    use_safety=use_safety,
                    sigma_history=list(st.session_state.sigma_history_blind),
                    is_blind=True
                )
                st.session_state.sigma_history_blind.pop(0)
                st.session_state.sigma_history_blind.append(result_b['norm_std'])
                st.session_state.history_blind.append(result_b)

                # Log blind events
                cycle_num = cycle_idx - WINDOW_SIZE + 1
                if result_b['was_overridden']:
                    st.session_state.events_blind.append(
                        ('override', f"Cycle {cycle_num}: SAFETY OVERRIDE (true RUL={result_b['true_rul']:.0f})")
                    )
                    st.session_state.outcome_blind = 'safety_override'
                elif result_b['final_action'] == 1:
                    true_b = result_b['true_rul']
                    lbl = 'jackpot' if true_b <= 20 else ('safe' if true_b <= 50 else 'early')
                    st.session_state.events_blind.append(
                        (lbl, f"Cycle {cycle_num}: {lbl.upper()} (true RUL={true_b:.0f})")
                    )
                    st.session_state.outcome_blind = f'dqn_{lbl}'
                elif cycle_idx >= max_cycles - 1:
                    st.session_state.events_blind.append(
                        ('crash', f"Cycle {cycle_num}: ENGINE FAILURE")
                    )
                    st.session_state.outcome_blind = 'crash'

            # ── Log primary events ──
            if st.session_state.outcome is None:
                cycle_num = cycle_idx - WINDOW_SIZE + 1
                true_rul = result['true_rul']

                if result['was_overridden']:
                    st.session_state.events.append(
                        ('override', f"Cycle {cycle_num}: SAFETY OVERRIDE — Forced maintenance "
                         f"(pred={result['pred_rul']:.0f}, true={true_rul:.0f})")
                    )
                    st.session_state.outcome = 'safety_override'
                    if not use_comparison or st.session_state.outcome_blind is not None:
                        st.session_state.running = False
                elif result['final_action'] == 1:
                    if true_rul <= 20:
                        label, etype = "JACKPOT", 'jackpot'
                    elif true_rul <= 50:
                        label, etype = "SAFE", 'safe'
                    else:
                        label, etype = "EARLY (wasteful)", 'early'
                    st.session_state.events.append(
                        (etype, f"Cycle {cycle_num}: {label} — DQN triggered "
                         f"(pred={result['pred_rul']:.0f}, true={true_rul:.0f})")
                    )
                    st.session_state.outcome = 'dqn_maintain'
                    if not use_comparison or st.session_state.outcome_blind is not None:
                        st.session_state.running = False
                elif cycle_idx >= max_cycles - 1:
                    st.session_state.events.append(
                        ('crash', f"Cycle {cycle_num}: ENGINE FAILURE — No maintenance triggered!")
                    )
                    st.session_state.outcome = 'crash'
                    st.session_state.running = False

            # Check if both done in comparison mode
            if use_comparison and st.session_state.outcome is not None and st.session_state.outcome_blind is not None:
                st.session_state.running = False

            # ── Render ──
            render_dashboard(
                st.session_state.history, st.session_state.events,
                st.session_state.outcome, st.session_state.history_blind,
                st.session_state.events_blind, st.session_state.outcome_blind,
                progress_ph, pipeline_ph, metric_ph, chart_ph, bottom_ph,
                agent_type, use_safety, inject_noise, max_cycles,
                comparison_mode and blind_agent is not None
            )
            rendered_in_loop = True

            if not st.session_state.running:
                break
            time.sleep(speed)

    # Static render
    if st.session_state.history and not rendered_in_loop:
        render_dashboard(
            st.session_state.history, st.session_state.events,
            st.session_state.outcome, st.session_state.history_blind,
            st.session_state.events_blind, st.session_state.outcome_blind,
            progress_ph, pipeline_ph, metric_ph, chart_ph, bottom_ph,
            agent_type, use_safety, inject_noise, max_cycles,
            comparison_mode and blind_agent is not None
        )
    elif not st.session_state.history:
        st.info(
            f"⚙️ Engine {engine_id} loaded ({max_cycles} cycles, "
            f"{'FD001' if engine_id <= 100 else 'FD002'}). "
            f"Press **Start Simulation** to begin."
        )


# =================================================================
# 4. RENDERING
# =================================================================

def render_dashboard(history, events, outcome, history_blind, events_blind,
                     outcome_blind, progress_ph, pipeline_ph, metric_ph,
                     chart_ph, bottom_ph, agent_type, use_safety,
                     inject_noise, max_cycles, show_comparison):
    latest = history[-1]
    n = len(history)
    total_usable = max_cycles - WINDOW_SIZE

    # ── Progress Bar ──
    pct = min(n / total_usable * 100, 100)
    if latest['true_rul'] > 50:
        bar_color = "#22c55e"
    elif latest['true_rul'] > 20:
        bar_color = "#eab308"
    else:
        bar_color = "#ef4444"

    with progress_ph.container():
        st.markdown(f"""
        <div class="progress-container">
            <div class="progress-label">
                <span>Engine Lifecycle</span>
                <span>Cycle {n} / {total_usable} &nbsp;({pct:.0f}%)</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width: {pct}%; background: {bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Pipeline Flow ──
    l1_active = "pn-active"
    l2_active = "pn-active"
    l3_active = "pn-active" if latest['was_overridden'] else ""
    out_action = "MAINTAIN" if latest['final_action'] == 1 else "WAIT"
    out_color = "#4ade80" if latest['final_action'] == 1 else "#60a5fa"
    dqn_label = "MAINTAIN" if latest['dqn_action'] == 1 else "WAIT"
    safety_label = "OVERRIDE!" if latest['was_overridden'] else ("ON" if use_safety else "OFF")
    pred_val = latest['pred_rul']
    unc_val = latest['uncertainty']

    with pipeline_ph.container():
        st.markdown(f"""
        <div class="pipeline-flow">
            <div class="pipe-node pn-l1 {l1_active}">
                SENSORS<br>
                <span style="font-size:0.6rem;opacity:0.7">30x24 window</span>
            </div>
            <span class="pipe-arrow">&rarr;</span>
            <div class="pipe-node pn-l1 {l1_active}">
                L1: LSTM x5<br>
                <span style="font-size:0.6rem;opacity:0.7">RUL={pred_val:.0f} sig={unc_val:.1f}</span>
            </div>
            <span class="pipe-arrow">&rarr;</span>
            <div class="pipe-node pn-l2 {l2_active}">
                L2: DQN<br>
                <span style="font-size:0.6rem;opacity:0.7">{dqn_label}</span>
            </div>
            <span class="pipe-arrow">&rarr;</span>
            <div class="pipe-node pn-l3 {l3_active}">
                L3: Safety<br>
                <span style="font-size:0.6rem;opacity:0.7">{safety_label}</span>
            </div>
            <span class="pipe-arrow">&rarr;</span>
            <div class="pipe-node pn-out pn-active">
                OUTPUT<br>
                <span style="font-size:0.6rem;color:{out_color}">{out_action}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Metric Cards ──
    pred_rul = latest['pred_rul']
    true_rul = latest['true_rul']
    unc = latest['uncertainty']
    rolling_s = latest['rolling_sigma']
    pred_err = abs(pred_rul - true_rul)

    def rul_style(val):
        if val > 50: return "mc-safe", "mc-value-safe"
        elif val > 20: return "mc-warning", "mc-value-warning"
        else: return "mc-danger", "mc-value-danger"

    def unc_style(val):
        if val < 3: return "mc-safe", "mc-value-safe"
        elif val < 8: return "mc-warning", "mc-value-warning"
        else: return "mc-danger", "mc-value-danger"

    pc, pv = rul_style(pred_rul)
    tc, tv = rul_style(true_rul)
    uc, uv = unc_style(unc)

    action_text = "MAINTAIN" if latest['dqn_action'] == 1 else "WAIT"
    dqn_card_class = "mc-warning" if latest['dqn_action'] == 1 else "mc-info"
    dqn_value_class = "mc-value-warning" if latest['dqn_action'] == 1 else "mc-value-info"
    agent_label = "UA" if "UA" in agent_type else "Blind"

    if outcome == 'safety_override':
        status, sc, sv = "OVERRIDE", "mc-purple", "mc-value-purple"
    elif outcome == 'crash':
        status, sc, sv = "CRASHED", "mc-danger", "mc-value-danger"
    elif outcome == 'dqn_maintain':
        status, sc, sv = "DONE", "mc-cyan", "mc-value-cyan"
    else:
        status, sc, sv = "RUNNING", "mc-safe", "mc-value-safe"

    with metric_ph.container():
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card {pc}">
                <div class="mc-label">Predicted RUL</div>
                <div class="mc-value {pv}">{pred_rul:.0f}</div>
                <div class="mc-sub">cycles remaining</div>
            </div>
            <div class="metric-card {tc}">
                <div class="mc-label">True RUL</div>
                <div class="mc-value {tv}">{true_rul:.0f}</div>
                <div class="mc-sub">ground truth</div>
            </div>
            <div class="metric-card {uc}">
                <div class="mc-label">Uncertainty</div>
                <div class="mc-value {uv}">{unc:.1f}</div>
                <div class="mc-sub">ensemble disagreement</div>
            </div>
            <div class="metric-card mc-info">
                <div class="mc-label">Rolling Sigma</div>
                <div class="mc-value mc-value-info">{rolling_s:.2f}</div>
                <div class="mc-sub">3-cycle average</div>
            </div>
            <div class="metric-card {dqn_card_class}">
                <div class="mc-label">DQN Decision</div>
                <div class="mc-value {dqn_value_class}">{action_text}</div>
                <div class="mc-sub">{agent_label} Agent</div>
            </div>
            <div class="metric-card {sc}">
                <div class="mc-label">System Status</div>
                <div class="mc-value {sv}">{status}</div>
                <div class="mc-sub">cycle {n}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Charts ──
    with chart_ph.container():
        cycles = list(range(1, n + 1))
        true_ruls = [h['true_rul'] for h in history]
        pred_ruls = [h['pred_rul'] for h in history]
        uncertainties = [h['uncertainty'] for h in history]
        sigmas = [h['norm_std'] for h in history]
        rolling_sigmas = [h['rolling_sigma'] for h in history]

        # ══════════════════════════════════════════════
        # CHART 1: RUL Prediction with Individual Models
        # ══════════════════════════════════════════════
        upper = [p + 2 * u for p, u in zip(pred_ruls, uncertainties)]
        lower = [max(0, p - 2 * u) for p, u in zip(pred_ruls, uncertainties)]

        fig1 = go.Figure()

        # Uncertainty band
        fig1.add_trace(go.Scatter(
            x=cycles, y=upper, mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))
        fig1.add_trace(go.Scatter(
            x=cycles, y=lower, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(96,165,250,0.12)',
            name='±2σ Band', hoverinfo='skip'
        ))

        # Individual model predictions (ghost lines)
        model_colors = ['#1e40af', '#1d4ed8', '#2563eb', '#3b82f6', '#60a5fa']
        for m_idx in range(ENSEMBLE_SIZE):
            model_preds = [h['individual_preds'][m_idx] for h in history]
            fig1.add_trace(go.Scatter(
                x=cycles, y=model_preds, mode='lines',
                line=dict(color=model_colors[m_idx], width=0.8),
                opacity=0.25,
                name=f'LSTM {m_idx+1}',
                showlegend=(m_idx == 0),
                legendgroup='individual',
            ))
        # Rename first one for legend
        fig1.data[-ENSEMBLE_SIZE].name = 'Individual LSTMs'

        # Predicted RUL (ensemble mean)
        fig1.add_trace(go.Scatter(
            x=cycles, y=pred_ruls, mode='lines',
            line=dict(color='#60a5fa', width=2.5),
            name='Predicted RUL'
        ))

        # True RUL
        fig1.add_trace(go.Scatter(
            x=cycles, y=true_ruls, mode='lines',
            line=dict(color='#e2e8f0', width=2, dash='dash'),
            name='True RUL'
        ))

        # Comparison: Blind prediction overlay
        if show_comparison and history_blind:
            blind_preds = [h['pred_rul'] for h in history_blind]
            fig1.add_trace(go.Scatter(
                x=list(range(1, len(blind_preds) + 1)), y=blind_preds,
                mode='lines', line=dict(color='#f87171', width=2, dash='dot'),
                name='Blind Agent Pred'
            ))

        # Maintenance marker
        if outcome is not None and outcome != 'crash':
            marker_color = '#a855f7' if outcome == 'safety_override' else '#4ade80'
            fig1.add_trace(go.Scatter(
                x=[n], y=[pred_ruls[-1]], mode='markers',
                marker=dict(size=16, color=marker_color, symbol='star',
                            line=dict(width=2, color='white')),
                name='Maintenance'
            ))

        # Zones
        fig1.add_hrect(y0=0, y1=20, fillcolor="rgba(239,68,68,0.06)", line_width=0)
        fig1.add_hline(y=20, line_dash="dot", line_color="rgba(239,68,68,0.5)", line_width=1.5)
        fig1.add_hline(y=15, line_dash="dash", line_color="rgba(168,85,247,0.5)", line_width=1)
        fig1.add_annotation(xref="paper", x=0.99, y=21, yref="y",
            text="Jackpot Zone (RUL < 20)", showarrow=False,
            xanchor="right", yanchor="bottom",
            font=dict(size=9, color="rgba(248,113,113,0.8)"))
        fig1.add_annotation(xref="paper", x=0.99, y=14, yref="y",
            text="Safety Threshold", showarrow=False,
            xanchor="right", yanchor="top",
            font=dict(size=9, color="rgba(168,85,247,0.8)"))

        fig1.update_layout(
            title=dict(text="Layer 1 — RUL Prediction with Uncertainty Band",
                       x=0.5, xanchor="center", font=dict(size=14, color='#94a3b8')),
            height=380,
            template="plotly_dark",
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22,
                        bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            margin=dict(l=55, r=20, t=50, b=65),
            xaxis=dict(title="Cycle", gridcolor="rgba(255,255,255,0.03)"),
            yaxis=dict(title="RUL (cycles)", gridcolor="rgba(255,255,255,0.03)"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig1, use_container_width=True, key="chart_rul")

        # ══════════════════════════════════════════════
        # CHART 2 & 3: Uncertainty + Q-Values side by side
        # ══════════════════════════════════════════════
        col_c2, col_c3 = st.columns(2)

        with col_c2:
            # Uncertainty chart
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=cycles, y=sigmas, mode='lines',
                line=dict(color='#f59e0b', width=2),
                name='σ (instantaneous)',
                fill='tozeroy', fillcolor='rgba(245,158,11,0.08)'
            ))
            fig2.add_trace(go.Scatter(
                x=cycles, y=rolling_sigmas, mode='lines',
                line=dict(color='#ef4444', width=2, dash='dash'),
                name='σ (rolling avg)'
            ))
            fig2.add_hline(y=0.55, line_dash="dot",
                           line_color="rgba(168,85,247,0.5)", line_width=1)
            fig2.add_annotation(xref="paper", x=0.99, y=0.56, yref="y",
                text="Supervisor σ threshold (0.55)", showarrow=False,
                xanchor="right", yanchor="bottom",
                font=dict(size=8, color="rgba(168,85,247,0.7)"))

            fig2.update_layout(
                title=dict(text="Ensemble Uncertainty (σ)",
                           x=0.5, xanchor="center", font=dict(size=13, color='#94a3b8')),
                height=250,
                template="plotly_dark",
                legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.35,
                            bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                margin=dict(l=50, r=15, t=45, b=55),
                xaxis=dict(title="Cycle", gridcolor="rgba(255,255,255,0.03)"),
                yaxis=dict(title="σ (scaled)", range=[0, 1.05], gridcolor="rgba(255,255,255,0.03)"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig2, use_container_width=True, key="chart_sigma")

        with col_c3:
            # Q-value chart
            q_waits = [h['q_wait'] for h in history]
            q_maintains = [h['q_maintain'] for h in history]
            q_diff = [m - w for w, m in zip(q_waits, q_maintains)]

            fig3 = go.Figure()
            # Color fill: green when positive (prefers maintain), red when negative
            pos_fill = [max(0, d) for d in q_diff]
            neg_fill = [min(0, d) for d in q_diff]

            fig3.add_trace(go.Scatter(
                x=cycles, y=q_diff, mode='lines',
                line=dict(color='#22d3ee', width=2),
                name='Q(MAINTAIN) − Q(WAIT)',
            ))
            fig3.add_trace(go.Scatter(
                x=cycles, y=pos_fill, mode='lines',
                line=dict(width=0), fill='tozeroy',
                fillcolor='rgba(34,197,94,0.12)',
                showlegend=False, hoverinfo='skip'
            ))
            fig3.add_trace(go.Scatter(
                x=cycles, y=neg_fill, mode='lines',
                line=dict(width=0), fill='tozeroy',
                fillcolor='rgba(239,68,68,0.08)',
                showlegend=False, hoverinfo='skip'
            ))
            fig3.add_hline(y=0, line_color="rgba(255,255,255,0.15)")

            fig3.update_layout(
                title=dict(text="Layer 2 — DQN Decision Pressure",
                           x=0.5, xanchor="center", font=dict(size=13, color='#94a3b8')),
                height=250,
                template="plotly_dark",
                legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.35,
                            bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                margin=dict(l=50, r=15, t=45, b=55),
                xaxis=dict(title="Cycle", gridcolor="rgba(255,255,255,0.03)"),
                yaxis=dict(title="Q-Value Diff", gridcolor="rgba(255,255,255,0.03)"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig3, use_container_width=True, key="chart_qvalue")

    # ── Bottom: Event Log + Stats/Outcome ──
    with bottom_ph.container():
        if show_comparison:
            col_left, col_mid, col_right = st.columns([2, 2, 2])
        else:
            col_left, col_right = st.columns([3, 2])
            col_mid = None

        # Event Log
        with col_left:
            st.markdown(f"**Event Log** {'(UA Agent)' if show_comparison else ''}")
            if events:
                log_html = "<div class='event-log'>"
                for etype, msg in reversed(events[-8:]):
                    css = f"ev-{etype}"
                    icon = {'jackpot': '◆', 'safe': '●', 'override': '▲',
                            'crash': '✕', 'early': '◇', 'info': '·'}.get(etype, '·')
                    log_html += f"<span class='{css}'>{icon} {msg}</span><br>"
                log_html += "</div>"
                st.markdown(log_html, unsafe_allow_html=True)
            else:
                st.markdown("<div class='event-log'><span class='ev-info'>Awaiting events...</span></div>",
                           unsafe_allow_html=True)

        # Blind event log (comparison mode)
        if show_comparison and col_mid is not None:
            with col_mid:
                st.markdown("**Event Log** (Blind Agent)")
                if events_blind:
                    log_html = "<div class='event-log'>"
                    for etype, msg in reversed(events_blind[-8:]):
                        css = f"ev-{etype}"
                        icon = {'jackpot': '◆', 'safe': '●', 'override': '▲',
                                'crash': '✕', 'early': '◇'}.get(etype, '·')
                        log_html += f"<span class='{css}'>{icon} {msg}</span><br>"
                    log_html += "</div>"
                    st.markdown(log_html, unsafe_allow_html=True)
                else:
                    st.markdown("<div class='event-log'><span class='ev-info'>Awaiting events...</span></div>",
                               unsafe_allow_html=True)

        # Stats & Outcome
        with col_right:
            # Summary stats
            if n > 1:
                avg_err = np.mean([abs(h['pred_rul'] - h['true_rul']) for h in history])
                avg_sigma = np.mean([h['uncertainty'] for h in history])
                overrides = sum(1 for h in history if h['was_overridden'])
                max_sigma = max(h['uncertainty'] for h in history)

                st.markdown(f"""
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Avg Error</div>
                        <div class="stat-val">{avg_err:.1f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Avg σ</div>
                        <div class="stat-val">{avg_sigma:.1f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Peak σ</div>
                        <div class="stat-val">{max_sigma:.1f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Overrides</div>
                        <div class="stat-val">{overrides}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Outcome banner
            if outcome is not None:
                true_at_end = history[-1]['true_rul']

                if outcome == 'safety_override':
                    ob_class, ob_title_text = "ob-override", "SAFETY OVERRIDE"
                    ob_detail = f"Supervisor intervened at true RUL = {true_at_end:.0f} cycles"
                elif outcome == 'crash':
                    ob_class, ob_title_text = "ob-crash", "ENGINE FAILURE"
                    ob_detail = "No maintenance was triggered in time"
                elif outcome == 'dqn_maintain':
                    if true_at_end <= 20:
                        ob_class, ob_title_text = "ob-jackpot", "JACKPOT"
                        ob_detail = f"Optimal timing! True RUL = {true_at_end:.0f} cycles"
                    elif true_at_end <= 50:
                        ob_class, ob_title_text = "ob-safe", "SAFE REPAIR"
                        ob_detail = f"Acceptable timing at true RUL = {true_at_end:.0f}"
                    else:
                        ob_class, ob_title_text = "ob-early", "EARLY REPAIR"
                        ob_detail = f"Wasteful — true RUL was still {true_at_end:.0f}"
                else:
                    ob_class, ob_title_text = "ob-safe", "COMPLETED"
                    ob_detail = ""

                st.markdown(f"""
                <div class="outcome-banner {ob_class}">
                    <div class="ob-title">{ob_title_text}</div>
                    <div class="ob-detail">{ob_detail}</div>
                </div>
                """, unsafe_allow_html=True)

                # Comparison outcome
                if show_comparison and outcome_blind is not None:
                    blind_true = history_blind[-1]['true_rul'] if history_blind else 0
                    if outcome_blind == 'safety_override':
                        b_class, b_title = "ob-override", "BLIND: OVERRIDE"
                        b_detail = f"Supervisor saved blind agent at RUL = {blind_true:.0f}"
                    elif outcome_blind == 'crash':
                        b_class, b_title = "ob-crash", "BLIND: CRASHED"
                        b_detail = "Blind agent failed to trigger maintenance"
                    else:
                        if blind_true <= 20:
                            b_class, b_title = "ob-jackpot", "BLIND: JACKPOT"
                        elif blind_true <= 50:
                            b_class, b_title = "ob-safe", "BLIND: SAFE"
                        else:
                            b_class, b_title = "ob-early", "BLIND: EARLY"
                        b_detail = f"True RUL = {blind_true:.0f}"

                    st.markdown(f"""
                    <div class="outcome-banner {b_class}" style="margin-top:8px;">
                        <div class="ob-title" style="font-size:1rem;">{b_title}</div>
                        <div class="ob-detail">{b_detail}</div>
                    </div>
                    """, unsafe_allow_html=True)


# =================================================================
# 5. RUN
# =================================================================
if __name__ == "__main__":
    main()
