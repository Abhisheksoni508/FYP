"""
=================================================================
  REAL-TIME ENGINE HEALTH MONITORING DASHBOARD
  Uncertainty-Aware RL for Predictive Maintenance
=================================================================

This is the DEMO for your FYP presentation. It creates an interactive
web dashboard that simulates what a real maintenance engineer would
see if your system was deployed.

HOW IT WORKS:
  1. Loads your trained models (LSTM ensemble + DQN agent)
  2. Lets you pick an engine from the C-MAPSS dataset
  3. Steps through the engine's lifecycle cycle-by-cycle
  4. At each cycle, runs the FULL 3-layer pipeline live:
     - Layer 1: LSTM ensemble predicts RUL + uncertainty
     - Layer 2: DQN agent decides WAIT or MAINTAIN
     - Layer 3: Safety supervisor can override
  5. Shows everything in real-time with live charts

WHERE EVERYTHING COMES FROM:
  - Model loading: Same as main_visualize.py (loads from models/)
  - Prediction logic: Same as gym_env.py _get_observation()
  - Q-value extraction: Same as main_visualize.py simulate_engine()
  - Safety override: Imported from src.gym_env.safety_override()
  - Data pipeline: Same as all scripts (load_combined_data → process_data)

USAGE:
  pip install streamlit plotly
  streamlit run dashboard.py

PRESENTATION TIP:
  Open this on a big screen. Pick an engine. Hit "Start".
  Let it run. When it gets to low RUL, inject noise.
  The uncertainty spikes. The safety supervisor intervenes.
  That's your demo in 2 minutes.
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import time
import os

# ── Your project imports ──────────────────────────────────────
# These are the SAME modules your training/evaluation scripts use.
# The dashboard doesn't duplicate any logic — it imports everything.
from src.config import *
from src.preprocessing import load_combined_data, calculate_rul, process_data
from src.lstm_model import RUL_LSTM
from src.gym_env import safety_override


# =================================================================
# 1. MODEL LOADING
# =================================================================
# These functions are identical to what main_visualize.py and
# main_evaluate.py use. We load once and cache with @st.cache_resource
# so models stay in memory between dashboard interactions.

@st.cache_resource
def load_all_models():
    """
    Load LSTM ensemble + DQN agent + preprocessed data.
    
    @st.cache_resource means this only runs ONCE, even if the
    dashboard re-renders. Models stay in memory.
    
    Returns:
        lstm_models: List of 5 LSTM models (same as gym_env._load_ensemble)
        dqn_agent: Trained DQN policy (same as DQN.load in main_evaluate.py)
        blind_agent: Trained Blind DQN policy (or None if not found)
        df_clean: Preprocessed DataFrame (same as all evaluation scripts)
        df_raw: DataFrame with raw RUL values for ground truth
    """
    # ── Load LSTM Ensemble (Layer 1) ──
    # This is the same loop as gym_env.py lines 49-58
    lstm_models = []
    for i in range(ENSEMBLE_SIZE):
        path = f"models/ensemble_model_{i}.pth"
        if os.path.exists(path):
            model = RUL_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()  # Set to inference mode (disables dropout)
            lstm_models.append(model)

    # ── Load DQN Agent (Layer 2) ──
    # Same as main_evaluate.py: DQN.load("models/dqn_pdm_agent")
    from stable_baselines3 import DQN
    dqn_agent = DQN.load("models/dqn_pdm_agent")

    # ── Load Blind Agent (for comparison toggle) ──
    # Created by main_experiment_ablation.py
    blind_agent = None
    if os.path.exists("models/dqn_blind_agent.zip"):
        blind_agent = DQN.load("models/dqn_blind_agent")

    # ── Load and preprocess data ──
    # Same pipeline as every script: load → calculate_rul → process
    df = load_combined_data()
    df = calculate_rul(df)
    df_raw = df.copy()  # Keep raw version for true RUL values
    df_clean, _ = process_data(df, DROP_SENSORS, DROP_SETTINGS)

    return lstm_models, dqn_agent, blind_agent, df_clean, df_raw


def predict_ensemble(models, seq_tensor):
    """
    Run one sequence through all 5 LSTM models.
    
    This is the SAME logic as gym_env.py lines 91-98.
    Each model outputs a RUL prediction (0-1 normalised).
    We take the mean (prediction) and std (uncertainty).
    
    Args:
        models: List of 5 trained LSTM models
        seq_tensor: Shape (1, WINDOW_SIZE, num_features) — one engine's sensor window
        
    Returns:
        mean_pred: Average prediction across models (0-1 normalised)
        std_pred: Standard deviation across models (raw, before scaling)
        all_preds: Individual model predictions (for the fan chart)
    """
    preds = []
    with torch.no_grad():  # No gradient computation needed for inference
        for model in models:
            pred = model(seq_tensor).item()
            preds.append(pred)
    return np.mean(preds), np.std(preds), preds


def get_q_values(agent, obs):
    """
    Extract Q-values from the DQN agent's neural network.
    
    This is the SAME logic as main_visualize.py lines 77-79.
    Q(WAIT) = expected future reward if we wait
    Q(MAINTAIN) = expected future reward if we maintain now
    
    The agent picks whichever action has the higher Q-value.
    
    Args:
        agent: Trained DQN model (stable_baselines3)
        obs: Observation array [mean_rul, sigma, trend]
        
    Returns:
        q_wait: Q-value for action 0 (WAIT)
        q_maintain: Q-value for action 1 (MAINTAIN)
    """
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_values = agent.q_net(obs_tensor).cpu().numpy()[0]
    return float(q_values[0]), float(q_values[1])


# =================================================================
# 2. SIMULATION STEP
# =================================================================

def run_one_cycle(engine_data, engine_raw, cycle_idx, lstm_models, agent,
                  inject_noise=False, noise_level=0.15, use_safety=True,
                  sigma_history=None):
    """
    Run ONE cycle of the full 3-layer pipeline.
    
    This combines logic from:
    - gym_env.py _get_observation() (Lines 76-105): sensor → LSTM → observation
    - main_visualize.py simulate_engine() (Lines 72-97): DQN decision + Q-values
    - gym_env.py safety_override() (Lines 163-175): Layer 3 check
    
    Args:
        engine_data: Preprocessed sensor data for one engine (from df_clean)
        engine_raw: Raw data with true RUL values (from df_raw)
        cycle_idx: Current timestep index (starts at WINDOW_SIZE)
        lstm_models: List of 5 LSTM models
        agent: DQN agent (UA or Blind)
        inject_noise: Whether to corrupt sensor readings
        noise_level: Gaussian noise std (same as NOISE_LEVEL in config.py)
        use_safety: Whether Layer 3 is active
        
    Returns:
        dict with all information for the dashboard to display
    """
    # ── Get sensor features ──
    # Same column filtering as gym_env.py line 82
    features = [c for c in engine_data.columns if c not in ['unit', 'time', 'RUL']]

    # ── Extract the sliding window ──
    # Same as gym_env.py lines 79-83: take last WINDOW_SIZE cycles
    seq = engine_data.iloc[cycle_idx - WINDOW_SIZE:cycle_idx][features].values.copy()

    # ── Optional noise injection ──
    # Same as gym_env.py lines 86-88
    # This is what you toggle in the dashboard to show uncertainty spikes
    if inject_noise:
        noise = np.random.normal(0, noise_level, seq.shape)
        seq = np.clip(seq + noise, 0, 1)

    # ── Layer 1: LSTM Ensemble Prediction ──
    # Same as gym_env.py lines 91-98
    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
    mean_pred, std_pred, individual_preds = predict_ensemble(lstm_models, seq_tensor)

    # ── Build 4D observation vector ──
    # Matches the new gym_env 4D obs: [mean_rul, sigma_now, sigma_rolling_avg, trend]
    norm_mean = np.clip(mean_pred, 0, 1)
    norm_std = np.clip(std_pred * UNCERTAINTY_SCALE, 0, 1)
    sensor_trend = np.clip(np.mean(seq[-1, :]), 0, 1)

    # Option 3: Rolling sigma average from history passed by caller
    if sigma_history is None:
        sigma_history = [0.0, 0.0, 0.0]
    rolling_sigma = float(np.mean(sigma_history))

    obs = np.array([norm_mean, norm_std, rolling_sigma, sensor_trend], dtype=np.float32)

    # ── Layer 2: DQN Decision ──
    # Old blind agent was trained on 3D obs [mean_rul, 0.0, sensor_trend].
    # UA agent uses full 4D [mean_rul, sigma_now, rolling_sigma, sensor_trend].
    # Detect which format the loaded model expects and pass accordingly.
    if agent.observation_space.shape == (3,):
        obs_for_agent = np.array([norm_mean, 0.0, sensor_trend], dtype=np.float32)
    else:
        obs_for_agent = obs

    dqn_action, _ = agent.predict(obs_for_agent, deterministic=True)
    dqn_action = int(dqn_action)

    # ── Extract Q-values (for visualisation) ──
    # Same as main_visualize.py lines 77-79
    q_wait, q_maintain = get_q_values(agent, obs_for_agent)

    # ── Option 1: Uncertainty Gate ──
    # Only fires for UA agent (blind agent always has norm_std=0, so gate never triggers).
    # If DQN says MAINTAIN but sigma is very high and RUL isn't critically low,
    # suppress the decision — "the prediction is too unreliable to act on right now."
    was_uncertainty_gated = False
    if dqn_action == 1 and norm_std > 0.35 and norm_mean > CRITICAL_RUL_NORM:
        dqn_action = 0
        was_uncertainty_gated = True

    # ── Layer 3: Safety Supervisor ──
    # Imported from gym_env.py safety_override()
    final_action = dqn_action
    was_overridden = False
    if use_safety:
        final_action, was_overridden = safety_override(dqn_action, obs)

    # ── Get ground truth ──
    true_rul = engine_raw.iloc[cycle_idx]['RUL']

    return {
        'true_rul': true_rul,
        'pred_rul': max(0, mean_pred * 125.0),          # Convert back to cycles
        'uncertainty': std_pred * 125.0,          # In cycles
        'norm_std': norm_std,                     # Normalised (for display)
        'individual_preds': [max(0, p * 125.0) for p in individual_preds],
        'sensor_trend': sensor_trend,
        'dqn_action': dqn_action,                 # 0=WAIT, 1=MAINTAIN
        'final_action': final_action,             # After safety supervisor
        'was_overridden': was_overridden,
        'was_uncertainty_gated': was_uncertainty_gated,  # Option 1 gate fired
        'q_wait': q_wait,
        'q_maintain': q_maintain,
        'obs': obs,
    }


# =================================================================
# 3. DASHBOARD UI
# =================================================================

def main():
    # ── Page Configuration ──
    st.set_page_config(
        page_title="Engine Health Monitor",
        page_icon="🔧",
        layout="wide"
    )

    # ── Custom CSS for a cleaner, more professional look ──
    st.markdown("""
    <style>
        /* Main title styling */
        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            color: #a0a0de;
            margin-bottom: 0;
        }
        .sub-title {
            font-size: 0.95rem;
            color: #8888aa;
            margin-top: -8px;
            margin-bottom: 20px;
        }

        /* Status cards */
        .status-card {
            padding: 16px 10px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.35);
        }
        .status-safe     { background: linear-gradient(135deg, #d4edda, #b8dfc5); border: 2px solid #28a745; }
        .status-warning  { background: linear-gradient(135deg, #fff3cd, #ffe58a); border: 2px solid #ffc107; }
        .status-danger   { background: linear-gradient(135deg, #f8d7da, #f0b8bc); border: 2px solid #dc3545; }
        .status-override { background: linear-gradient(135deg, #e8daef, #c9a8e0); border: 2px solid #8e44ad; }
        .status-maintained { background: linear-gradient(135deg, #d1ecf1, #a8d8e4); border: 2px solid #17a2b8; }

        .status-label { font-size: 0.75rem; color: #222; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.6px; }
        .status-value { font-size: 1.9rem; color: #111; font-weight: 800; margin: 6px 0; }
        .status-sub   { font-size: 0.72rem; color: #444; }

        /* Event log */
        .event-log {
            background: #0d0d1e;
            color: #00e676;
            font-family: 'Courier New', monospace;
            padding: 14px 16px;
            border-radius: 10px;
            border: 1px solid #1e1e3a;
            max-height: 210px;
            overflow-y: auto;
            font-size: 0.78rem;
            line-height: 1.7;
        }

        /* Layer indicator badges */
        .layer-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 0.75rem;
            font-weight: 700;
            margin: 2px 4px;
            letter-spacing: 0.3px;
        }
        .layer-1 { background: #2980b9; color: white; }
        .layer-2 { background: #27ae60; color: white; }
        .layer-3 { background: #c0392b; color: white; }
    </style>
    """, unsafe_allow_html=True)

    # ── Title ──
    st.markdown('<p class="main-title">🔧 Engine Health Monitoring Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Uncertainty-Aware Reinforcement Learning for Predictive Maintenance</p>', unsafe_allow_html=True)

    # ── Layer Architecture Indicator ──
    st.markdown("""
    <div style="margin-bottom: 15px;">
        <span class="layer-badge layer-1">Layer 1: LSTM Ensemble</span>
        <span class="layer-badge layer-2">Layer 2: DQN Agent</span>
        <span class="layer-badge layer-3">Layer 3: Safety Supervisor</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Models (cached — only runs once) ──
    with st.spinner("Loading trained models..."):
        lstm_models, dqn_agent, blind_agent, df_clean, df_raw = load_all_models()

    # =============================================================
    # SIDEBAR — Controls
    # =============================================================
    st.sidebar.header("🎛️ Simulation Controls")

    # Engine selection
    # Get all unique engines and their lifecycle lengths
    all_units = sorted(df_clean['unit'].unique())
    unit_lengths = {u: len(df_clean[df_clean['unit'] == u]) for u in all_units}

    # Show some useful engine options
    st.sidebar.markdown("**Select Engine**")
    engine_id = st.sidebar.selectbox(
        "Engine Unit ID",
        all_units,
        index=all_units.index(134) if 134 in all_units else 0,
        format_func=lambda u: f"Engine {u}  ({unit_lengths[u]} cycles, {'FD001' if u <= 100 else 'FD002'})"
    )

    st.sidebar.divider()

    # ── Layer Toggles ──
    st.sidebar.markdown("**Layer Controls**")

    # Agent selection (UA vs Blind)
    # This is the key comparison from Experiment 1
    agent_type = st.sidebar.radio(
        "Layer 2: Agent Type",
        ["Uncertainty-Aware (UA)", "Blind (No σ)"],
        help="UA agent uses uncertainty (σ) in its decisions. Blind agent ignores it."
    )

    # Safety supervisor toggle
    # This is what Experiment 2 tests
    use_safety = st.sidebar.checkbox(
        "Layer 3: Safety Supervisor",
        value=True,
        help="When ON, forces maintenance if predicted RUL < 15 cycles AND ensemble is confident (rolling σ < 0.25). UA agent: supervisor ignores noisy false alarms. Blind agent: rolling σ is always 0, so supervisor fires on every low prediction."
    )

    st.sidebar.divider()

    # ── Noise Controls ──
    st.sidebar.markdown("**Sensor Noise Injection**")
    inject_noise = st.sidebar.checkbox(
        "🔊 Inject Sensor Noise",
        value=False,
        help="Simulates sensor degradation. Watch uncertainty spike!"
    )
    noise_level = st.sidebar.slider(
        "Noise Level (σ)",
        0.01, 0.10, 0.05, 0.01,
        disabled=not inject_noise,
        help="Low (0.03–0.06): UA advantage visible. High (0.09–0.10): both agents struggle."
    )

    st.sidebar.divider()

    # ── Speed Control ──
    speed = st.sidebar.slider(
        "⏩ Simulation Speed",
        0.05, 1.0, 0.3, 0.05,
        help="Seconds per cycle. Lower = faster."
    )

    # ── Select the correct agent ──
    if agent_type == "Blind (No σ)" and blind_agent is not None:
        active_agent = blind_agent
    else:
        active_agent = dqn_agent
        if agent_type == "Blind (No σ)" and blind_agent is None:
            st.sidebar.warning("Blind agent not found. Run main_experiment_ablation.py first.")

    # =============================================================
    # SESSION STATE — Simulation memory
    # =============================================================
    # Streamlit re-runs the entire script on every interaction.
    # st.session_state persists data between re-runs.

    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'events' not in st.session_state:
        st.session_state.events = []
    if 'outcome' not in st.session_state:
        st.session_state.outcome = None
    if 'current_engine' not in st.session_state:
        st.session_state.current_engine = None
    if 'sigma_history' not in st.session_state:
        st.session_state.sigma_history = [0.0, 0.0, 0.0]   # Option 3: rolling sigma

    # Reset if engine changed
    if st.session_state.current_engine != engine_id:
        st.session_state.history = []
        st.session_state.events = []
        st.session_state.outcome = None
        st.session_state.running = False
        st.session_state.current_engine = engine_id
        st.session_state.sigma_history = [0.0, 0.0, 0.0]

    # ── Control Buttons ──
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        start = st.button("▶️ Start Simulation", use_container_width=True, type="primary")
    with col_btn2:
        stop = st.button("⏹️ Stop", use_container_width=True)
    with col_btn3:
        reset = st.button("🔄 Reset", use_container_width=True)

    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False
    if reset:
        st.session_state.history = []
        st.session_state.events = []
        st.session_state.outcome = None
        st.session_state.running = False
        st.session_state.sigma_history = [0.0, 0.0, 0.0]
        st.rerun()

    # =============================================================
    # MAIN DISPLAY — Placeholders for live updates
    # =============================================================
    # We create empty containers that get updated each cycle.
    # This is how Streamlit does "real-time" updates.

    # Row 1: Status cards
    metric_placeholder = st.empty()

    # Row 2: Charts
    chart_placeholder = st.empty()

    # Row 3: Q-values + Event log
    bottom_placeholder = st.empty()

    # =============================================================
    # SIMULATION LOOP
    # =============================================================

    # Get engine data
    engine_data = df_clean[df_clean['unit'] == engine_id].reset_index(drop=True)
    engine_raw = df_raw[df_raw['unit'] == engine_id].reset_index(drop=True)
    max_cycles = len(engine_data)

    rendered_in_loop = False
    if st.session_state.running and st.session_state.outcome is None:
        # Start from where we left off (or from WINDOW_SIZE)
        start_idx = WINDOW_SIZE + len(st.session_state.history)

        for cycle_idx in range(start_idx, max_cycles):
            if not st.session_state.running:
                break

            # ── Run one cycle through the full 3-layer pipeline ──
            result = run_one_cycle(
                engine_data, engine_raw, cycle_idx,
                lstm_models, active_agent,
                inject_noise=inject_noise,
                noise_level=noise_level,
                use_safety=use_safety,
                sigma_history=list(st.session_state.sigma_history)
            )
            # Update sigma history for next cycle (Option 3)
            st.session_state.sigma_history.pop(0)
            st.session_state.sigma_history.append(result['norm_std'])
            st.session_state.history.append(result)

            # ── Log events ──
            cycle_num = cycle_idx - WINDOW_SIZE + 1
            true_rul = result['true_rul']

            # Log non-terminating uncertainty gate event
            if result.get('was_uncertainty_gated'):
                st.session_state.events.append(
                    f"🛡️ Cycle {cycle_num}: UNCERTAINTY GATE — Suppressed premature MAINTAIN "
                    f"(σ={result['norm_std']:.2f}, pred={result['pred_rul']:.0f})"
                )

            if result['was_overridden']:
                st.session_state.events.append(
                    f"⚠️ Cycle {cycle_num}: SAFETY OVERRIDE — Forced maintenance "
                    f"(pred RUL={result['pred_rul']:.0f}, true RUL={true_rul:.0f})"
                )
                st.session_state.outcome = 'safety_override'
                st.session_state.running = False
            elif result['final_action'] == 1:
                # DQN chose to maintain on its own
                if true_rul <= 20:
                    label = "🎯 JACKPOT"
                elif true_rul <= 50:
                    label = "✅ SAFE"
                else:
                    label = "⚡ EARLY (wasteful)"
                st.session_state.events.append(
                    f"{label} — Cycle {cycle_num}: DQN triggered maintenance "
                    f"(pred={result['pred_rul']:.0f}, true={true_rul:.0f})"
                )
                st.session_state.outcome = 'dqn_maintain'
                st.session_state.running = False
            elif cycle_idx >= max_cycles - 1:
                st.session_state.events.append(
                    f"💥 Cycle {cycle_num}: ENGINE FAILURE — No maintenance triggered!"
                )
                st.session_state.outcome = 'crash'
                st.session_state.running = False

            # ── Update the dashboard display ──
            render_dashboard(
                st.session_state.history,
                st.session_state.events,
                st.session_state.outcome,
                metric_placeholder,
                chart_placeholder,
                bottom_placeholder,
                agent_type,
                use_safety,
                inject_noise
            )
            rendered_in_loop = True

            if not st.session_state.running:
                break

            time.sleep(speed)

    # Render current state (for when simulation is paused/stopped but not just run)
    if st.session_state.history and not rendered_in_loop:
        render_dashboard(
            st.session_state.history,
            st.session_state.events,
            st.session_state.outcome,
            metric_placeholder,
            chart_placeholder,
            bottom_placeholder,
            agent_type,
            use_safety,
            inject_noise
        )
    elif not st.session_state.history:
        # Only show the "not started" message when no simulation has run yet
        metric_placeholder.info(
            f"🔧 Engine {engine_id} loaded ({max_cycles} cycles). "
            f"Press **Start Simulation** to begin."
        )


# =================================================================
# 4. RENDERING FUNCTIONS
# =================================================================

def render_dashboard(history, events, outcome,
                     metric_ph, chart_ph, bottom_ph,
                     agent_type, use_safety, inject_noise):
    """
    Render the full dashboard display.
    Called every cycle during simulation.
    """
    latest = history[-1]
    n = len(history)

    # ── ROW 1: Status Cards ──
    with metric_ph.container():
        c1, c2, c3, c4, c5 = st.columns(5)

        # Card 1: Predicted RUL
        pred_rul = latest['pred_rul']
        if pred_rul > 50:
            card_class = "status-safe"
        elif pred_rul > 20:
            card_class = "status-warning"
        else:
            card_class = "status-danger"

        c1.markdown(f"""
        <div class="status-card {card_class}">
            <div class="status-label">Predicted RUL</div>
            <div class="status-value">{pred_rul:.0f}</div>
            <div class="status-sub">cycles remaining</div>
        </div>
        """, unsafe_allow_html=True)

        # Card 2: True RUL (ground truth)
        true_rul = latest['true_rul']
        if true_rul > 50:
            card_class = "status-safe"
        elif true_rul > 20:
            card_class = "status-warning"
        else:
            card_class = "status-danger"

        c2.markdown(f"""
        <div class="status-card {card_class}">
            <div class="status-label">True RUL</div>
            <div class="status-value">{true_rul:.0f}</div>
            <div class="status-sub">actual cycles left</div>
        </div>
        """, unsafe_allow_html=True)

        # Card 3: Uncertainty
        unc = latest['uncertainty']
        if unc < 3:
            card_class = "status-safe"
        elif unc < 8:
            card_class = "status-warning"
        else:
            card_class = "status-danger"

        c3.markdown(f"""
        <div class="status-card {card_class}">
            <div class="status-label">Uncertainty (σ)</div>
            <div class="status-value">{unc:.1f}</div>
            <div class="status-sub">ensemble disagreement</div>
        </div>
        """, unsafe_allow_html=True)

        # Card 4: DQN Decision
        action_text = "MAINTAIN" if latest['dqn_action'] == 1 else "WAIT"
        action_class = "status-warning" if latest['dqn_action'] == 1 else "status-safe"
        c4.markdown(f"""
        <div class="status-card {action_class}">
            <div class="status-label">DQN Decision</div>
            <div class="status-value">{action_text}</div>
            <div class="status-sub">{"UA Agent" if "UA" in agent_type else "Blind Agent"}</div>
        </div>
        """, unsafe_allow_html=True)

        # Card 5: System Status
        if outcome == 'safety_override':
            card_class = "status-override"
            status = "OVERRIDE"
            sub = "Safety supervisor intervened"
        elif outcome == 'dqn_maintain':
            card_class = "status-maintained"
            status = "MAINTAINED"
            sub = f"True RUL was {latest['true_rul']:.0f}"
        elif outcome == 'crash':
            card_class = "status-danger"
            status = "CRASHED"
            sub = "Engine failed!"
        elif latest['was_overridden']:
            card_class = "status-override"
            status = "OVERRIDE"
            sub = "Layer 3 activated"
        else:
            card_class = "status-safe"
            status = "RUNNING"
            sub = f"Cycle {n}/{n + WINDOW_SIZE}"

        c5.markdown(f"""
        <div class="status-card {card_class}">
            <div class="status-label">System Status</div>
            <div class="status-value">{status}</div>
            <div class="status-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── ROW 2: Live Charts ──
    with chart_ph.container():
        cycles        = list(range(1, n + 1))
        true_ruls     = [h['true_rul']    for h in history]
        pred_ruls     = [h['pred_rul']    for h in history]
        uncertainties = [h['uncertainty'] for h in history]

        # ── Chart 1: RUL Prediction ──
        upper = [p + 2 * u for p, u in zip(pred_ruls, uncertainties)]
        lower = [max(0, p - 2 * u) for p, u in zip(pred_ruls, uncertainties)]

        fig1 = go.Figure()

        # Uncertainty band (±2σ) — fill between upper and lower
        fig1.add_trace(go.Scatter(
            x=cycles, y=upper, mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))
        fig1.add_trace(go.Scatter(
            x=cycles, y=lower, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(52,152,219,0.18)',
            name='±2σ Uncertainty', hoverinfo='skip'
        ))

        # Predicted RUL
        fig1.add_trace(go.Scatter(
            x=cycles, y=pred_ruls, mode='lines',
            line=dict(color='#5dade2', width=2.5),
            name='Predicted RUL (Layer 1)'
        ))

        # True RUL
        fig1.add_trace(go.Scatter(
            x=cycles, y=true_ruls, mode='lines',
            line=dict(color='#ecf0f1', width=2, dash='dash'),
            name='True RUL (Ground Truth)'
        ))

        # Maintenance trigger marker
        if outcome is not None and outcome != 'crash':
            marker_color = '#9b59b6' if outcome == 'safety_override' else '#2ecc71'
            fig1.add_trace(go.Scatter(
                x=[n], y=[pred_ruls[-1]], mode='markers',
                marker=dict(size=14, color=marker_color, symbol='star',
                            line=dict(width=2, color='white')),
                name='Maintenance Triggered'
            ))

        # Reference zones (no built-in annotation — use add_annotation to avoid overlap)
        fig1.add_hrect(y0=0, y1=20, fillcolor="rgba(231,76,60,0.07)", line_width=0)
        fig1.add_hline(y=20, line_dash="dot",  line_color="rgba(231,76,60,0.75)",  line_width=1.5)
        fig1.add_hline(y=15, line_dash="dash", line_color="rgba(155,89,182,0.75)", line_width=1.5)

        # Staggered annotations so they don't collide
        fig1.add_annotation(
            xref="paper", x=0.99, y=21, yref="y",
            text="Jackpot Zone (RUL < 20)", showarrow=False,
            xanchor="right", yanchor="bottom",
            font=dict(size=10, color="rgba(231,76,60,0.9)")
        )
        fig1.add_annotation(
            xref="paper", x=0.99, y=14, yref="y",
            text="Safety Threshold (15)", showarrow=False,
            xanchor="right", yanchor="top",
            font=dict(size=10, color="rgba(155,89,182,0.9)")
        )

        fig1.update_layout(
            title=dict(text="RUL Prediction with Uncertainty Band",
                       x=0.5, xanchor="center", font=dict(size=15)),
            height=430,
            template="plotly_dark",
            legend=dict(
                orientation="h", x=0.5, xanchor="center",
                y=-0.18, yanchor="top",
                bgcolor="rgba(0,0,0,0)", font=dict(size=12)
            ),
            margin=dict(l=60, r=20, t=55, b=70),
            xaxis=dict(title="Cycle", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="RUL (cycles)", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # ── Chart 2: Q-Values ──
        q_waits    = [h['q_wait']    for h in history]
        q_maintains = [h['q_maintain'] for h in history]
        q_diff = [m - w for w, m in zip(q_waits, q_maintains)]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=cycles, y=q_diff, mode='lines',
            line=dict(color='#e74c3c', width=2),
            name='Q(MAINTAIN) − Q(WAIT)',
            fill='tozeroy', fillcolor='rgba(231,76,60,0.12)'
        ))
        fig2.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_dash="solid")

        fig2.update_layout(
            title=dict(text="DQN Q-Values — Agent Reasoning  (positive = prefers MAINTAIN)",
                       x=0.5, xanchor="center", font=dict(size=13)),
            height=230,
            template="plotly_dark",
            legend=dict(
                orientation="h", x=0.5, xanchor="center",
                y=-0.35, yanchor="top",
                bgcolor="rgba(0,0,0,0)", font=dict(size=12)
            ),
            margin=dict(l=60, r=20, t=45, b=60),
            xaxis=dict(title="Cycle", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Q-Value Diff", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── ROW 3: Event Log + Info ──
    with bottom_ph.container():
        col_log, col_info = st.columns([3, 2])

        with col_log:
            st.markdown("**📋 Event Log**")
            if events:
                log_html = "<div class='event-log'>"
                for e in reversed(events[-10:]):  # Show last 10 events
                    log_html += f"{e}<br>"
                log_html += "</div>"
                st.markdown(log_html, unsafe_allow_html=True)
            else:
                st.markdown("<div class='event-log'>Waiting for events...</div>",
                           unsafe_allow_html=True)

        with col_info:
            st.markdown("**Active Configuration**")
            st.markdown(f"""
            | Setting | Value |
            |---------|-------|
            | Agent | {'**UA** (uses σ)' if 'UA' in agent_type else '**Blind** (ignores σ)'} |
            | Safety Supervisor | {'✅ ON' if use_safety else '❌ OFF'} |
            | Noise Injection | {'🔊 ON (σ={uncertainties[-1] if uncertainties else 0:.1f})' if inject_noise else '🔇 OFF'} |
            | Cycles Elapsed | {n} |
            | Prediction Error | {abs(pred_ruls[-1] - true_ruls[-1]):.1f} cycles |
            """)

            # Show outcome summary if simulation ended
            if outcome == 'safety_override':
                st.success(f"🛡️ **Safety Supervisor saved the engine!** "
                          f"Intervened at true RUL = {latest['true_rul']:.0f} cycles.")
            elif outcome == 'dqn_maintain':
                if latest['true_rul'] <= 20:
                    st.success(f"🎯 **JACKPOT!** DQN triggered at true RUL = {latest['true_rul']:.0f}. "
                              f"Optimal timing!")
                elif latest['true_rul'] <= 50:
                    st.info(f"✅ **Safe repair** at true RUL = {latest['true_rul']:.0f}. "
                           f"Acceptable timing.")
                else:
                    st.warning(f"⚡ **Early repair** at true RUL = {latest['true_rul']:.0f}. "
                              f"Wasteful but safe.")
            elif outcome == 'crash':
                st.error("💥 **ENGINE FAILURE!** No maintenance was triggered in time.")


# =================================================================
# 5. RUN
# =================================================================
if __name__ == "__main__":
    main()
