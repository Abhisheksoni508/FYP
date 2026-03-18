"""
Dissertation-Quality Visualization of the 3-Layer Hybrid AI System.

Generates a 3-panel figure:
  Panel 1 (Top):    RUL Prediction with uncertainty bands and danger zones
  Panel 2 (Middle): Uncertainty signal showing ensemble disagreement over time
  Panel 3 (Bottom): Decision timeline showing all 3 layers working together

Usage: python main_visualize.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import DQN

from src.preprocessing import load_combined_data, calculate_rul, process_data
from src.lstm_model import RUL_LSTM
from src.gym_env import safety_override
from src.config import *


def load_ensemble(models_dir):
    models = []
    for i in range(ENSEMBLE_SIZE):
        path = f"{models_dir}/ensemble_model_{i}.pth"
        m = RUL_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        models.append(m)
    return models


def predict_ensemble(models, seq):
    seq_tensor = torch.FloatTensor(seq).to(DEVICE)
    preds = []
    with torch.no_grad():
        for m in models:
            preds.append(m(seq_tensor).item())
    return np.mean(preds), np.std(preds)


def simulate_engine(unit_id, df, df_clean, lstm_models, agent):
    """Simulate the full lifecycle of one engine."""
    unit_df = df[df['unit'] == unit_id].reset_index(drop=True)
    unit_clean = df_clean[df['unit'] == unit_id].reset_index(drop=True)
    
    if len(unit_clean) <= WINDOW_SIZE + 10:
        return None
    
    true_ruls, pred_ruls, uncertainties = [], [], []
    norm_stds = []
    dqn_actions, safety_flags, final_actions = [], [], []
    q_maintain_history = []  # Q-value for MAINTAIN action over time
    q_wait_history = []      # Q-value for WAIT action over time
    repaired = False
    trigger_type = None
    trigger_true_rul = None
    trigger_idx = None
    sigma_history = [0.0, 0.0, 0.0]
    
    for t in range(WINDOW_SIZE, len(unit_clean)):
        features = [c for c in unit_clean.columns if c not in ['unit', 'time', 'RUL']]
        seq = unit_clean.iloc[t - WINDOW_SIZE:t][features].values
        seq = seq.reshape(1, WINDOW_SIZE, -1)
        
        mean_pred, std_pred = predict_ensemble(lstm_models, seq)
        norm_std = np.clip(std_pred * UNCERTAINTY_SCALE, 0, 1)
        sigma_history.pop(0)
        sigma_history.append(float(norm_std))
        rolling_sigma = float(np.mean(sigma_history))
        sensor_trend = float(np.mean(seq[0, -1, :]))
        obs = np.array([mean_pred, norm_std, rolling_sigma, sensor_trend], dtype=np.float32)
        
        # Layer 2: DQN decision + Q-value extraction
        dqn_action = 0
        if not repaired and agent:
            dqn_action, _ = agent.predict(obs, deterministic=True)
            dqn_action = int(dqn_action)
            # Extract Q-values to show agent's "thinking"
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_values = agent.q_net(obs_tensor).cpu().numpy()[0]
            q_wait_history.append(float(q_values[0]))
            q_maintain_history.append(float(q_values[1]))
        else:
            q_wait_history.append(0.0)
            q_maintain_history.append(0.0)
        
        # Layer 3: Safety supervisor
        safety_flag = False
        final_action = dqn_action
        if not repaired:
            final_action, safety_flag = safety_override(dqn_action, obs)
        
        if not repaired and final_action == 1:
            repaired = True
            trigger_true_rul = unit_df.iloc[t]['RUL']
            trigger_type = 'safety' if safety_flag else 'dqn'
            trigger_idx = len(true_ruls)
        
        true_ruls.append(unit_df.iloc[t]['RUL'])
        pred_ruls.append(mean_pred * 125.0)
        uncertainties.append(std_pred * 125.0)
        norm_stds.append(norm_std)
        
        # Record what happened at this timestep
        if final_action == 1 and trigger_idx == len(true_ruls) - 1:
            dqn_actions.append(dqn_action)
            safety_flags.append(safety_flag)
            final_actions.append(1)
        else:
            dqn_actions.append(0)
            safety_flags.append(False)
            final_actions.append(0)
    
    return {
        'unit_id': unit_id,
        'time_steps': list(range(WINDOW_SIZE, len(unit_clean))),
        'true_ruls': true_ruls,
        'pred_ruls': pred_ruls,
        'uncertainties': uncertainties,
        'norm_stds': norm_stds,
        'dqn_actions': dqn_actions,
        'safety_flags': safety_flags,
        'final_actions': final_actions,
        'trigger_type': trigger_type,
        'trigger_true_rul': trigger_true_rul,
        'trigger_idx': trigger_idx,
        'q_wait': q_wait_history,
        'q_maintain': q_maintain_history,
    }


def create_visualization(data):
    """Create the 3-panel dissertation figure."""
    
    ts = data['time_steps']
    true_ruls = data['true_ruls']
    pred_ruls = data['pred_ruls']
    uncerts = data['uncertainties']
    norm_stds = data['norm_stds']
    final_acts = data['final_actions']
    safety = data['safety_flags']
    uid = data['unit_id']
    trigger_idx = data['trigger_idx']
    
    # Colours
    C_PRED = '#1a5276'
    C_TRUE = '#555555'
    C_UNCERT = '#3498db'
    C_JACKPOT = '#27ae60'
    C_SAFE = '#f39c12'
    C_TRIGGER = '#e74c3c'
    C_SAFETY = '#8e44ad'
    
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.5], hspace=0.12)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # =================================================================
    # PANEL 1: RUL Prediction with Danger Zones
    # =================================================================
    
    # Zone shading
    ax1.axhspan(0, 20, alpha=0.08, color=C_JACKPOT, zorder=0)
    ax1.axhspan(20, 50, alpha=0.05, color=C_SAFE, zorder=0)
    
    # Zone labels
    ax1.text(max(ts) + 2, 10, 'JACKPOT\nZONE', fontsize=8, color=C_JACKPOT, 
             fontweight='bold', va='center', ha='left', clip_on=False)
    ax1.text(max(ts) + 2, 35, 'SAFE\nZONE', fontsize=8, color=C_SAFE, 
             fontweight='bold', va='center', ha='left', clip_on=False)
    
    # True RUL
    ax1.plot(ts, true_ruls, color=C_TRUE, linestyle='--', linewidth=1.5, 
             alpha=0.5, label='True RUL (ground truth)')
    
    # LSTM Prediction + uncertainty
    ax1.plot(ts, pred_ruls, color=C_PRED, linewidth=2.2, label='LSTM Ensemble Prediction')
    lower = np.array(pred_ruls) - np.array(uncerts)
    upper = np.array(pred_ruls) + np.array(uncerts)
    ax1.fill_between(ts, lower, upper, color=C_UNCERT, alpha=0.18, label='Uncertainty (±σ)')
    
    # Threshold lines
    ax1.axhline(y=20, color=C_JACKPOT, linestyle=':', linewidth=1, alpha=0.5)
    ax1.axhline(y=50, color=C_SAFE, linestyle=':', linewidth=1, alpha=0.3)
    
    # Maintenance trigger
    if trigger_idx is not None:
        t_time = ts[trigger_idx]
        t_pred = pred_ruls[trigger_idx]
        t_true = true_ruls[trigger_idx]
        
        marker_color = C_SAFETY if data['trigger_type'] == 'safety' else C_TRIGGER
        ax1.scatter(t_time, t_pred, c=marker_color, s=300, marker='X', zorder=5,
                   edgecolors='black', linewidth=1)
        
        if data['trigger_type'] == 'safety':
            label_text = f'Layer 3: Safety Supervisor\noverrides DQN → MAINTAIN\nTrue RUL = {t_true:.0f} cycles'
        else:
            label_text = f'Layer 2: DQN Agent\ntriggers maintenance\nTrue RUL = {t_true:.0f} cycles'
        
        jackpot = "✓ JACKPOT (+500)" if t_true < 20 else "~ SAFE (+10)"
        label_text += f'\n{jackpot}'
        
        ax1.annotate(
            label_text,
            xy=(t_time, t_pred), xytext=(-160, 60),
            textcoords='offset points', fontsize=9, fontweight='bold',
            color=marker_color,
            arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor=marker_color, alpha=0.9)
        )
    
    ax1.set_ylabel('Remaining Useful Life (Cycles)', fontsize=11)
    ax1.set_title(f'Engine {uid}: 3-Layer Hybrid AI Decision System', 
                  fontsize=14, fontweight='bold', pad=12)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.set_ylim(bottom=-5, top=max(true_ruls) * 1.08)
    ax1.grid(True, alpha=0.2)
    ax1.text(0.01, 0.97, 'Layer 1: LSTM Deep Ensemble (×5)', transform=ax1.transAxes,
             fontsize=9, fontweight='bold', va='top', color=C_PRED,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # =================================================================
    # PANEL 2: Uncertainty Signal
    # =================================================================
    
    ax2.fill_between(ts, 0, norm_stds, color=C_UNCERT, alpha=0.3)
    ax2.plot(ts, norm_stds, color=C_PRED, linewidth=1.5)
    
    ax2.axhline(y=0.3, color='orange', linestyle='--', linewidth=1, alpha=0.6)
    ax2.text(min(ts) + 3, 0.32, 'High uncertainty threshold', fontsize=8, 
             color='orange', alpha=0.8)
    
    if trigger_idx is not None:
        ax2.axvline(x=ts[trigger_idx], color=C_TRIGGER, linestyle='-', 
                    linewidth=1.5, alpha=0.5)
    
    ax2.set_ylabel('Uncertainty\n(scaled σ)', fontsize=10)
    ax2.set_ylim(0, max(max(norm_stds) * 1.3, 0.5))
    ax2.grid(True, alpha=0.2)
    ax2.text(0.01, 0.88, 'Ensemble Disagreement → DQN Input', transform=ax2.transAxes,
             fontsize=9, fontweight='bold', va='top', color=C_PRED,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # =================================================================
    # PANEL 3: DQN Decision Process (Q-Values)
    # =================================================================
    
    q_wait = data['q_wait']
    q_maintain = data['q_maintain']
    
    # Only plot Q-values up to the trigger point (after that they're zeroed)
    if trigger_idx is not None:
        plot_end = trigger_idx + 1
    else:
        plot_end = len(ts)
    
    ts_q = ts[:plot_end]
    q_diff = [q_maintain[i] - q_wait[i] for i in range(plot_end)]
    
    # Plot Q(MAINTAIN) - Q(WAIT): positive = agent wants to maintain
    ax3.fill_between(ts_q, 0, q_diff, 
                     where=[d > 0 for d in q_diff],
                     color=C_TRIGGER, alpha=0.3, label='Favours MAINTAIN')
    ax3.fill_between(ts_q, 0, q_diff, 
                     where=[d <= 0 for d in q_diff],
                     color=C_PRED, alpha=0.2, label='Favours WAIT')
    ax3.plot(ts_q, q_diff, color='#2c3e50', linewidth=1.5)
    
    # Zero line = decision boundary
    ax3.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.5)
    ax3.text(min(ts) + 3, 0.5, 'Decision boundary: above = MAINTAIN, below = WAIT', 
             fontsize=8, color='#666666', style='italic')
    
    # Mark trigger point
    if trigger_idx is not None:
        marker_color = C_SAFETY if data['trigger_type'] == 'safety' else C_TRIGGER
        trigger_label = 'Safety Override' if data['trigger_type'] == 'safety' else 'DQN Triggers'
        ax3.axvline(x=ts[trigger_idx], color=marker_color, linewidth=2, 
                   linestyle='--', alpha=0.7)
        ax3.annotate(trigger_label, xy=(ts[trigger_idx], max(q_diff[-5:])),
                    xytext=(-60, 15), textcoords='offset points',
                    fontsize=8, fontweight='bold', color=marker_color,
                    arrowprops=dict(arrowstyle='->', color=marker_color, lw=1))
    
    ax3.set_ylabel('Q(Maintain) − Q(Wait)', fontsize=10)
    ax3.set_xlabel('Time (Cycles)', fontsize=11)
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax3.text(0.01, 0.92, 'Layer 2: DQN Agent Decision Pressure', 
             transform=ax3.transAxes, fontsize=9, fontweight='bold', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =================================================================
    # Summary footer
    # =================================================================
    if data['trigger_true_rul'] is None:
        result_label = f'Crash ({CRASH_PENALTY})'
        trigger_label = 'No maintenance trigger'
        trigger_rul = 'N/A'
    else:
        result_label = 'Jackpot (+500+)' if data['trigger_true_rul'] < 20 else 'Safe (+10+)'
        trigger_label = 'Safety Supervisor' if data['trigger_type'] == 'safety' else 'DQN Agent'
        trigger_rul = f'{data["trigger_true_rul"]:.0f} cycles'
    
    summary = (f"Architecture: LSTM Ensemble (×5) → DQN Agent → Safety Supervisor   |   "
               f"Triggered by: {trigger_label}   |   "
               f"True RUL at trigger: {trigger_rul}   |   "
               f"Outcome: {result_label}")
    
    fig.text(0.5, 0.008, summary, ha='center', fontsize=9, style='italic', color='#444444')
    
    save_path = f"final_engine_{uid}_timeline.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  3-LAYER HYBRID AI SYSTEM — ENGINE LIFECYCLE VISUALIZATION")
    print("=" * 60)
    
    print("\n--- Loading Data ---")
    df = load_combined_data()
    df = calculate_rul(df)
    df_clean, _ = process_data(df, DROP_SENSORS, DROP_SETTINGS)
    
    print("--- Loading Models ---")
    lstm_models = load_ensemble('models')
    agent = DQN.load("models/dqn_pdm_agent")
    
    # Three engines across different operating regimes:
    #   50  = FD001 (single operating condition)
    #   134 = FD001 (long lifecycle, good tracking)
    #   200 = FD002 (six operating conditions — harder environment)
    engines = [50, 134, 200]
    
    for uid in engines:
        print(f"\n--- Simulating Engine {uid} ---")
        data = simulate_engine(uid, df, df_clean, lstm_models, agent)
        
        if data:
            print(f"    Trigger type: {data['trigger_type']}")
            print(f"    True RUL at trigger: {data['trigger_true_rul']:.0f} cycles")
            print(f"--- Generating Figure ---")
            create_visualization(data)
        else:
            print(f"    Engine {uid} not found or too short, skipping.")
