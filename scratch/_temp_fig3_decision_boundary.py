"""Figure 3: Decision boundary heatmap — UA vs Blind DQN over (RUL, sigma)."""
import os, sys, numpy as np, torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = r"c:\Users\Abhishek Soni\OneDrive\Desktop\FINAL_FYP"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from stable_baselines3 import DQN
from src.config import *

OUT = os.path.join(ROOT, "figures", "decision_boundary.png")

print("Loading agents...")
ua = DQN.load(os.path.join(ROOT, "models", "dqn_pdm_agent.zip"), device='cpu')
try:
    blind = DQN.load(os.path.join(ROOT, "models", "dqn_blind_agent.zip"), device='cpu')
    has_blind = True
except Exception as e:
    print(f"No blind agent: {e}")
    blind = None
    has_blind = False

N = 60
rul_grid = np.linspace(0.001, 1.0, N)
sig_grid = np.linspace(0.0, 1.0, N)
R, S = np.meshgrid(rul_grid, sig_grid)

SENSOR_TREND_FIX = 0.5  # median-ish value

def query(agent, use_blind_obs=False):
    actions = np.zeros_like(R, dtype=np.int32)
    qdiff = np.zeros_like(R, dtype=np.float32)
    obs_all = np.zeros((R.size, 4), dtype=np.float32)
    obs_all[:, 0] = R.ravel()
    if use_blind_obs:
        obs_all[:, 1] = 0.0
        obs_all[:, 2] = 0.0
    else:
        obs_all[:, 1] = S.ravel()
        obs_all[:, 2] = S.ravel()   # rolling_sigma = sigma_now
    obs_all[:, 3] = SENSOR_TREND_FIX

    # predict in batches
    import torch as T
    with T.no_grad():
        t = T.from_numpy(obs_all)
        q = agent.q_net(t).cpu().numpy()  # (N*N, 2)
    acts = np.argmax(q, axis=1).reshape(R.shape)
    qd = (q[:, 1] - q[:, 0]).reshape(R.shape)
    return acts, qd

print("Querying UA agent...")
ua_act, ua_q = query(ua, use_blind_obs=False)

if has_blind:
    print("Querying Blind agent...")
    blind_act, blind_q = query(blind, use_blind_obs=True)

# Plot
ncols = 2 if has_blind else 1
fig, axes = plt.subplots(1, ncols, figsize=(10 if has_blind else 6, 4.6), squeeze=False)
axes = axes[0]

def plot_panel(ax, acts, qd, title):
    # Use diverging Q-diff colormap
    vmax = max(abs(qd.min()), abs(qd.max()))
    im = ax.pcolormesh(R * 125.0, S, qd, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    # Contour of decision boundary (acts==1)
    ax.contour(R * 125.0, S, acts, levels=[0.5], colors='black', linewidths=2.0)
    # Supervisor boundaries (normalized)
    ax.axvline(HARD_CRITICAL_RUL_NORM * 125.0, color='#e53e3e', linestyle='--', lw=1.4,
               label=f'Tier 1 ({HARD_CRITICAL_RUL_NORM*125:.0f} cyc)')
    # Tier 2 L-shape: fires for rul<critical AND sigma<threshold
    ax.plot([CRITICAL_RUL_NORM * 125.0, CRITICAL_RUL_NORM * 125.0],
            [0, SUPERVISOR_SIGMA_THRESHOLD], color='#dd6b20', linestyle='--', lw=1.4)
    ax.plot([0, CRITICAL_RUL_NORM * 125.0],
            [SUPERVISOR_SIGMA_THRESHOLD, SUPERVISOR_SIGMA_THRESHOLD],
            color='#dd6b20', linestyle='--', lw=1.4, label=f'Tier 2 gate')
    ax.set_xlabel('Predicted RUL (cycles)', fontsize=11)
    ax.set_ylabel('Ensemble uncertainty σ (normalised)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label('Q(MAINTAIN) − Q(WAIT)', fontsize=9)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    # Text annotations
    ax.text(0.98, 0.02, 'Black line = decision boundary', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

plot_panel(axes[0], ua_act, ua_q, 'UA Agent: σ-conditional policy')
if has_blind:
    plot_panel(axes[1], blind_act, blind_q, 'Blind Agent: σ ignored')

plt.suptitle('DQN Policy Heatmap over (RUL, σ)  —  MAINTAIN vs WAIT decision surface',
             fontsize=12, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"Saved {OUT}")

# Report key stats
ua_maintain_frac = ua_act.mean()
print(f"UA MAINTAIN fraction over grid: {ua_maintain_frac:.3f}")
if has_blind:
    print(f"Blind MAINTAIN fraction over grid: {blind_act.mean():.3f}")
# Check whether UA boundary bends with sigma: compare RUL threshold at sigma=0 vs sigma=0.8
def rul_boundary_at_sigma(acts, rul_grid, sig_grid, sig_val):
    si = np.argmin(np.abs(sig_grid - sig_val))
    row = acts[si]
    # find smallest RUL index where action becomes 0 (WAIT) scanning from left
    # The boundary is where action switches from 1 (MAINTAIN) to 0 (WAIT)
    idxs = np.where(row == 1)[0]
    if len(idxs) == 0:
        return None
    return rul_grid[idxs.max()] * 125.0

b_lo = rul_boundary_at_sigma(ua_act, rul_grid, sig_grid, 0.05)
b_hi = rul_boundary_at_sigma(ua_act, rul_grid, sig_grid, 0.85)
print(f"UA MAINTAIN max RUL at σ=0.05: {b_lo}")
print(f"UA MAINTAIN max RUL at σ=0.85: {b_hi}")
