"""Figure 2: Training dynamics (LSTM loss + DQN reward) — representative slices."""
import os, sys, numpy as np, torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = r"c:\Users\Abhishek Soni\OneDrive\Desktop\FINAL_FYP"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.preprocessing import load_combined_data, calculate_rul, process_data, create_sequences, CMAPSSDataset
from src.lstm_model import RUL_LSTM
from src.config import *

OUT = os.path.join(ROOT, "figures", "training_curves.png")

# ---------- LSTM: short reconstructive training of 5 members ----------
print("Preparing LSTM data...")
df = load_combined_data()
df = calculate_rul(df)
df_proc, scaler = process_data(df, DROP_SENSORS, DROP_SETTINGS)
features = [c for c in df_proc.columns if c not in ['unit', 'time', 'RUL']]
seqs, labels = create_sequences(df_proc, WINDOW_SIZE, features)
print(f"sequences: {seqs.shape}")

# Subsample for speed
rng = np.random.default_rng(42)
idx = rng.choice(len(seqs), size=min(15000, len(seqs)), replace=False)
seqs_s = seqs[idx]
labels_s = labels[idx]

dataset = CMAPSSDataset(seqs_s, labels_s)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

lstm_epochs = 8
lstm_losses = np.zeros((ENSEMBLE_SIZE, lstm_epochs))

for mi in range(ENSEMBLE_SIZE):
    torch.manual_seed(42 + mi*17)
    np.random.seed(42 + mi*17)
    print(f"Training member {mi+1}/{ENSEMBLE_SIZE}...")
    model = RUL_LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit = nn.MSELoss()
    for ep in range(lstm_epochs):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            p = model(xb)
            loss = crit(p, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        lstm_losses[mi, ep] = running / n
        print(f"  m{mi} ep{ep+1} loss={lstm_losses[mi, ep]:.5f}")
    del model
    torch.cuda.empty_cache() if DEVICE == 'cuda' else None

np.save(os.path.join(ROOT, "scratch", "lstm_losses.npy"), lstm_losses)

# ---------- DQN: short training slice ----------
print("Preparing DQN environment for training slice...")
from src.gym_env import PdMEnvironment
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

env = PdMEnvironment(df_proc, models_dir=os.path.join(ROOT, "models"),
                     noise_prob=NOISE_PROB, noise_level=NOISE_LEVEL, variable_noise=True)
env = Monitor(env)

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__(0)
        self.steps = []
        self.rewards = []
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0 and self.num_timesteps % 500 == 0:
            recent = list(self.model.ep_info_buffer)[-20:]
            mean_r = np.mean([ep['r'] for ep in recent])
            self.steps.append(self.num_timesteps)
            self.rewards.append(mean_r)
        return True

model = DQN("MlpPolicy", env,
            policy_kwargs=dict(net_arch=RL_NET_ARCH),
            learning_rate=RL_LR, batch_size=RL_BATCH, buffer_size=RL_BUFFER,
            learning_starts=2000, exploration_fraction=RL_EXPLORE_FRAC,
            exploration_final_eps=RL_EXPLORE_FINAL,
            target_update_interval=RL_TARGET_UPDATE,
            train_freq=RL_TRAIN_FREQ, gamma=RL_GAMMA, verbose=0)

cb = RewardLogger()
print("Training DQN slice (50000 steps)...")
model.learn(total_timesteps=50000, callback=cb, progress_bar=False)

dqn_steps = np.array(cb.steps)
dqn_rewards = np.array(cb.rewards)
np.save(os.path.join(ROOT, "scratch", "dqn_steps.npy"), dqn_steps)
np.save(os.path.join(ROOT, "scratch", "dqn_rewards.npy"), dqn_rewards)

# Do NOT save model

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

ax = axes[0]
if len(dqn_steps) > 0:
    # Smooth with rolling mean
    if len(dqn_rewards) >= 5:
        sm = np.convolve(dqn_rewards, np.ones(5)/5, mode='valid')
        sm_x = dqn_steps[2:2+len(sm)]
        ax.plot(dqn_steps, dqn_rewards, color='#a0aec0', lw=1, alpha=0.6, label='Raw mean reward')
        ax.plot(sm_x, sm, color='#2b6cb0', lw=2.2, label='Smoothed (window=5)')
    else:
        ax.plot(dqn_steps, dqn_rewards, color='#2b6cb0', lw=2)
ax.axhline(0, color='black', lw=0.6, alpha=0.5)
ax.set_xlabel('Environment timesteps', fontsize=11)
ax.set_ylabel('Mean episode reward (last 20 episodes)', fontsize=11)
ax.set_title('DQN Training Reward\n(representative 50k slice)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.25)

ax = axes[1]
colors = plt.cm.viridis(np.linspace(0.1, 0.85, ENSEMBLE_SIZE))
for mi in range(ENSEMBLE_SIZE):
    ax.plot(np.arange(1, lstm_epochs+1), lstm_losses[mi], 'o-',
            color=colors[mi], lw=1.8, ms=5, alpha=0.9, label=f'Member {mi+1}')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Training MSE loss', fontsize=11)
ax.set_title('LSTM Ensemble Training Loss\n(representative trace, 5 members)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right', ncol=2)
ax.grid(True, alpha=0.25)

plt.suptitle('Training Dynamics — Representative Slices (original long runs not logged)',
             fontsize=11, y=1.03, style='italic')
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"Saved {OUT}")
