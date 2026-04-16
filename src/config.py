import os
import warnings

# Suppress noisy warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'           # Hide TensorFlow/oneDNN messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'           # Disable oneDNN custom operations
warnings.filterwarnings('ignore', category=FutureWarning)  # Hide PyTorch weights_only warning

import torch

# ============================================================
# DATA
# ============================================================
WINDOW_SIZE = 30
BATCH_SIZE = 64
TRAIN_SPLIT = 0.8

# FD002 needs all sensors and settings (operating conditions vary)
DROP_SENSORS = []
DROP_SETTINGS = []

# ============================================================
# LSTM ENSEMBLE
# ============================================================
INPUT_DIM = 24          # 21 sensors + 3 settings
HIDDEN_DIM = 100
NUM_LAYERS = 2
DROPOUT = 0.2
ENSEMBLE_SIZE = 5

EPOCHS = 80
LEARNING_RATE = 0.0005
PATIENCE = 15
GRAD_CLIP = 1.0
# Historical name kept for compatibility: this is a per-member random subset
# fraction drawn without replacement, not classical bootstrap resampling.
BOOTSTRAP_RATIO = 0.8

# ============================================================
# RL AGENT (DQN)
# ============================================================
RL_TIMESTEPS = 1_500_000  # 2× longer training — UA needs more steps to learn sigma-conditional policy
RL_NET_ARCH = [256, 256, 128]   # Deeper network — 3 layers for better sigma-conditional learning
RL_LR = 0.0003
RL_BATCH = 128
RL_BUFFER = 150000
RL_LEARNING_STARTS = 10000
RL_EXPLORE_FRAC = 0.4
RL_EXPLORE_FINAL = 0.03
RL_TARGET_UPDATE = 1000
RL_TRAIN_FREQ = 4
RL_GAMMA = 0.99

# ============================================================
# NOISE AUGMENTATION
# ============================================================
# During training, this fraction of episodes have sensor noise injected.
# This teaches the UA agent what high uncertainty MEANS.
NOISE_PROB = 0.7        # 70% of training episodes are noisy (was 0.5)
NOISE_LEVEL = 0.20      # Max Gaussian noise std (raised from 0.15 so σ=0.15 eval is mid-range)
NOISE_LEVEL_MIN = 0.02  # Min Gaussian noise std — sampled per episode during training

# Uncertainty scaling: raw ensemble std (~0.01-0.10) is too narrow
# for DQN to learn from. We scale by this factor to fill [0,1].
# Scale 15 gives stronger gradation: clean→0.15-0.3, medium→0.6, high→0.9+
UNCERTAINTY_SCALE = 15.0

# ============================================================
# SAFETY SUPERVISOR (Layer 3)
# ============================================================
CRITICAL_RUL_NORM = 0.12      # 15 cycles / 125

# Hard-critical fallback: at extreme noise the rolling sigma can exceed the
# confidence threshold, silencing the supervisor entirely. This secondary
# threshold fires regardless of uncertainty — it's the final safety net.
# At RUL < 5 cycles there is no justification for waiting, even if noisy.
HARD_CRITICAL_RUL_NORM = 0.04   # ~5 cycles / 125 — always fire regardless of sigma

# Supervisor only fires when rolling sigma is BELOW this threshold.
# This is what makes the supervisor uncertainty-aware:
#   - Blind agent: rolling sigma is always 0.0 → always below threshold
#     → supervisor fires on every low prediction (same as before, many false alarms)
#   - UA agent: rolling sigma reflects real ensemble disagreement
#     → supervisor only fires when predictions are trustworthy (low uncertainty)
#     → ignores noisy dips, fires only on genuinely confident low predictions
# Adjusted for UNCERTAINTY_SCALE=15: raw_std ~0.037 → scaled 0.55
# Below 0.55 = supervisor fires (confident predictions)
# Above 0.55 = supervisor backs off (uncertain — trust the UA agent)
SUPERVISOR_SIGMA_THRESHOLD = 0.55

# ============================================================
# RL TRAINING — END-OF-LIFE BIAS
# ============================================================
# The DQN was learning "always WAIT" because only 13% of random
# episode starts land in the critical end-of-life zone.
# Fix: force 50% of episodes to start in the last EOL_WINDOW cycles
# so the agent sees many more maintenance timing decisions.
EOL_EPISODE_PROB = 0.6   # fraction of episodes that start near end of life (raised from 0.5)
EOL_WINDOW       = 50    # how many cycles from the end counts as "end of life" (raised from 40)

# Crash penalty raised so it dominates even with heavy discounting.
# Old: -100. At gamma=0.99 and 160 steps out → present value = -20 (trivial).
# New: -500. Present value at 160 steps → -100 (agent actually cares).
CRASH_PENALTY = -500

# Time-pressure shaping: WAIT reward decays linearly once true RUL drops
# below this threshold. This forces the DQN to MAINTAIN near end-of-life
# even when sigma is persistently high (noisy sensors), because waiting
# becomes increasingly costly rather than always giving +1.
#
# True RUL > 30:  reward = +1   (normal, wait freely)
# True RUL = 20:  reward =  0   (neutral, should be thinking about acting)
# True RUL = 10:  reward = -1   (costly to keep waiting)
# True RUL =  0:  reward = -2   (very costly — then crash fires at -500)
TIME_PRESSURE_START = 25   # cycles from end where WAIT reward starts decaying

# ============================================================
# DEVICE
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
