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
BOOTSTRAP_RATIO = 0.8

# ============================================================
# RL AGENT (DQN)
# ============================================================
RL_TIMESTEPS = 300000
RL_NET_ARCH = [128, 128]
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
NOISE_PROB = 0.5        # 50% of training episodes are noisy
NOISE_LEVEL = 0.15      # Gaussian noise std when active

# Uncertainty scaling: raw ensemble std (~0.01-0.10) is too narrow
# for DQN to learn from. We scale by this factor to fill [0,1].
UNCERTAINTY_SCALE = 5.0

# ============================================================
# SAFETY SUPERVISOR (Layer 3)
# ============================================================
CRITICAL_RUL_NORM = 0.12  # 15 cycles / 125

# ============================================================
# DEVICE
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'