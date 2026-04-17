# Full Project Pipeline — End-to-End Walkthrough

## The Dataset — What You're Actually Working With

Imagine you have **360 jet engines** in a factory. Each engine runs until it fails. While it's running, **21 sensors** are recording stuff every cycle (temperature, pressure, fan speed, etc.) plus **3 operational settings** (altitude, throttle, Mach number).

**Raw data looks like this** (one row = one cycle of one engine):

```
unit | time | setting1 | setting2 | setting3 | s1 | s2 | s3 | ... | s21
  1  |   1  |  0.0023  |  0.0003  |   100    | 518| 642| 1589| ... | 23.4
  1  |   2  |  0.0015  |  0.0002  |   100    | 519| 641| 1591| ... | 23.3
  ...
  1  | 192  |  0.0021  |  0.0004  |   100    | 555| 620| 1540| ... | 22.1  ← engine 1 dies here
  2  |   1  |  ...     |  ...     |   ...    | ...                        ← engine 2 starts
```

FD001 has 100 engines (simple — 1 operating condition). FD002 has 260 engines (harder — 6 operating conditions). You merge them → **360 engines total**. FD002 unit IDs get shifted by +100 so engine 1 from FD002 becomes engine 101 (otherwise they'd clash).

---

## Step 1: Preprocessing (`preprocessing.py`)

**Calculate RUL:**
Engine 1 dies at cycle 192. So at cycle 1, its RUL = 191. At cycle 190, RUL = 2. At cycle 192, RUL = 0 (dead). But RUL is **capped at 125** because when an engine has 200 cycles left vs 300 cycles left, the difference doesn't matter — it's healthy either way. Only the last ~125 cycles matter.

**Normalize everything:**
All 24 features (21 sensors + 3 settings) get squished to [0, 1] using MinMaxScaler. RUL also gets normalized: `RUL / 125` → so 0.0 = dead, 1.0 = fully healthy.

**Create sliding windows:**
The LSTM can't look at one cycle in isolation — it needs context. So you slide a window of **30 consecutive cycles** across each engine's life:

```
Engine 1:  [cycle 1-30] → RUL at cycle 31
           [cycle 2-31] → RUL at cycle 32
           [cycle 3-32] → RUL at cycle 33
           ...
           [cycle 161-190] → RUL at cycle 191
```

Each window = a **30 × 24 matrix** (30 timesteps, 24 features). The label = the normalized RUL at the cycle right after the window ends. Across the merged FD001+FD002 dataset used in this project, this gives **63,590 windows**.

---

## Step 2: Train the LSTM Ensemble (`main_train_ensemble.py`)

You train **5 separate LSTM models**, not one. Each model:

1. Gets a **different random 80%** of the training data (random subset sampling without replacement)
2. Has a **different random seed** (42, 59, 76, 93, 110)
3. Learns slightly **different patterns** because of this

**What the LSTM does internally:**

```
Input: [30 × 24] matrix (30 cycles of sensor readings)
  ↓
LSTM Layer 1 (100 hidden units) — learns temporal patterns
  ↓
LSTM Layer 2 (100 hidden units) — learns higher-level degradation patterns
  ↓
Take ONLY the last timestep's hidden state (the summary of all 30 cycles)
  ↓
Linear layer: 100 → 1
  ↓
Output: single number ∈ [0, 1] = predicted normalized RUL
```

So after training, you have 5 models that all predict RUL, but they'll **disagree with each other** — and that disagreement IS your uncertainty signal.

---

## Step 3: Train the DQN Agent (`main_train_rl.py`)

This is where it gets interesting. The DQN doesn't see raw sensor data. It sees a **4-number summary** produced by the LSTM ensemble.

**Every cycle during training, here's what happens:**

### 3a. The environment picks an engine and a starting point

```python
# 60% of the time → start in last 50 cycles (where decisions matter)
# 40% of the time → random start (so it also learns "just wait" for healthy engines)
```

### 3b. At each cycle, build the observation

The environment grabs the last 30 cycles of sensor data, and **maybe adds noise**:

```
Raw sensor window [30 × 24]
        ↓
70% chance: add Gaussian noise (σ randomly between 0.02 and 0.20)
        ↓
Feed into all 5 LSTM models
        ↓
Model 1 predicts: 0.35
Model 2 predicts: 0.32
Model 3 predicts: 0.38    ← clean data: models mostly agree
Model 4 predicts: 0.34
Model 5 predicts: 0.36

Mean (μ) = 0.35      ← "the engine has ~44 cycles left"
Std  (σ) = 0.02      ← "we're pretty confident"
σ_scaled = 0.02 × 15 = 0.30  ← scaled to be meaningful in [0,1]
```

With noise injection, the same models might say:
```
Model 1 predicts: 0.41
Model 2 predicts: 0.18
Model 3 predicts: 0.52    ← noisy data: models DISAGREE wildly
Model 4 predicts: 0.29
Model 5 predicts: 0.45

Mean (μ) = 0.37
Std  (σ) = 0.13      ← high disagreement!
σ_scaled = 0.13 × 15 = 1.0 (clipped)  ← "we have NO idea"
```

The **4D observation** sent to the DQN:
```
obs = [
    0.35,    # mean_rul       → "how much life is left?"
    0.30,    # sigma_now      → "how much do models disagree RIGHT NOW?"
    0.25,    # sigma_rolling   → "average disagreement over last 3 cycles" (smoother)
    0.48     # sensor_health  → aggregate current sensor-health cue
]
```

### 3c. DQN makes a decision

The DQN is a neural network with layers [256, 256, 128]. It takes the 4D observation and outputs **two Q-values**:

```
Q(WAIT)     = 45.2    ← "expected future reward if I wait"
Q(MAINTAIN) = 12.8    ← "expected future reward if I maintain now"

45.2 > 12.8 → action = WAIT
```

### 3d. Environment gives a reward

If the DQN chose **WAIT**:
- Engine is still alive? Reward = `1.0 - time_pressure - uncertainty_urgency`
  - Far from failure → reward ≈ +1.0 (good, keep waiting)
  - Close to failure with high σ → reward goes negative (bad, you're gambling!)
- Engine just died? Reward = **-500** (catastrophic, game over)

If the DQN chose **MAINTAIN**:
- RUL was < 20 cycles? Reward = **+500** (JACKPOT — perfect timing!)
- RUL was < 50 cycles? Reward = **+10** (acceptable, a bit early)
- RUL was ≥ 50 cycles? Reward = **-20** (wasteful, too early)
- Episode ends.

### 3e. Why noise augmentation matters

Without noise: σ is always low during training → the DQN never learns what high σ means → at deployment, if sensors get noisy, the agent is confused.

With noise: the DQN **experiences** episodes where σ spikes → learns "high σ near failure = MAINTAIN NOW, don't gamble" → robust at deployment.

The **blind agent** is identical but obs[1] and obs[2] are **always set to 0**. It literally can't see uncertainty. So it can't learn to react to it.

---

## Step 4: At Deployment — The Full 3-Layer Pipeline

An engine is running. Every cycle:

```
┌──────────────────────────────────────────────────┐
│ SENSOR DATA: last 30 cycles × 24 features        │
└──────────────────┬───────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────────┐
│ LAYER 1: Feed into 5 LSTM models                 │
│                                                  │
│   Model 1 → 0.12                                 │
│   Model 2 → 0.10     μ = 0.11 (~14 cycles left) │
│   Model 3 → 0.13     σ = 0.015 (confident)      │
│   Model 4 → 0.09                                 │
│   Model 5 → 0.12                                 │
└──────────────────┬───────────────────────────────┘
                   ↓
         obs = [0.11, 0.22, 0.18, 0.43]
                   ↓
┌──────────────────────────────────────────────────┐
│ LAYER 2: DQN looks at obs                        │
│                                                  │
│   Q(WAIT) = -12.5                                │
│   Q(MAINTAIN) = 485.3                            │
│                                                  │
│   → action = MAINTAIN ✓                          │
└──────────────────┬───────────────────────────────┘
                   ↓
         DQN said MAINTAIN → no override needed
                   ↓
┌──────────────────────────────────────────────────┐
│ LAYER 3: Safety supervisor checks                │
│                                                  │
│   DQN said MAINTAIN → supervisor does nothing    │
│   (supervisor only overrides WAIT decisions)     │
│                                                  │
│   → Final action: MAINTAIN                       │
└──────────────────────────────────────────────────┘
                   ↓
            🎯 JACKPOT — RUL was 14 cycles!
```

But what if the DQN messes up?

```
         DQN says WAIT (oops)
                   ↓
┌──────────────────────────────────────────────────┐
│ LAYER 3: Safety supervisor checks                │
│                                                  │
│   obs[0] = 0.11 (< 0.12 = CRITICAL threshold)   │
│   obs[2] = 0.18 (< 0.55 = confident)            │
│                                                  │
│   "RUL is critically low AND I'm confident"      │
│   → OVERRIDE! Force MAINTAIN                     │
└──────────────────────────────────────────────────┘
                   ↓
            ⚠️ SAFETY OVERRIDE — engine saved
```

---

## Step 5: The Dashboard (`dashboard.py`)

The Streamlit dashboard just runs this exact pipeline **visually, one cycle at a time**:

1. You pick an engine from the dropdown (say engine 134)
2. You toggle UA agent vs Blind agent
3. You optionally inject noise with a slider
4. You hit "Run" and watch cycle by cycle:
   - The RUL prediction line drops as the engine degrades
   - The uncertainty band widens if you injected noise
   - The Q-value chart shows the DQN's "thinking" — when Q(MAINTAIN) overtakes Q(WAIT)
   - A star appears on the chart where maintenance was triggered
   - The event log shows "JACKPOT" or "SAFETY OVERRIDE" or "CRASH"

It's the same pipeline as above, just with Plotly charts updating in real-time so you can **see** all three layers working.

---

**TL;DR the whole flow:**

```
Raw sensor CSVs
  → normalize + sliding windows of 30 cycles
    → 5 LSTM models each predict RUL → get mean (μ) and disagreement (σ)
      → pack into 4D vector [μ, σ_now, σ_rolling, trend]
        → DQN reads this → outputs WAIT or MAINTAIN
          → Safety supervisor checks → maybe overrides WAIT to MAINTAIN
            → Dashboard shows all of this live
```
