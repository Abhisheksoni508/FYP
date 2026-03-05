"""
Cost-Effectiveness Analysis — Real-World Financial Impact of Uncertainty-Aware
Predictive Maintenance vs Blind Approach.

SCENARIO:
  A mid-size UK airline operates a fleet of 50 turbofan engines (e.g., Airbus
  A320neo fleet). Each engine undergoes ~4 maintenance decision cycles per year,
  giving 200 maintenance decisions annually. We simulate these decisions under
  varying sensor degradation (noise) to compute annual costs for both strategies.

COST MODEL (GBP, based on industry data):
  ┌─────────────────────┬───────────┬─────────────────────────────────────────┐
  │ Outcome             │ Cost (£)  │ Rationale                               │
  ├─────────────────────┼───────────┼─────────────────────────────────────────┤
  │ Optimal (Jackpot)   │  £18,000  │ Planned maintenance, parts pre-ordered, │
  │                     │           │ minimal hangar time (2-3 days)           │
  │ Acceptable (Safe)   │  £35,000  │ Slightly early — some useful life lost,  │
  │                     │           │ unneeded parts replaced (5-7 days)       │
  │ Wasteful (Too Early)│  £52,000  │ Significant life discarded, full shop    │
  │                     │           │ visit triggered unnecessarily (7+ days)  │
  │ Unplanned Failure   │ £275,000  │ AOG event, emergency repair, flight      │
  │ (Crash)             │           │ cancellations, passenger compensation,   │
  │                     │           │ potential CAA safety review (14-21 days)  │
  └─────────────────────┴───────────┴─────────────────────────────────────────┘

  Additional: Each day of unplanned downtime costs ~£38,000 in lost revenue
  (source: IATA airline cost benchmarks). This is folded into the crash cost.

OUTPUTS:
  - report_cost_analysis.png    (4-panel publication figure)
  - report_cost_summary.csv     (detailed cost breakdown)

Usage:  python main_cost_analysis.py
Prereq: models/dqn_pdm_agent.zip, models/dqn_blind_agent.zip
Time:   ~15-20 mins
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from src.preprocessing import load_combined_data, calculate_rul, process_data
from src.gym_env import PdMEnvironment, BlindPdMEnvironment, safety_override
from src.config import *

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ============================================================
#  COST MODEL (GBP)
# ============================================================
COST_JACKPOT  =  18_000   # Optimal planned maintenance
COST_SAFE     =  35_000   # Acceptable but early
COST_WASTEFUL =  52_000   # Unnecessarily early, wasted engine life
COST_CRASH    = 275_000   # Unplanned failure — AOG, emergency, compensation

# Fleet scenario
FLEET_SIZE         = 50    # Number of turbofan engines
DECISIONS_PER_ENGINE = 4   # Maintenance decision cycles per engine per year
ANNUAL_DECISIONS   = FLEET_SIZE * DECISIONS_PER_ENGINE  # 200/year

# Evaluation config
NUM_EPISODES = 500         # Per noise level per agent
NOISE_LEVELS = [0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175]

# Plot style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
UA_COLOR     = '#2e86de'
BLIND_COLOR  = '#ee5253'
ACCENT_GREEN = '#10ac84'
ACCENT_AMBER = '#f39c12'


# ============================================================
#  EVALUATION WITH OUTCOME TRACKING
# ============================================================

def evaluate_outcomes(env, model, num_episodes, use_safety=True):
    """
    Run episodes and return per-episode outcome categories + costs.
    Returns dict with counts, cost arrays, and summary stats.
    """
    outcomes = []  # list of 'jackpot', 'safe', 'wasteful', 'crash'
    costs = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if use_safety:
                action, _ = safety_override(action, obs)
            obs, reward, done, _, _ = env.step(action)
            if done:
                if reward <= -100:
                    outcomes.append('crash')
                    costs.append(COST_CRASH)
                elif reward == 500:
                    outcomes.append('jackpot')
                    costs.append(COST_JACKPOT)
                elif reward == 10:
                    outcomes.append('safe')
                    costs.append(COST_SAFE)
                elif reward == -20:
                    outcomes.append('wasteful')
                    costs.append(COST_WASTEFUL)
                else:
                    # Edge case — treat as safe
                    outcomes.append('safe')
                    costs.append(COST_SAFE)

    n = len(outcomes)
    costs = np.array(costs)
    return {
        'n': n,
        'outcomes': outcomes,
        'costs': costs,
        'mean_cost': np.mean(costs),
        'median_cost': np.median(costs),
        'total_cost': np.sum(costs),
        'std_cost': np.std(costs),
        'ci_95': 1.96 * np.std(costs) / np.sqrt(n),
        'n_jackpot': outcomes.count('jackpot'),
        'n_safe': outcomes.count('safe'),
        'n_wasteful': outcomes.count('wasteful'),
        'n_crash': outcomes.count('crash'),
        'jackpot_rate': outcomes.count('jackpot') / n * 100,
        'crash_rate': outcomes.count('crash') / n * 100,
    }


def annual_fleet_cost(result):
    """
    Scale per-decision average cost to annual fleet cost.
    """
    return result['mean_cost'] * ANNUAL_DECISIONS


def annual_fleet_cost_breakdown(result):
    """
    Break down annual fleet cost by outcome type.
    """
    n = result['n']
    return {
        'jackpot': (result['n_jackpot'] / n) * ANNUAL_DECISIONS * COST_JACKPOT,
        'safe':    (result['n_safe'] / n) * ANNUAL_DECISIONS * COST_SAFE,
        'wasteful':(result['n_wasteful'] / n) * ANNUAL_DECISIONS * COST_WASTEFUL,
        'crash':   (result['n_crash'] / n) * ANNUAL_DECISIONS * COST_CRASH,
    }


# ============================================================
#  PLOTTING
# ============================================================

def format_gbp(val, pos=None):
    """Format large GBP values for axis ticks."""
    if abs(val) >= 1_000_000:
        return f'£{val/1_000_000:.1f}M'
    elif abs(val) >= 1_000:
        return f'£{val/1_000:.0f}k'
    else:
        return f'£{val:.0f}'


def plot_cost_analysis(noise_levels, ua_results, bl_results,
                       ua_safety_results, bl_safety_results):
    """
    Create 4-panel cost analysis figure.

    (a) Cost per maintenance decision — no supervisor
    (b) Cost per maintenance decision — with supervisor (full system)
    (c) Annual fleet savings (UA vs Blind)
    (d) Cost breakdown at representative noise level
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    nl = noise_levels

    # ─── (a) Cost per decision — Layer 2 only ────────────────
    ax = axes[0, 0]
    ua_mean = [r['mean_cost'] for r in ua_results]
    bl_mean = [r['mean_cost'] for r in bl_results]
    ua_ci   = [r['ci_95'] for r in ua_results]
    bl_ci   = [r['ci_95'] for r in bl_results]

    ax.plot(nl, ua_mean, '-o', color=UA_COLOR, lw=2, ms=7, label='UA Agent')
    ax.fill_between(nl, [m-c for m,c in zip(ua_mean, ua_ci)],
                        [m+c for m,c in zip(ua_mean, ua_ci)], alpha=0.15, color=UA_COLOR)
    ax.plot(nl, bl_mean, '--s', color=BLIND_COLOR, lw=2, ms=7, label='Blind Agent')
    ax.fill_between(nl, [m-c for m,c in zip(bl_mean, bl_ci)],
                        [m+c for m,c in zip(bl_mean, bl_ci)], alpha=0.15, color=BLIND_COLOR)
    ax.fill_between(nl, ua_mean, bl_mean,
                    where=[b > u for u, b in zip(ua_mean, bl_mean)],
                    alpha=0.10, color=ACCENT_GREEN, label='UA saves more')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_gbp))
    ax.set_xlabel('Sensor Noise Level (σ)')
    ax.set_ylabel('Average Cost per Decision')
    ax.set_title('(a)  Maintenance Cost — RL Agent Only (No Supervisor)')
    ax.legend(loc='upper left', framealpha=0.9)

    # ─── (b) Cost per decision — Full system ─────────────────
    ax = axes[0, 1]
    ua_s_mean = [r['mean_cost'] for r in ua_safety_results]
    bl_s_mean = [r['mean_cost'] for r in bl_safety_results]
    ua_s_ci   = [r['ci_95'] for r in ua_safety_results]
    bl_s_ci   = [r['ci_95'] for r in bl_safety_results]

    ax.plot(nl, ua_s_mean, '-o', color=UA_COLOR, lw=2, ms=7, label='UA + Supervisor')
    ax.fill_between(nl, [m-c for m,c in zip(ua_s_mean, ua_s_ci)],
                        [m+c for m,c in zip(ua_s_mean, ua_s_ci)], alpha=0.15, color=UA_COLOR)
    ax.plot(nl, bl_s_mean, '--s', color=BLIND_COLOR, lw=2, ms=7, label='Blind + Supervisor')
    ax.fill_between(nl, [m-c for m,c in zip(bl_s_mean, bl_s_ci)],
                        [m+c for m,c in zip(bl_s_mean, bl_s_ci)], alpha=0.15, color=BLIND_COLOR)
    ax.fill_between(nl, ua_s_mean, bl_s_mean,
                    where=[b > u for u, b in zip(ua_s_mean, bl_s_mean)],
                    alpha=0.10, color=ACCENT_GREEN, label='UA saves more')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_gbp))
    ax.set_xlabel('Sensor Noise Level (σ)')
    ax.set_ylabel('Average Cost per Decision')
    ax.set_title('(b)  Maintenance Cost — Full 3-Layer System')
    ax.legend(loc='upper left', framealpha=0.9)

    # ─── (c) Annual Fleet Savings ─────────────────────────────
    ax = axes[1, 0]
    ua_annual_raw  = [annual_fleet_cost(r) for r in ua_results]
    bl_annual_raw  = [annual_fleet_cost(r) for r in bl_results]
    ua_annual_safe = [annual_fleet_cost(r) for r in ua_safety_results]
    bl_annual_safe = [annual_fleet_cost(r) for r in bl_safety_results]

    savings_raw  = [(b - u) for u, b in zip(ua_annual_raw, bl_annual_raw)]
    savings_safe = [(b - u) for u, b in zip(ua_annual_safe, bl_annual_safe)]

    bw = (nl[1] - nl[0]) * 0.35
    bars1 = ax.bar([x - bw/2 for x in nl], [s/1000 for s in savings_raw], bw,
                   color=UA_COLOR, alpha=0.80, edgecolor='white', lw=0.6,
                   label='Without Supervisor')
    bars2 = ax.bar([x + bw/2 for x in nl], [s/1000 for s in savings_safe], bw,
                   color=ACCENT_GREEN, alpha=0.80, edgecolor='white', lw=0.6,
                   label='With Supervisor')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 10:
                ax.text(bar.get_x() + bar.get_width()/2, h + (8 if h >= 0 else -18),
                        f'£{h:.0f}k', ha='center', va='bottom' if h >= 0 else 'top',
                        fontsize=7, fontweight='bold')

    ax.axhline(y=0, color='grey', lw=0.8, ls='-')
    ax.set_xlabel('Sensor Noise Level (σ)')
    ax.set_ylabel('Annual Savings (£ thousands)')
    ax.set_title('(c)  Annual Fleet Savings — UA vs Blind (50 Engines)')
    ax.legend(loc='upper right', framealpha=0.9)

    # Total across all noise levels
    total_saving_raw = sum(savings_raw) / len(savings_raw)
    total_saving_safe = sum(savings_safe) / len(savings_safe)
    ax.text(0.02, 0.95, f'Avg annual saving:\n  No supervisor: £{total_saving_raw/1000:,.0f}k\n'
                         f'  Full system:    £{total_saving_safe/1000:,.0f}k',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=ACCENT_GREEN, alpha=0.95))

    # ─── (d) Cost Breakdown at σ=0.10 (representative noise) ─
    ax = axes[1, 1]
    # Pick noise level closest to 0.10
    idx = min(range(len(nl)), key=lambda i: abs(nl[i] - 0.10))

    categories = ['Optimal\n(Jackpot)', 'Acceptable\n(Safe)', 'Wasteful\n(Too Early)', 'Unplanned\nFailure']
    cat_costs  = [COST_JACKPOT, COST_SAFE, COST_WASTEFUL, COST_CRASH]
    cat_colors = [ACCENT_GREEN, '#48dbfb', ACCENT_AMBER, BLIND_COLOR]

    # Use safety results for the realistic scenario
    ua_r = ua_safety_results[idx]
    bl_r = bl_safety_results[idx]

    ua_pcts = [ua_r['n_jackpot']/ua_r['n']*100, ua_r['n_safe']/ua_r['n']*100,
               ua_r['n_wasteful']/ua_r['n']*100, ua_r['n_crash']/ua_r['n']*100]
    bl_pcts = [bl_r['n_jackpot']/bl_r['n']*100, bl_r['n_safe']/bl_r['n']*100,
               bl_r['n_wasteful']/bl_r['n']*100, bl_r['n_crash']/bl_r['n']*100]

    x = np.arange(len(categories))
    bw2 = 0.35
    bars_ua = ax.bar(x - bw2/2, ua_pcts, bw2, color=UA_COLOR, alpha=0.85,
                     edgecolor='white', lw=0.6, label='UA Agent')
    bars_bl = ax.bar(x + bw2/2, bl_pcts, bw2, color=BLIND_COLOR, alpha=0.75,
                     edgecolor='white', lw=0.6, label='Blind Agent')

    # Value labels
    for bars in [bars_ua, bars_bl]:
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                        f'{h:.1f}%', ha='center', va='bottom', fontsize=8)

    # Add cost annotations below categories
    for i, c in enumerate(cat_costs):
        ax.text(i, -4.5, f'£{c:,}', ha='center', fontsize=8, color='grey', style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Occurrence Rate (%)')
    ax.set_title(f'(d)  Outcome Distribution at σ = {nl[idx]:.2f} (Full System)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(-7, max(max(ua_pcts), max(bl_pcts)) * 1.25)
    ax.text(0.5, -0.12, 'Unit cost per event shown in italics below each category',
            transform=ax.transAxes, ha='center', fontsize=8, color='grey')

    # ─── Suptitle ──────────────────────────────────────────────
    fig.suptitle('Cost-Effectiveness Analysis — UA vs Blind Maintenance Strategy\n'
                 f'Fleet: {FLEET_SIZE} Turbofan Engines  ·  {ANNUAL_DECISIONS} Decisions/Year  ·  '
                 f'{NUM_EPISODES} Simulated Episodes per Condition',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig('report_cost_analysis.png')
    print("  Saved: report_cost_analysis.png")


def plot_executive_summary(noise_levels, ua_results, bl_results,
                           ua_safety_results, bl_safety_results):
    """
    Single-panel executive summary: 5-year cumulative cost projection.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    years = np.arange(1, 6)  # 1-5 years

    # Use average across all noise levels (realistic: sensors degrade over time)
    ua_annual = np.mean([annual_fleet_cost(r) for r in ua_safety_results])
    bl_annual = np.mean([annual_fleet_cost(r) for r in bl_safety_results])

    ua_cumulative = [ua_annual * y for y in years]
    bl_cumulative = [bl_annual * y for y in years]
    savings_cumulative = [(b - u) for u, b in zip(ua_cumulative, bl_cumulative)]

    # Plot bars
    bw = 0.25
    ax.bar(years - bw, [c/1_000_000 for c in bl_cumulative], bw * 2,
           color=BLIND_COLOR, alpha=0.80, edgecolor='white', lw=0.6,
           label='Blind Agent — Total Cost')
    ax.bar(years + bw, [c/1_000_000 for c in ua_cumulative], bw * 2,
           color=UA_COLOR, alpha=0.80, edgecolor='white', lw=0.6,
           label='UA Agent — Total Cost')

    # Savings annotation line
    for i, y in enumerate(years):
        saving = savings_cumulative[i]
        bl_h = bl_cumulative[i] / 1_000_000
        ua_h = ua_cumulative[i] / 1_000_000
        # Draw bracket
        mid = (bl_h + ua_h) / 2
        ax.annotate(f'£{saving/1_000_000:.2f}M\nsaved',
                    xy=(y + 0.35, mid), fontsize=9, fontweight='bold',
                    color=ACCENT_GREEN, ha='left', va='center')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Cumulative Maintenance Cost (£ millions)', fontsize=12)
    ax.set_xticks(years)
    ax.set_xticklabels([f'Year {y}' for y in years])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'£{v:.1f}M'))
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    ax.set_title(f'5-Year Cost Projection — {FLEET_SIZE}-Engine Fleet\n'
                 f'Uncertainty-Aware Maintenance vs Blind Approach',
                 fontsize=14, fontweight='bold')

    # Summary box
    total_5yr_saving = savings_cumulative[-1]
    pct_saving = (total_5yr_saving / bl_cumulative[-1]) * 100 if bl_cumulative[-1] > 0 else 0
    summary_text = (
        f'5-Year Summary\n'
        f'─────────────────────\n'
        f'Blind total:   £{bl_cumulative[-1]/1_000_000:.2f}M\n'
        f'UA total:       £{ua_cumulative[-1]/1_000_000:.2f}M\n'
        f'──────────────\n'
        f'Total saved:   £{total_5yr_saving/1_000_000:.2f}M\n'
        f'Saving:        {pct_saving:.1f}%'
    )
    ax.text(0.98, 0.55, summary_text, transform=ax.transAxes, fontsize=10,
            va='top', ha='right', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.6', fc='#f0f9ff', ec=UA_COLOR, alpha=0.95))

    plt.tight_layout()
    fig.savefig('report_cost_executive_summary.png')
    print("  Saved: report_cost_executive_summary.png")


# ============================================================
#  CSV EXPORT
# ============================================================

def export_csv(noise_levels, ua_raw, bl_raw, ua_safe, bl_safe):
    """Export detailed cost breakdown to CSV."""
    rows = []
    for i, nl in enumerate(noise_levels):
        for scenario, ua_r, bl_r in [('No_Supervisor', ua_raw[i], bl_raw[i]),
                                      ('Full_System',   ua_safe[i], bl_safe[i])]:
            ua_annual = ua_r['mean_cost'] * ANNUAL_DECISIONS
            bl_annual = bl_r['mean_cost'] * ANNUAL_DECISIONS
            saving = bl_annual - ua_annual

            rows.append({
                'Scenario': scenario,
                'Noise_Level': nl,
                'UA_Cost_Per_Decision': round(ua_r['mean_cost'], 0),
                'UA_Cost_CI95': round(ua_r['ci_95'], 0),
                'Blind_Cost_Per_Decision': round(bl_r['mean_cost'], 0),
                'Blind_Cost_CI95': round(bl_r['ci_95'], 0),
                'UA_Annual_Fleet': round(ua_annual, 0),
                'Blind_Annual_Fleet': round(bl_annual, 0),
                'Annual_Saving_GBP': round(saving, 0),
                'Saving_%': round(saving / bl_annual * 100 if bl_annual > 0 else 0, 1),
                'UA_Jackpot%': round(ua_r['jackpot_rate'], 1),
                'Blind_Jackpot%': round(bl_r['jackpot_rate'], 1),
                'UA_Crash%': round(ua_r['crash_rate'], 1),
                'Blind_Crash%': round(bl_r['crash_rate'], 1),
            })
    df = pd.DataFrame(rows)
    df.to_csv('report_cost_summary.csv', index=False)
    print("  Saved: report_cost_summary.csv")
    return df


# ============================================================
#  MAIN
# ============================================================

def main():
    print('=' * 65)
    print('  COST-EFFECTIVENESS ANALYSIS')
    print('  Fleet: {} engines × {} decisions/year = {} decisions/year'.format(
        FLEET_SIZE, DECISIONS_PER_ENGINE, ANNUAL_DECISIONS))
    print('=' * 65)

    # ── Load data ──────────────────────────────────────────────
    print('\n[1/4] Loading data & models ...')
    df = load_combined_data()
    df = calculate_rul(df)
    df_clean, _ = process_data(df, DROP_SENSORS, DROP_SETTINGS)

    ua_model    = DQN.load('models/dqn_pdm_agent')
    blind_model = DQN.load('models/dqn_blind_agent')
    print('      Both agents loaded.\n')

    # ── Evaluate ──────────────────────────────────────────────
    print('[2/4] Running cost evaluation ({} eps × {} noise levels × 2 agents × 2 modes) ...'.format(
        NUM_EPISODES, len(NOISE_LEVELS)))

    ua_raw_results   = []  # Layer 2 only
    bl_raw_results   = []
    ua_safe_results  = []  # Full system (with supervisor)
    bl_safe_results  = []

    for i, nl in enumerate(NOISE_LEVELS):
        print(f'      Noise σ={nl:.3f} ({i+1}/{len(NOISE_LEVELS)}) ...', end=' ', flush=True)

        if nl == 0:
            ua_env = PdMEnvironment(df_clean, 'models', noise_prob=0.0)
            bl_env = BlindPdMEnvironment(df_clean, 'models', noise_prob=0.0)
        else:
            ua_env = PdMEnvironment(df_clean, 'models', noise_prob=1.0, noise_level=nl)
            bl_env = BlindPdMEnvironment(df_clean, 'models', noise_prob=1.0, noise_level=nl)

        # Layer 2 only (no supervisor)
        ua_raw_results.append(evaluate_outcomes(ua_env, ua_model, NUM_EPISODES, use_safety=False))
        bl_raw_results.append(evaluate_outcomes(bl_env, blind_model, NUM_EPISODES, use_safety=False))

        # Full system (with supervisor)
        ua_safe_results.append(evaluate_outcomes(ua_env, ua_model, NUM_EPISODES, use_safety=True))
        bl_safe_results.append(evaluate_outcomes(bl_env, blind_model, NUM_EPISODES, use_safety=True))

        ua_cost = ua_safe_results[-1]['mean_cost']
        bl_cost = bl_safe_results[-1]['mean_cost']
        saving = bl_cost - ua_cost
        print(f'UA: £{ua_cost:,.0f}  Blind: £{bl_cost:,.0f}  '
              f'Saving: £{saving:,.0f}/decision')

    # ── Summary Stats ────────────────────────────────────────
    print('\n' + '─' * 60)
    avg_ua_cost  = np.mean([r['mean_cost'] for r in ua_safe_results])
    avg_bl_cost  = np.mean([r['mean_cost'] for r in bl_safe_results])
    avg_saving   = avg_bl_cost - avg_ua_cost
    annual_save  = avg_saving * ANNUAL_DECISIONS
    five_yr_save = annual_save * 5

    print(f'  Average cost per decision (full system):')
    print(f'    UA Agent:    £{avg_ua_cost:>10,.0f}')
    print(f'    Blind Agent: £{avg_bl_cost:>10,.0f}')
    print(f'    Saving:      £{avg_saving:>10,.0f}/decision ({avg_saving/avg_bl_cost*100:.1f}%)')
    print(f'\n  Annual fleet saving ({FLEET_SIZE} engines):  £{annual_save:>12,.0f}')
    print(f'  5-year fleet saving:               £{five_yr_save:>12,.0f}')
    print('─' * 60)

    # ── Plots ────────────────────────────────────────────────
    print('\n[3/4] Generating cost analysis figures ...')
    plot_cost_analysis(NOISE_LEVELS, ua_raw_results, bl_raw_results,
                       ua_safe_results, bl_safe_results)
    plot_executive_summary(NOISE_LEVELS, ua_raw_results, bl_raw_results,
                           ua_safe_results, bl_safe_results)

    # ── CSV ──────────────────────────────────────────────────
    print('\n[4/4] Exporting CSV ...')
    cost_df = export_csv(NOISE_LEVELS, ua_raw_results, bl_raw_results,
                         ua_safe_results, bl_safe_results)

    # Print a nice table
    print('\n' + '=' * 65)
    print('  ANNUAL FLEET COST (FULL SYSTEM) BY NOISE LEVEL')
    print('=' * 65)
    print(f'  {"σ":>6}  {"UA Annual":>14}  {"Blind Annual":>14}  {"Saving":>14}  {"Saving%":>8}')
    print('  ' + '─' * 60)
    for i, nl in enumerate(NOISE_LEVELS):
        ua_a = ua_safe_results[i]['mean_cost'] * ANNUAL_DECISIONS
        bl_a = bl_safe_results[i]['mean_cost'] * ANNUAL_DECISIONS
        sv = bl_a - ua_a
        pct = sv / bl_a * 100 if bl_a > 0 else 0
        print(f'  {nl:>6.3f}  £{ua_a:>12,.0f}  £{bl_a:>12,.0f}  £{sv:>12,.0f}  {pct:>7.1f}%')
    print('=' * 65)

    print('\n✓ Cost analysis complete!')
    print('  Outputs: report_cost_analysis.png')
    print('           report_cost_executive_summary.png')
    print('           report_cost_summary.csv')


if __name__ == '__main__':
    main()
