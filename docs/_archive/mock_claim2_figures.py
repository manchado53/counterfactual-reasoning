"""
Mock figures for Claim 2 metrics.

Generates synthetic example visualizations showing what each metric's figure
will look like once real experiment data is available. All data is fabricated
from plausible sigmoid learning curves + Gaussian noise.

Run from the repo root:
    python docs/mock_claim2_figures.py

Outputs written to docs/figures/:
    fig1_iqm_curves.png               -- IQM learning curves per environment
    fig2_final_iqm.png                -- Final IQM bar chart
    fig3_steps_to_threshold.png       -- Steps-to-threshold dot chart
    fig4_prob_improvement.png         -- P(method > PER) bar chart
    fig5_perf_profiles.png            -- Performance profiles
    fig6_wallclock.png                -- Wall-clock cost summary
    fig7_cce_correlation.png          -- CCE vs ground-truth (Claim 1 / FrozenLake)
    fig8_training_time_breakdown.png  -- Stacked bar: training vs scoring overhead per algorithm
    fig9_component_breakdown.png      -- Horizontal breakdown: % time per component per algorithm
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)

# ── palette ───────────────────────────────────────────────────────────────────
COLORS = {
    'DQN-Uniform':   '#757575',
    'DQN+PER':       '#2196F3',
    'CCE-only':      '#FF9800',
    'CCE+TD (add)':  '#4CAF50',
    'CCE+TD (mul)':  '#9C27B0',
}
ALGORITHMS = list(COLORS.keys())
ENVS = ['SMAX 3m', 'SMAX 8m', 'Gardner Chess']

N_SEEDS      = 10
N_CHECKPOINTS = 120   # eval checkpoints over full training run
EARLY_CUTOFF  = 48    # checkpoint index separating "early" from "late" (~40%)

THRESHOLDS = {'SMAX 3m': 0.60, 'SMAX 8m': 0.55, 'Gardner Chess': 0.45}

# ── synthetic learning curves ──────────────────────────────────────────────────
# Parameters: (asymptote, steepness, midpoint_x_fraction, noise_std)
CURVE_PARAMS = {
    'SMAX 3m': {
        'DQN-Uniform':  (0.64, 8,  0.58, 0.055),
        'DQN+PER':      (0.68, 9,  0.52, 0.048),
        'CCE-only':     (0.66, 10, 0.44, 0.058),
        'CCE+TD (add)': (0.72, 13, 0.38, 0.045),
        'CCE+TD (mul)': (0.70, 11, 0.40, 0.062),
    },
    'SMAX 8m': {
        'DQN-Uniform':  (0.54, 7,  0.60, 0.065),
        'DQN+PER':      (0.59, 8,  0.54, 0.055),
        'CCE-only':     (0.57, 9,  0.46, 0.065),
        'CCE+TD (add)': (0.63, 11, 0.41, 0.055),
        'CCE+TD (mul)': (0.61, 10, 0.43, 0.072),
    },
    'Gardner Chess': {
        'DQN-Uniform':  (0.52, 6,  0.62, 0.075),
        'DQN+PER':      (0.56, 7,  0.56, 0.065),
        'CCE-only':     (0.54, 8,  0.49, 0.075),
        'CCE+TD (add)': (0.59, 10, 0.43, 0.060),
        'CCE+TD (mul)': (0.57, 9,  0.46, 0.080),
    },
}


def sigmoid_curve(n, asymptote, steepness, midpoint, noise_std, n_seeds):
    x = np.linspace(0, 1, n)
    base = asymptote / (1 + np.exp(-steepness * (x - midpoint)))
    noise = RNG.normal(0, noise_std, (n_seeds, n))
    return np.clip(base[None] + noise, 0, 1)


def iqm(arr):
    q25, q75 = np.percentile(arr, [25, 75])
    mask = (arr >= q25) & (arr <= q75)
    return float(arr[mask].mean()) if mask.any() else float(arr.mean())


def iqm_curve(curves):
    return np.array([iqm(curves[:, t]) for t in range(curves.shape[1])])


def bootstrap_ci(curves, n_boot=2000, alpha=0.05):
    n_seeds = curves.shape[0]
    boot = np.array([iqm_curve(curves[RNG.integers(0, n_seeds, n_seeds)])
                     for _ in range(n_boot)])
    lo = np.percentile(boot, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


# Pre-compute all curves once
ALL_CURVES = {}
for env in ENVS:
    ALL_CURVES[env] = {}
    for alg in ALGORITHMS:
        p = CURVE_PARAMS[env][alg]
        ALL_CURVES[env][alg] = sigmoid_curve(N_CHECKPOINTS, *p, N_SEEDS)


# ── Figure 1: IQM learning curves ────────────────────────────────────────────
def fig1_iqm_curves():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    fig.suptitle('Figure 1  —  IQM Win Rate (± 95% CI) across Training\n'
                 '[MOCK DATA]', fontsize=11, y=1.01)

    x = np.arange(N_CHECKPOINTS)
    early_shade_alpha = 0.07

    for ax, env in zip(axes, ENVS):
        thresh = THRESHOLDS[env]

        # shade early-training window
        ax.axvspan(0, EARLY_CUTOFF, color='gold', alpha=early_shade_alpha,
                   label='Early window' if env == ENVS[0] else '_')

        # threshold line
        ax.axhline(thresh, color='#555', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.text(N_CHECKPOINTS - 1, thresh + 0.012, f'threshold={thresh:.0%}',
                ha='right', va='bottom', fontsize=7, color='#555')

        for alg in ALGORITHMS:
            curves = ALL_CURVES[env][alg]
            iqm_vals = iqm_curve(curves)
            lo, hi = bootstrap_ci(curves)
            c = COLORS[alg]
            ax.plot(x, iqm_vals, color=c, linewidth=2.0, label=alg)
            ax.fill_between(x, lo, hi, color=c, alpha=0.15)

        ax.set_title(env, fontsize=10, fontweight='bold')
        ax.set_xlabel('Training checkpoints', fontsize=9)
        ax.set_ylabel('IQM win rate', fontsize=9)
        ax.set_ylim(0.15, 0.85)
        ax.set_xlim(0, N_CHECKPOINTS - 1)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis='y', alpha=0.3, linewidth=0.6)

    handles = [mpatches.Patch(color=COLORS[a], label=a) for a in ALGORITHMS]
    handles += [mpatches.Patch(color='gold', alpha=0.4, label='Early window')]
    fig.legend(handles=handles, loc='lower center', ncol=6,
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_iqm_curves.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ── Figure 2: Final IQM bar chart ────────────────────────────────────────────
def fig2_final_iqm():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    fig.suptitle('Figure 2  —  Final IQM Win Rate (last 10% of training)\n'
                 '[MOCK DATA]', fontsize=11, y=1.01)

    cutoff = int(0.9 * N_CHECKPOINTS)

    for ax, env in zip(axes, ENVS):
        iqms, cis_lo, cis_hi = [], [], []
        for alg in ALGORITHMS:
            curves = ALL_CURVES[env][alg]
            final = curves[:, cutoff:].mean(axis=1)  # per-seed final score
            val = iqm(final)
            # bootstrap CI on scalar IQM
            boot = [iqm(final[RNG.integers(0, N_SEEDS, N_SEEDS)])
                    for _ in range(5000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            iqms.append(val)
            cis_lo.append(val - lo)
            cis_hi.append(hi - val)

        y = np.arange(len(ALGORITHMS))
        bars = ax.barh(y, iqms, xerr=[cis_lo, cis_hi],
                       color=[COLORS[a] for a in ALGORITHMS],
                       error_kw=dict(ecolor='#333', capsize=4, linewidth=1.2),
                       height=0.6, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(ALGORITHMS, fontsize=8.5)
        ax.set_xlabel('IQM win rate', fontsize=9)
        ax.set_title(env, fontsize=10, fontweight='bold')
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_xlim(0.3, 0.85)
        ax.axvline(iqms[ALGORITHMS.index('DQN+PER')], color='#2196F3',
                   linewidth=1, linestyle=':', alpha=0.7)
        ax.grid(axis='x', alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_final_iqm.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ── Figure 3: Steps-to-threshold dot chart ───────────────────────────────────
def fig3_steps_to_threshold():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    fig.suptitle('Figure 3  —  Steps to Threshold (median + IQR across seeds)\n'
                 '[MOCK DATA]', fontsize=11, y=1.01)

    # Synthetic steps-to-threshold: earlier = fewer steps = better
    # Shape: relative fractions of total training budget
    STEP_FRACTIONS = {
        'DQN-Uniform':  [0.72, 0.68, 0.75],
        'DQN+PER':      [0.58, 0.55, 0.62],
        'CCE-only':     [0.50, 0.48, 0.54],
        'CCE+TD (add)': [0.38, 0.36, 0.42],
        'CCE+TD (mul)': [0.43, 0.40, 0.47],
    }
    TOTAL_STEPS = {'SMAX 3m': 25000, 'SMAX 8m': 50000, 'Gardner Chess': 100000}

    for ax, env, (_, total) in zip(axes, ENVS,
                                    [(e, TOTAL_STEPS[e]) for e in ENVS]):
        y = np.arange(len(ALGORITHMS))
        medians, q25s, q75s = [], [], []

        for alg in ALGORITHMS:
            frac = STEP_FRACTIONS[alg][ENVS.index(env)]
            noise_seeds = RNG.normal(0, 0.05, N_SEEDS)
            per_seed = np.clip(frac + noise_seeds, 0.1, 0.99)
            med = np.median(per_seed) * total
            q25 = np.percentile(per_seed, 25) * total
            q75 = np.percentile(per_seed, 75) * total
            medians.append(med)
            q25s.append(q25)
            q75s.append(q75)

        # horizontal lollipop
        for i, alg in enumerate(ALGORITHMS):
            ax.hlines(i, q25s[i], q75s[i], colors=COLORS[alg],
                      linewidth=3.5, alpha=0.4)
            ax.plot(medians[i], i, 'o', color=COLORS[alg],
                    markersize=8, zorder=5)
            ax.text(medians[i], i + 0.22,
                    f'{int(medians[i]):,}', ha='center', fontsize=7.5,
                    color=COLORS[alg], fontweight='bold')

        ax.set_yticks(y)
        ax.set_yticklabels(ALGORITHMS if env == ENVS[0] else [], fontsize=8.5)
        ax.set_xlabel('Training steps', fontsize=9)
        ax.set_title(f'{env}\n(threshold: {THRESHOLDS[env]:.0%})',
                     fontsize=10, fontweight='bold')
        ax.set_xlim(0, total * 1.05)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{int(v/1000)}k'))
        ax.grid(axis='x', alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_steps_to_threshold.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ── Figure 4: P(method > PER) bar chart ──────────────────────────────────────
def fig4_prob_improvement():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    fig.suptitle('Figure 4  —  P(algorithm > DQN+PER) on final win rate\n'
                 '[MOCK DATA]', fontsize=11, y=1.01)

    CCE_ALGS = ['DQN-Uniform', 'CCE-only', 'CCE+TD (add)', 'CCE+TD (mul)']
    # Synthetic P(improvement) and CI half-widths per env
    P_IMP = {
        'SMAX 3m':      [0.28, 0.55, 0.78, 0.70],
        'SMAX 8m':      [0.25, 0.52, 0.74, 0.65],
        'Gardner Chess': [0.30, 0.50, 0.72, 0.63],
    }
    CI_HW = 0.13  # half-width; wide because n=30 pairs

    for ax, env in zip(axes, ENVS):
        probs = P_IMP[env]
        y = np.arange(len(CCE_ALGS))
        colors = [COLORS[a] for a in CCE_ALGS]

        bars = ax.barh(y, probs, xerr=CI_HW,
                       color=colors,
                       error_kw=dict(ecolor='#333', capsize=4, linewidth=1.2),
                       height=0.55, alpha=0.85)
        ax.axvline(0.5, color='#333', linewidth=1.2, linestyle='--', alpha=0.7)
        ax.text(0.5, len(CCE_ALGS) - 0.15, 'chance\n(0.5)',
                ha='center', va='top', fontsize=7.5, color='#555')
        ax.set_yticks(y)
        ax.set_yticklabels(CCE_ALGS if env == ENVS[0] else [], fontsize=8.5)
        ax.set_xlabel('P(method > PER)', fontsize=9)
        ax.set_title(env, fontsize=10, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis='x', alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_prob_improvement.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ── Figure 5: Performance profiles ───────────────────────────────────────────
def fig5_perf_profiles():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    fig.suptitle('Figure 5  —  Performance Profiles (fraction of runs ≥ τ)\n'
                 '[MOCK DATA — Appendix]', fontsize=11, y=1.01)

    tau = np.linspace(0.1, 0.9, 200)
    cutoff = int(0.9 * N_CHECKPOINTS)

    for ax, env in zip(axes, ENVS):
        for alg in ALGORITHMS:
            curves = ALL_CURVES[env][alg]
            final_scores = curves[:, cutoff:].mean(axis=1)  # (n_seeds,)
            frac_above = np.array([(final_scores >= t).mean() for t in tau])
            ax.plot(tau, frac_above, color=COLORS[alg], linewidth=2.0, label=alg)

        ax.axvline(THRESHOLDS[env], color='#555', linewidth=0.8,
                   linestyle='--', alpha=0.6)
        ax.set_xlabel('Win rate τ', fontsize=9)
        ax.set_ylabel('Fraction of runs ≥ τ' if env == ENVS[0] else '',
                      fontsize=9)
        ax.set_title(env, fontsize=10, fontweight='bold')
        ax.set_xlim(0.1, 0.9)
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=8)

    handles = [mpatches.Patch(color=COLORS[a], label=a) for a in ALGORITHMS]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig5_perf_profiles.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ── Figure 6: Wall-clock cost ─────────────────────────────────────────────────
def fig6_wallclock():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('Figure 6  —  Compute Cost\n[MOCK DATA]', fontsize=11, y=1.01)

    # Left: wall-clock per update step (seconds), SMAX 3m only as representative
    # CCE adds rollout overhead; PER is baseline; Uniform is cheapest
    per_step_seconds = {
        'DQN-Uniform':  0.38,
        'DQN+PER':      0.42,
        'CCE-only':     1.85,
        'CCE+TD (add)': 1.92,
        'CCE+TD (mul)': 1.95,
    }
    algs = list(per_step_seconds.keys())
    vals = [per_step_seconds[a] for a in algs]
    colors = [COLORS[a] for a in algs]

    bars = ax1.bar(range(len(algs)), vals, color=colors, alpha=0.85, width=0.6)
    ax1.set_xticks(range(len(algs)))
    ax1.set_xticklabels(algs, rotation=18, ha='right', fontsize=8.5)
    ax1.set_ylabel('Seconds per update', fontsize=9)
    ax1.set_title('Wall-clock per update step\n(SMAX 3m, representative)',
                  fontsize=9.5, fontweight='bold')
    ax1.axhline(per_step_seconds['DQN+PER'], color='#2196F3',
                linewidth=1, linestyle=':', alpha=0.7)
    ax1.set_ylim(0, 2.5)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.6)
    ax1.tick_params(labelsize=8)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.04,
                 f'{v:.2f}s', ha='center', va='bottom', fontsize=8)

    # Right: estimated wall-clock to threshold (steps-to-thresh × time-per-step)
    # SMAX 3m, using mock median steps from fig3
    STEP_MED = {
        'DQN-Uniform':  0.72 * 25000,
        'DQN+PER':      0.58 * 25000,
        'CCE-only':     0.50 * 25000,
        'CCE+TD (add)': 0.38 * 25000,
        'CCE+TD (mul)': 0.43 * 25000,
    }
    wc_to_thresh_hours = {
        a: (STEP_MED[a] * per_step_seconds[a]) / 3600 for a in algs
    }
    wc_vals = [wc_to_thresh_hours[a] for a in algs]
    bars2 = ax2.bar(range(len(algs)), wc_vals, color=colors, alpha=0.85, width=0.6)
    ax2.set_xticks(range(len(algs)))
    ax2.set_xticklabels(algs, rotation=18, ha='right', fontsize=8.5)
    ax2.set_ylabel('Wall-clock hours', fontsize=9)
    ax2.set_title('Estimated wall-clock to threshold\n(steps-to-threshold × time-per-step)',
                  fontsize=9.5, fontweight='bold')
    ax2.axhline(wc_to_thresh_hours['DQN+PER'], color='#2196F3',
                linewidth=1, linestyle=':', alpha=0.7)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.6)
    ax2.tick_params(labelsize=8)
    for bar, v in zip(bars2, wc_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                 f'{v:.1f}h', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig6_wallclock.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ── Figure 7: CCE-vs-ground-truth correlation (FrozenLake / Claim 1) ──────────
def fig7_cce_correlation():
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle('Figure 7  —  CCE Score vs Ground-Truth Consequence\n'
                 '(FrozenLake 4×4, Claim 1 / Mechanistic Validation)\n'
                 '[MOCK DATA]', fontsize=10)

    # Ground-truth: exact DP-computed expected return difference between best and
    # taken action, for each (state, action) pair in training buffer.
    # CCE estimate: algorithm's Wasserstein-based score for the same transitions.
    n_points = 200
    true_consequence = RNG.exponential(scale=0.3, size=n_points)
    true_consequence = np.clip(true_consequence, 0, 1.2)

    # CCE estimate is correlated but noisy, with some bias at high values
    noise = RNG.normal(0, 0.08, n_points)
    cce_estimate = 0.85 * true_consequence + 0.05 + noise
    cce_estimate = np.clip(cce_estimate, 0, 1.2)

    # Colour by state type (some states are "pivotal" — at forks near holes)
    state_type = RNG.choice(['regular', 'pivotal'], n_points, p=[0.7, 0.3])
    colors_pt = ['#2196F3' if s == 'regular' else '#F44336' for s in state_type]

    ax.scatter(true_consequence, cce_estimate, c=colors_pt, alpha=0.55,
               s=28, edgecolors='none')

    # Regression line
    m, b = np.polyfit(true_consequence, cce_estimate, 1)
    x_line = np.linspace(0, 1.2, 100)
    ax.plot(x_line, m * x_line + b, color='#333', linewidth=1.5,
            linestyle='--', label=f'OLS fit (slope={m:.2f})')

    # Perfect-correlation reference
    ax.plot([0, 1.2], [0, 1.2], color='#aaa', linewidth=1, linestyle=':',
            label='Perfect correlation')

    # Spearman ρ annotation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(true_consequence, cce_estimate)
    ax.text(0.04, 1.08, f'Spearman ρ = {rho:.3f}  (p < 0.001)',
            fontsize=9.5, color='#222',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5',
                      edgecolor='#ccc', alpha=0.9))

    patches = [
        mpatches.Patch(color='#2196F3', label='Regular state'),
        mpatches.Patch(color='#F44336', label='Pivotal state (near hole)'),
    ]
    ax.legend(handles=patches + [
        plt.Line2D([0], [0], color='#333', linewidth=1.5, linestyle='--',
                   label=f'OLS fit (slope={m:.2f})'),
        plt.Line2D([0], [0], color='#aaa', linewidth=1, linestyle=':',
                   label='Perfect correlation'),
    ], fontsize=8.5, loc='upper left', framealpha=0.9)

    ax.set_xlabel('Ground-truth consequence\n(DP-computed expected return gap)', fontsize=9)
    ax.set_ylabel('CCE estimated consequence score\n(Wasserstein, Algorithm 1)', fontsize=9)
    ax.set_xlim(-0.05, 1.3)
    ax.set_ylim(-0.05, 1.3)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig7_cce_correlation.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ── Figure 8: Training time breakdown — stacked bar (training vs scoring overhead) ──
def fig8_training_time_breakdown():
    """
    Stacked bar chart: one bar per algorithm, split into training time (blue)
    and scoring overhead (red). Annotated with total hours and X.Xx multiplier
    relative to DQN-Uniform. One panel per environment.

    Data comes from timing.jsonl — the `update.training` and `update.scoring`
    fields per episode, summed across the full run.
    """
    ENVS_ALL = ['SMAX 3m', 'SMAX 8m', 'Gardner Chess', 'FrozenLake']
    ALGS = ['DQN-Uniform', 'DQN+PER', 'CCE-only', 'CCE+TD (add)', 'CCE+TD (mul)']

    # Synthetic hours: (training_h, scoring_overhead_h) per algorithm per environment
    # Scoring overhead is ~0 for baselines, large for CCE variants.
    TIME_DATA = {
        'SMAX 3m': {
            'DQN-Uniform':  (0.37, 0.00),
            'DQN+PER':      (0.35, 0.02),
            'CCE-only':     (0.28, 1.10),
            'CCE+TD (add)': (0.28, 1.32),
            'CCE+TD (mul)': (0.28, 1.35),
        },
        'SMAX 8m': {
            'DQN-Uniform':  (0.72, 0.00),
            'DQN+PER':      (0.70, 0.04),
            'CCE-only':     (0.55, 2.20),
            'CCE+TD (add)': (0.55, 2.65),
            'CCE+TD (mul)': (0.55, 2.70),
        },
        'Gardner Chess': {
            'DQN-Uniform':  (1.10, 0.00),
            'DQN+PER':      (1.08, 0.06),
            'CCE-only':     (0.90, 3.80),
            'CCE+TD (add)': (0.90, 4.20),
            'CCE+TD (mul)': (0.90, 4.30),
        },
        'FrozenLake': {
            'DQN-Uniform':  (0.04, 0.00),
            'DQN+PER':      (0.04, 0.00),
            'CCE-only':     (0.03, 0.12),
            'CCE+TD (add)': (0.03, 0.14),
            'CCE+TD (mul)': (0.03, 0.14),
        },
    }

    TRAIN_COLOR   = '#2196F3'   # blue — training
    SCORING_COLOR = '#F44336'   # red  — scoring overhead

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=False)
    fig.suptitle('Figure 8  —  Training Time Breakdown: Training vs. Scoring Overhead\n'
                 '[MOCK DATA]', fontsize=11, y=1.02)

    for ax, env in zip(axes, ENVS_ALL):
        data = TIME_DATA[env]
        x = np.arange(len(ALGS))
        train_h   = np.array([data[a][0] for a in ALGS])
        scoring_h = np.array([data[a][1] for a in ALGS])
        total_h   = train_h + scoring_h
        baseline  = total_h[0]   # DQN-Uniform is the reference

        bars_train   = ax.bar(x, train_h,   color=TRAIN_COLOR,   alpha=0.9,
                              width=0.55, label='Training')
        bars_scoring = ax.bar(x, scoring_h, bottom=train_h,
                              color=SCORING_COLOR, alpha=0.9,
                              width=0.55, label='Scoring overhead')

        # Annotate each bar with total hours and slowdown multiplier
        for i, (tot, base) in enumerate(zip(total_h, [baseline] * len(ALGS))):
            multiplier = tot / base if base > 0 else 1.0
            ax.text(x[i], tot + ax.get_ylim()[1] * 0.01,
                    f'{tot:.2f}h', ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', color='#111')
            if i > 0:
                ax.annotate(f'{multiplier:.1f}× slower',
                            xy=(x[i], tot),
                            xytext=(x[i] + 0.05, tot + ax.get_ylim()[1] * 0.08),
                            fontsize=6.5, color='#c0392b',
                            arrowprops=dict(arrowstyle='->', color='#c0392b',
                                            lw=1.0))

        ax.set_xticks(x)
        ax.set_xticklabels(ALGS, rotation=20, ha='right', fontsize=7.5)
        ax.set_ylabel('Wall Time (hours)' if env == ENVS_ALL[0] else '', fontsize=9)
        ax.set_title(env, fontsize=10, fontweight='bold')
        ax.set_ylim(0, max(total_h) * 1.35)
        ax.grid(axis='y', alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=8)

    # Shared legend
    handles = [
        mpatches.Patch(color=TRAIN_COLOR,   label='Training (Q-network updates)'),
        mpatches.Patch(color=SCORING_COLOR, label='Scoring overhead (CCE rollouts)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig8_training_time_breakdown.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ── Figure 9: Component-level timing breakdown (horizontal stacked bar) ───────
def fig9_component_breakdown():
    """
    Horizontal stacked bar — one row per algorithm, each bar divided into
    timing components: env, action, buffer.add, scoring:rollouts,
    scoring:buffer_update, scoring:beam, scoring:sample, scoring:stack,
    scoring:metrics, q_update, eval.

    Shows WHERE exactly the wall-clock time goes for each algorithm.
    Baselines have no scoring components. CCE variants are dominated by
    scoring:rollouts and scoring:buffer_update.

    Data comes from timing.jsonl — each episode logs time per component.
    """
    ALGS = ['DQN-Uniform', 'DQN+PER', 'CCE-only', 'CCE+TD (add)', 'CCE+TD (mul)']

    # Component colours matching the existing timing_breakdown.png style
    COMPONENTS = [
        ('env',                    '#2196F3'),   # blue
        ('action',                 '#26A69A'),   # teal
        ('buffer.add',             '#7E57C2'),   # purple
        ('scoring: beam',          '#E91E63'),   # pink-red
        ('scoring: buffer_update', '#EC407A'),   # magenta
        ('scoring: metrics',       '#FDD835'),   # yellow
        ('scoring: rollouts',      '#FF9800'),   # orange
        ('scoring: sample',        '#BF360C'),   # burnt orange
        ('scoring: stack',         '#6D4C41'),   # brown
        ('q_update',               '#4CAF50'),   # green
        ('eval',                   '#FFA726'),   # amber
    ]
    COMP_NAMES  = [c[0] for c in COMPONENTS]
    COMP_COLORS = [c[1] for c in COMPONENTS]

    # Synthetic % breakdown per algorithm (must sum to 100 per row)
    # Baselines have 0% on all scoring components
    PCT = {
        'DQN-Uniform': [22, 12, 8,  0,  0,  0,  0,  0,  0, 28, 30],
        'DQN+PER':     [20, 11, 7,  0,  0,  0,  0,  0,  0, 27, 35],
        'CCE-only':    [ 7,  4, 3,  3, 18,  1, 38,  3,  5,  9, 10],
        'CCE+TD (add)':[ 6,  4, 2,  3, 17,  1, 40,  3,  5,  9, 10],
        'CCE+TD (mul)':[ 6,  4, 2,  3, 17,  1, 40,  3,  5,  9, 10],
    }
    # Synthetic total hours (same as fig8, SMAX 3m as representative)
    TOTAL_H = {
        'DQN-Uniform':  0.37,
        'DQN+PER':      0.37,
        'CCE-only':     1.38,
        'CCE+TD (add)': 1.60,
        'CCE+TD (mul)': 1.63,
    }

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.suptitle('Figure 9  —  Component-Level Time Breakdown per Algorithm\n'
                 '(SMAX 3m, representative)  [MOCK DATA]', fontsize=11, y=1.02)

    y_pos = np.arange(len(ALGS))

    for alg_i, alg in enumerate(ALGS):
        pcts  = PCT[alg]
        total = TOTAL_H[alg]
        left  = 0.0
        for comp_i, (pct, color) in enumerate(zip(pcts, COMP_COLORS)):
            width_h = pct / 100.0 * total
            if width_h < 0.001:
                left += width_h
                continue
            bar = ax.barh(alg_i, width_h, left=left, color=color,
                          height=0.55, alpha=0.92)
            # Label if wide enough to fit text
            if pct >= 5:
                ax.text(left + width_h / 2, alg_i,
                        f'{pct}%', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
            left += width_h

        # Total hours label at end of bar
        ax.text(total + 0.01, alg_i, f'{total:.2f}h',
                va='center', fontsize=8.5, fontweight='bold', color='#222')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ALGS, fontsize=9.5)
    ax.set_xlabel('Time (hours)', fontsize=9)
    ax.set_xlim(0, max(TOTAL_H.values()) * 1.12)
    ax.grid(axis='x', alpha=0.3, linewidth=0.6)
    ax.tick_params(labelsize=8)
    ax.invert_yaxis()

    # Legend
    handles = [mpatches.Patch(color=COMP_COLORS[i], label=COMP_NAMES[i])
               for i in range(len(COMPONENTS))]
    fig.legend(handles=handles, loc='lower center', ncol=6,
               fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.18))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig9_component_breakdown.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ── run all ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating mock figures...')
    fig1_iqm_curves()
    fig2_final_iqm()
    fig3_steps_to_threshold()
    fig4_prob_improvement()
    fig5_perf_profiles()
    fig6_wallclock()
    fig7_cce_correlation()
    fig8_training_time_breakdown()
    fig9_component_breakdown()
    print(f'\nAll figures written to {OUT_DIR}/')
