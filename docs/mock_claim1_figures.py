"""
Mock figures for Claim 1 metrics.

Generates synthetic example visualizations showing what each metric's figure
will look like once real experiment data is available. All data is fabricated
from plausible distributions grounded in the FrozenLake 8×8 geometry.

Run from the repo root:
    python docs/mock_claim1_figures.py

Outputs written to docs/figures/claim1/:
    fig_c1_scatter_stages.png     -- CCE vs oracle scatter at 3 training stages
    fig_c2_grid_heatmaps.png      -- 8×8 importance heatmaps: oracle vs CCE
    fig_c3_metric_progression.png -- All 3 metrics over training checkpoints
    fig_c4_precision_at_k.png     -- Precision@K vs random baseline
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from scipy.stats import spearmanr

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures', 'claim1')
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)

# ── FrozenLake 8×8 map ────────────────────────────────────────────────────────
MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]
N = 8

# Build state-type arrays
HOLE_STATES  = set()
GOAL_STATE   = 63
START_STATE  = 0
for r, row in enumerate(MAP):
    for c, ch in enumerate(row):
        if ch == 'H':
            HOLE_STATES.add(r * N + c)

# Non-terminal, non-hole states we score (excludes holes and goal)
SCORED_STATES = [s for s in range(N * N) if s not in HOLE_STATES and s != GOAL_STATE]
N_STATES = len(SCORED_STATES)   # 53 for 8×8 default

def state_to_rc(s):
    return s // N, s % N

# ── Oracle importance scores ───────────────────────────────────────────────────
# Ground truth: Q*(s, optimal) − mean_{a≠opt} Q*(s, a)
# Approximated here by proximity to holes on the near-optimal path.
# States adjacent to a hole (within 1 step) receive high oracle importance.
def _oracle_scores():
    scores = np.zeros(N * N)
    hole_positions = np.array([list(state_to_rc(s)) for s in HOLE_STATES])

    for s in SCORED_STATES:
        r, c = state_to_rc(s)
        # Distance to nearest hole
        dists = np.abs(hole_positions[:, 0] - r) + np.abs(hole_positions[:, 1] - c)
        min_dist = dists.min()

        # States right next to a hole: high importance (wrong action costs a lot)
        # States far from holes: low importance (any action is roughly OK)
        if min_dist == 1:
            scores[s] = RNG.uniform(0.65, 0.95)
        elif min_dist == 2:
            scores[s] = RNG.uniform(0.30, 0.65)
        else:
            scores[s] = RNG.uniform(0.02, 0.30)

    return scores

ORACLE = _oracle_scores()
ORACLE_SCORED = ORACLE[SCORED_STATES]

# Classify state types for scatter coloring
def _state_types():
    hole_positions = np.array([list(state_to_rc(s)) for s in HOLE_STATES])
    types = []
    for s in SCORED_STATES:
        r, c = state_to_rc(s)
        dists = np.abs(hole_positions[:, 0] - r) + np.abs(hole_positions[:, 1] - c)
        if dists.min() == 1:
            types.append('near-hole')
        elif dists.min() == 2:
            types.append('mid-range')
        else:
            types.append('safe')
    return types

STATE_TYPES = _state_types()

TYPE_COLORS = {
    'near-hole': '#F44336',  # red
    'mid-range': '#FF9800',  # orange
    'safe':      '#2196F3',  # blue
}

# ── CCE scores at 3 training stages ───────────────────────────────────────────
# Stage correlation parameters: (rho_target, noise_std, bias)
STAGES = [
    ('Untrained\n(ep ≈ 0)',       0.08, 0.28, 0.05),
    ('Mid-training\n(ep ≈ 7 500)', 0.52, 0.14, 0.03),
    ('Fully trained\n(ep ≈ 15 000)', 0.74, 0.07, 0.01),
]

def _cce_at_stage(rho_target, noise_std, bias):
    """Synthetic CCE scores correlated with ORACLE_SCORED at given rho_target."""
    n = N_STATES
    # Mix oracle signal with noise to achieve approximate target rank correlation
    signal = ORACLE_SCORED + RNG.normal(0, noise_std, n)
    noise  = RNG.normal(0, 0.3, n)
    alpha  = rho_target
    raw    = alpha * signal + (1 - alpha) * noise + bias
    return np.clip(raw, 0, 1.2)

CCE_STAGES = [_cce_at_stage(rho, ns, b) for _, rho, ns, b in STAGES]

# ── Metric progression over training ─────────────────────────────────────────
N_CHECKPOINTS = 100   # 100 checkpoints over 15k episodes
N_SEEDS       = 3
K_VALUE       = 0.10  # Precision@10% for progression plot

def _sigmoid(x, lo, hi, steepness, midpoint):
    return lo + (hi - lo) / (1 + np.exp(-steepness * (x - midpoint)))

def _metric_curves(lo, hi, steepness, midpoint, noise_std):
    x = np.linspace(0, 1, N_CHECKPOINTS)
    base = _sigmoid(x, lo, hi, steepness, midpoint)
    curves = []
    for _ in range(N_SEEDS):
        noise = RNG.normal(0, noise_std, N_CHECKPOINTS)
        # Smooth the noise a bit
        from numpy.lib.stride_tricks import sliding_window_view
        smoothed = np.convolve(noise, np.ones(5)/5, mode='same')
        curves.append(np.clip(base + smoothed, 0, 1))
    return np.array(curves)   # (N_SEEDS, N_CHECKPOINTS)

# Spearman ρ: starts near 0, rises to ~0.74
SPEARMAN_CURVES = _metric_curves(lo=0.05, hi=0.74, steepness=8, midpoint=0.45,
                                  noise_std=0.05)

# Precision@10%: starts near 0.10 (random), rises to ~0.62
PRECISION_CURVES = _metric_curves(lo=0.10, hi=0.62, steepness=8, midpoint=0.48,
                                   noise_std=0.04)

# Sampling KL: starts high (~1.8), falls to near 0
KL_CURVES = _metric_curves(lo=0.08, hi=1.80, steepness=8, midpoint=0.45,
                             noise_std=0.10)
KL_CURVES = KL_CURVES[:, ::-1]   # flip so it decreases


# ── Figure C1: Scatter at 3 training stages ───────────────────────────────────
def fig_c1_scatter_stages():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('Figure C1  —  CCE Score vs Oracle Q* Consequence (FrozenLake 8×8)\n'
                 'Each point is one non-terminal state.  [MOCK DATA]',
                 fontsize=10, y=1.02)

    for ax, (label, rho_target, _, _), cce in zip(axes, STAGES, CCE_STAGES):
        colors_pt = [TYPE_COLORS[t] for t in STATE_TYPES]

        ax.scatter(ORACLE_SCORED, cce, c=colors_pt, alpha=0.65,
                   s=35, edgecolors='none', zorder=3)

        # OLS fit
        m, b = np.polyfit(ORACLE_SCORED, cce, 1)
        x_line = np.linspace(0, 1.05, 100)
        ax.plot(x_line, m * x_line + b, color='#333', linewidth=1.4,
                linestyle='--', zorder=4)

        # Perfect-correlation reference
        ax.plot([0, 1.1], [0, 1.1], color='#bbb', linewidth=1,
                linestyle=':', zorder=2)

        # Spearman ρ annotation
        rho, pval = spearmanr(ORACLE_SCORED, cce)
        p_str = 'p < 0.001' if pval < 0.001 else f'p = {pval:.3f}'
        ax.text(0.04, 1.07, f'Spearman ρ = {rho:.3f}  ({p_str})',
                fontsize=8.5, color='#222',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5',
                          edgecolor='#ccc', alpha=0.9))

        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Oracle Q* importance\n(suboptimality gap)', fontsize=8.5)
        ax.set_ylabel('CCE score (Wasserstein)' if ax is axes[0] else '', fontsize=8.5)
        ax.set_xlim(-0.04, 1.15)
        ax.set_ylim(-0.04, 1.25)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8)

    patches = [mpatches.Patch(color=TYPE_COLORS[t], label=t.replace('-', ' ').title())
               for t in ['near-hole', 'mid-range', 'safe']]
    patches += [
        plt.Line2D([0], [0], color='#333', linewidth=1.4, linestyle='--', label='OLS fit'),
        plt.Line2D([0], [0], color='#bbb', linewidth=1,   linestyle=':',  label='Perfect correlation'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=5,
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.10))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_c1_scatter_stages.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ── Figure C2: 8×8 importance heatmaps ───────────────────────────────────────
def fig_c2_grid_heatmaps():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle('Figure C2  —  Importance Heatmaps: Oracle vs CCE (FrozenLake 8×8)\n'
                 '[MOCK DATA]', fontsize=10, y=1.02)

    titles = ['Oracle Q* Importance\n(ground truth)',
              'CCE Score — Untrained\n(ep ≈ 0)',
              'CCE Score — Fully Trained\n(ep ≈ 15 000)']

    # Build 8×8 grids
    def _to_grid(scores_per_state):
        grid = np.full((N, N), np.nan)
        for i, s in enumerate(SCORED_STATES):
            r, c = state_to_rc(s)
            grid[r, c] = scores_per_state[i]
        return grid

    grids = [
        _to_grid(ORACLE_SCORED),
        _to_grid(CCE_STAGES[0]),   # untrained
        _to_grid(CCE_STAGES[2]),   # fully trained
    ]

    vmin, vmax = 0.0, 1.0
    cmap = plt.cm.YlOrRd

    for ax, grid, title in zip(axes, grids, titles):
        # Background for holes (dark gray)
        bg = np.zeros((N, N))
        im = ax.imshow(bg, cmap='Greys', vmin=0, vmax=1, alpha=0.0)

        # Plot importance (masked — holes stay NaN → show as gray)
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest')

        # Hole cells: dark background
        for s in HOLE_STATES:
            r, c = state_to_rc(s)
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                        color='#37474F', zorder=2))
            ax.text(c, r, '✕', ha='center', va='center',
                    fontsize=9, color='white', zorder=3)

        # Goal marker
        gr, gc = state_to_rc(GOAL_STATE)
        ax.add_patch(plt.Rectangle((gc - 0.5, gr - 0.5), 1, 1,
                                    color='#1B5E20', zorder=2))
        ax.text(gc, gr, 'G', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold', zorder=3)

        # Start marker
        sr, sc = state_to_rc(START_STATE)
        ax.text(sc, sr, 'S', ha='center', va='center',
                fontsize=9, color='#111', fontweight='bold', zorder=4)

        # State indices (tiny, for reference)
        for s in SCORED_STATES:
            r, c = state_to_rc(s)
            ax.text(c, r + 0.35, str(s), ha='center', va='center',
                    fontsize=4.5, color='#555', zorder=4)

        ax.set_title(title, fontsize=9.5, fontweight='bold')
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(range(N), fontsize=7)
        ax.set_yticklabels(range(N), fontsize=7)
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(N - 0.5, -0.5)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Importance score')

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_c2_grid_heatmaps.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ── Figure C3: Metric progression over training ───────────────────────────────
def fig_c3_metric_progression():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle('Figure C3  —  Claim 1 Metrics vs Training Progress  (FrozenLake 8×8, 3 seeds)\n'
                 '[MOCK DATA]', fontsize=10, y=1.02)

    x = np.linspace(0, 15000, N_CHECKPOINTS)
    mid_ep = 7500

    datasets = [
        ('Spearman ρ\n(primary)',        SPEARMAN_CURVES,  '#4CAF50', (0, 1.0),  False),
        ('Precision@10%',                PRECISION_CURVES, '#2196F3', (0, 1.0),  False),
        ('Sampling KL divergence\n(↓ better)', KL_CURVES, '#F44336', (0, 2.2),  True),
    ]
    SEED_COLORS = ['#aaa', '#bbb', '#ccc']

    for ax, (ylabel, curves, color, ylim, invert) in zip(axes, datasets):
        mean = curves.mean(axis=0)
        lo   = np.percentile(curves, 5,  axis=0)
        hi   = np.percentile(curves, 95, axis=0)

        # Individual seed traces
        for i, seed_curve in enumerate(curves):
            ax.plot(x, seed_curve, color=color, alpha=0.25,
                    linewidth=1.0, zorder=2)

        # Mean + CI
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, zorder=3)
        ax.plot(x, mean, color=color, linewidth=2.2, zorder=4,
                label='Mean (3 seeds)')

        # Mark the 3 snapshot stages
        stage_eps = [0, 7500, 15000]
        stage_labels = ['untrained', 'mid', 'trained']
        for ep, lbl in zip(stage_eps, stage_labels):
            idx = np.argmin(np.abs(x - ep))
            ax.axvline(ep, color='#777', linewidth=0.8,
                       linestyle='--', alpha=0.6, zorder=1)
            ax.text(ep + 150, ylim[1] * 0.97, lbl,
                    fontsize=7, color='#555', va='top')

        # Random baseline for Precision@K
        if 'Precision' in ylabel:
            ax.axhline(K_VALUE, color='#888', linewidth=1,
                       linestyle=':', alpha=0.8)
            ax.text(200, K_VALUE + 0.02, 'random baseline',
                    fontsize=7, color='#666')

        ax.set_xlabel('Training episodes', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(0, 15000)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{int(v/1000)}k' if v > 0 else '0'))
        ax.grid(alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_c3_metric_progression.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ── Figure C4: Precision@K ────────────────────────────────────────────────────
def fig_c4_precision_at_k():
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle('Figure C4  —  Precision@K: CCE vs Random Baseline\n'
                 'Fully trained policy, 3 seeds  (FrozenLake 8×8)  [MOCK DATA]',
                 fontsize=10, y=1.02)

    K_THRESHOLDS = [0.05, 0.10, 0.20]
    K_LABELS = ['Top 5%', 'Top 10%', 'Top 20%']

    # Synthetic Precision@K for fully trained policy across 3 seeds
    # Must be > K (beat random) with margin that shrinks at larger K
    MOCK_PRECISION = {
        0.05: [0.68, 0.72, 0.65],
        0.10: [0.58, 0.63, 0.56],
        0.20: [0.47, 0.51, 0.45],
    }

    x = np.arange(len(K_THRESHOLDS))
    width = 0.45

    means = [np.mean(MOCK_PRECISION[k]) for k in K_THRESHOLDS]
    stds  = [np.std(MOCK_PRECISION[k])  for k in K_THRESHOLDS]

    bars = ax.bar(x, means, width, yerr=stds,
                  color='#4CAF50', alpha=0.85,
                  error_kw=dict(ecolor='#333', capsize=5, linewidth=1.3),
                  label='CCE (Wasserstein)', zorder=3)

    # Random baseline bars
    ax.bar(x + width + 0.05, K_THRESHOLDS, width,
           color='#9E9E9E', alpha=0.70,
           label='Random baseline (= K)', zorder=3)

    # Annotate CCE values
    for bar, mean, k in zip(bars, means, K_THRESHOLDS):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + stds[K_THRESHOLDS.index(k)] + 0.01,
                f'{mean:.2f}', ha='center', fontsize=8.5,
                fontweight='bold', color='#2E7D32')

    # Lift annotations
    for i, (mean, k) in enumerate(zip(means, K_THRESHOLDS)):
        lift = mean / k
        ax.annotate(f'{lift:.1f}× random',
                    xy=(x[i] + width / 2, mean),
                    xytext=(x[i] + width / 2 + 0.05, mean + 0.06),
                    fontsize=7.5, color='#1B5E20',
                    arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=0.9))

    ax.set_xticks(x + (width + 0.05) / 2)
    ax.set_xticklabels(K_LABELS, fontsize=9.5)
    ax.set_ylabel('Precision@K', fontsize=9)
    ax.set_ylim(0, 0.95)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linewidth=0.6)
    ax.tick_params(labelsize=8.5)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_c4_precision_at_k.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ── run all ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating Claim 1 mock figures...')
    fig_c1_scatter_stages()
    fig_c2_grid_heatmaps()
    fig_c3_metric_progression()
    fig_c4_precision_at_k()
    print(f'\nAll figures written to {OUT_DIR}/')
