"""
State-importance heatmaps for DoorKey Claim 1.

Each grid cell aggregates (max) the score over all enumerated states that sit on that
cell (i.e. over agent direction, key possession, door state). Mirrors the FrozenLake
heatmap but for DoorKey geometry, marking the key (K), door (D), and goal (G).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from counterfactual_rl.envs.doorkey import WALL, KEY, DOOR, GOAL


def _scores_to_grid(scores, env, agg=np.max):
    """Aggregate a {state_idx: score} dict onto the (nrows, ncols) grid."""
    grid = np.full((env.nrows, env.ncols), np.nan)
    buckets = {}
    for si, val in scores.items():
        cell = env.state_to_cell[si]
        buckets.setdefault(cell, []).append(val)
    for (r, c), vals in buckets.items():
        grid[r, c] = agg(vals)
    return grid


def _normalize(grid):
    lo = np.nanmin(grid)
    hi = np.nanmax(grid)
    if hi - lo < 1e-12:
        return grid - lo
    return (grid - lo) / (hi - lo)


def _annotate(ax, env):
    for r in range(env.nrows):
        for c in range(env.ncols):
            t = env.desc[r][c]
            if t == WALL:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='0.25'))
            elif t in (KEY, DOOR, GOAL):
                ax.text(c, r, t, ha='center', va='center', fontsize=12, fontweight='bold',
                        color='black')


def plot_c2_heatmaps(oracle, cce_untrained, cce_trained, env, out_path):
    panels = [
        ('Oracle importance', oracle),
        ('CCE (untrained)', cce_untrained),
        ('CCE (trained)', cce_trained),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, scores) in zip(axes, panels):
        grid = _normalize(_scores_to_grid(scores, env))
        im = ax.imshow(grid, cmap='YlOrRd', vmin=0, vmax=1)
        _annotate(ax, env)
        ax.set_title(title)
        ax.set_xticks(range(env.ncols))
        ax.set_yticks(range(env.nrows))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
