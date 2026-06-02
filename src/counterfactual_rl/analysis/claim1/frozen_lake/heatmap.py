"""
Fig C2 — 8×8 importance heatmaps for FrozenLake (FL-specific).

Three panels: oracle importance | CCE untrained | CCE trained.
YlOrRd colormap, normalized [0,1] per panel.
Holes shown as dark ✕, goal as G.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from counterfactual_rl.envs.frozen_lake import FrozenLakeEnv, MAPS


_MAP_8x8 = MAPS['8x8']
_GRID_SIZE = 8


def _scores_to_grid(scores, n=8):
    """Convert {state: score} dict to NaN-filled 8×8 array."""
    grid = np.full((n, n), np.nan)
    for s, v in scores.items():
        row, col = divmod(s, n)
        grid[row, col] = v
    return grid


def _normalize(grid):
    """Normalize grid values to [0, 1], ignoring NaNs."""
    vmin = np.nanmin(grid)
    vmax = np.nanmax(grid)
    if vmax - vmin < 1e-10:
        return np.where(np.isnan(grid), np.nan, 0.0)
    return (grid - vmin) / (vmax - vmin)


def plot_c2_heatmaps(oracle, cce_untrained, cce_trained, out_path,
                     map_name='8x8', figsize=(13, 4)):
    """
    Parameters
    ----------
    oracle        : dict {state: float}
    cce_untrained : dict {state: float}
    cce_trained   : dict {state: float}
    out_path      : str or Path
    map_name      : '8x8' (only 8×8 supported)
    """
    desc = _MAP_8x8
    n = _GRID_SIZE

    hole_states = {r * n + c for r, row in enumerate(desc)
                   for c, ch in enumerate(row) if ch == 'H'}
    goal_state = next(r * n + c for r, row in enumerate(desc)
                      for c, ch in enumerate(row) if ch == 'G')

    panels = [
        ('Oracle Q* Importance', oracle),
        ('CCE Untrained (ep 150)', cce_untrained),
        ('CCE Trained (best)',     cce_trained),
    ]

    cmap = plt.cm.YlOrRd
    cmap.set_bad(color='#222222')  # terminal states shown dark

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (title, scores) in zip(axes, panels):
        grid = _normalize(_scores_to_grid(scores, n))

        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1,
                       interpolation='nearest', aspect='equal')

        # Mark holes and goal
        for s in hole_states:
            r, c = divmod(s, n)
            ax.text(c, r, '✕', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        gr, gc = divmod(goal_state, n)
        ax.text(gc, gr, 'G', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig
