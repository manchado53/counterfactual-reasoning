"""
Fig C5 — CCE Score vs Oracle Importance scatter for Gardner chess.

Single panel (no training stages); points colored by game phase.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from typing import List

from .score_positions import PHASE_COLORS


def plot_c5_scatter(
    oracle_scores: List[float],
    cce_scores: List[float],
    game_phases: List[str],
    out_path,
    figsize=(6, 5),
):
    """
    Parameters
    ----------
    oracle_scores : oracle importance per position (value-head divergence)
    cce_scores    : CCE Wasserstein score per position
    game_phases   : 'opening' | 'middlegame' | 'endgame' per position
    out_path      : save path for figure
    """
    oracle_arr = np.array(oracle_scores)
    cce_arr    = np.array(cce_scores)
    colors     = [PHASE_COLORS[p] for p in game_phases]

    rho, pval = spearmanr(oracle_arr, cce_arr)
    pval_str = f'{pval:.3f}' if pval >= 0.001 else '<0.001'

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(oracle_arr, cce_arr, c=colors, alpha=0.65, s=25, linewidths=0)

    ax.set_xlabel('Oracle Importance (AlphaZero value-head divergence)', fontsize=11)
    ax.set_ylabel('CCE Total Variation', fontsize=11)
    ax.set_title('Gardner Chess — CCE Score vs Oracle Importance', fontsize=12)

    ax.annotate(
        f'ρ = {rho:.2f}\np = {pval_str}',
        xy=(0.97, 0.05), xycoords='axes fraction',
        ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.8),
    )

    # Phase legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PHASE_COLORS[p],
               markersize=7, label=p.capitalize())
        for p in ['opening', 'middlegame', 'endgame']
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Fig C5 saved → {out_path}')
