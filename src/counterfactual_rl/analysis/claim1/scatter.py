"""
Fig C1 — CCE score vs Oracle score scatter plot (shared across environments).

Three panels: untrained / mid-training / fully trained.
Each point is one state. Color encodes state category (caller-supplied).
Spearman ρ and p-value annotated per panel.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def plot_c1_scatter(oracle, cce_by_stage, state_colors, stage_labels, out_path,
                    figsize=(12, 4)):
    """
    Parameters
    ----------
    oracle        : dict {state: float}   — oracle importance scores
    cce_by_stage  : dict {'untrained': {state: float}, 'mid': ..., 'trained': ...}
    state_colors  : dict {state: str}     — color per state (e.g. hole proximity)
    stage_labels  : list[str]             — 3 panel titles
    out_path      : str or Path           — where to save the figure
    """
    stages = list(cce_by_stage.keys())
    assert len(stages) == 3, "Expected exactly 3 stages"

    states = sorted(oracle.keys())
    oracle_vals = np.array([oracle[s] for s in states])
    colors = [state_colors.get(s, '#2196F3') for s in states]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, stage, label in zip(axes, stages, stage_labels):
        cce_vals = np.array([cce_by_stage[stage].get(s, 0.0) for s in states])

        rho, pval = spearmanr(oracle_vals, cce_vals)

        ax.scatter(oracle_vals, cce_vals, c=colors, alpha=0.7, s=30, linewidths=0)
        ax.set_xlabel('Oracle Q* Consequence', fontsize=10)
        ax.set_ylabel('CCE Score', fontsize=10)
        ax.set_title(label, fontsize=11)

        pval_str = f'{pval:.3f}' if pval >= 0.001 else '<0.001'
        ax.annotate(f'ρ = {rho:.2f}\np = {pval_str}',
                    xy=(0.97, 0.05), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig
