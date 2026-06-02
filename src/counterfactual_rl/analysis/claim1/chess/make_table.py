"""
Generate Claim 1 chess metrics table as PNG.
Run: python -m counterfactual_rl.analysis.claim1.chess.make_table

Loads metrics from chess_claim1_results.json produced by run_analysis.py.
Multi-seed ρ stats (seeds 0-2, h=20): mean=0.360 ± 0.047.
"""
import json
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_PATH = (Path(__file__).parents[5] / 'docs' / 'figures' / 'real' / 'claim1' / 'chess'
                / 'chess_claim1_results.json')

with open(RESULTS_PATH) as f:
    DATA = json.load(f)

# Multi-seed Spearman ρ across seeds 0,1,2 (h=20, verified from SLURM logs)
_SEED_RHOS = [0.306, 0.384, 0.390]
MULTISEED_MEAN = float(np.mean(_SEED_RHOS))
MULTISEED_STD  = float(np.std(_SEED_RHOS))


def fmt_p(p):
    if p is None:
        return 'TBD'
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'


def fmt_val(v, decimals=3):
    if v is None:
        return 'TBD'
    return f'{v:.{decimals}f}'


def main():
    rows = [
        ['Spearman ρ',           fmt_val(DATA['spearman_rho']),
         '0 (random)',           fmt_p(DATA['spearman_pval'])],
        ['Precision@5%',         fmt_val(DATA['precision_5']),
         '0.05 (random)',        ''],
        ['Precision@10%',        fmt_val(DATA['precision_10']),
         '0.10 (random)',        ''],
        ['Precision@20%',        fmt_val(DATA['precision_20']),
         '0.20 (random)',        ''],
        ['Sampling KL (↓)',      fmt_val(DATA['sampling_kl']),
         '0 (perfect)',          ''],
        ['Sampling Pearson r',   fmt_val(DATA['sampling_pearr']),
         '0 (random)',           ''],
    ]

    col_labels = ['Metric', 'Value', 'Random baseline', 'p-value']

    row_colors_base = [
        '#EEF2FF', '#E8F5E9', '#E8F5E9', '#E8F5E9',
        '#FFF8E1', '#FFF8E1',
    ]
    row_colors = [[c] * 4 for c in row_colors_base]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(top=0.78)
    ax.axis('off')

    n_pos_str = str(DATA['n_positions']) if DATA['n_positions'] else 'TBD'
    title = (f'Claim 1 — Gardner Chess Metrics  '
             f'(n={n_pos_str}, AlphaZero oracle, seed 2)\n'
             f'3-seed ρ: {MULTISEED_MEAN:.3f} ± {MULTISEED_STD:.3f}  (seeds 0–2, h=20)')
    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.97)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=row_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight='bold', color='white')
        table[0, j].set_facecolor('#37474F')

    out_path = (Path(__file__).parents[5] / 'docs' / 'figures' / 'real' / 'claim1' / 'chess'
                / 'fig_c5_chess_rho_table.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Table saved → {out_path}')


if __name__ == '__main__':
    main()
