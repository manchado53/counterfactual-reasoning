"""
Generate Claim 1 Spearman rho table as PNG.
Run: python -m counterfactual_rl.analysis.claim1.frozen_lake.make_table
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

DATA = {
    'untrained': [(0, 0.393, 0.0036), (1, 0.407, 0.0025), (2, 0.158, 0.2580)],
    'mid':       [(0, 0.829, None),    (1, 0.836, None),    (2, 0.629, None)],
    'trained':   [(0, 0.849, None),    (1, 0.926, None),    (2, 0.891, None)],
}
STAGE_LABELS = {
    'untrained': 'Untrained\n(ep 150)',
    'mid':       'Mid-training\n(ep 3900)',
    'trained':   'Fully Trained\n(best)',
}
STAGES = ['untrained', 'mid', 'trained']


def fmt_p(p):
    if p is None:
        return '<0.001'
    if p < 0.0001:
        return '<0.001'
    return f'{p:.3f}'


def main():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.subplots_adjust(top=0.82)
    ax.axis('off')

    col_labels = ['Training Stage', 'Seed', 'Spearman ρ', 'p-value']
    rows = []
    row_colors = []

    stage_colors = {
        'untrained': '#EEF2FF',
        'mid':       '#E8F5E9',
        'trained':   '#FFF8E1',
    }

    for stage in STAGES:
        rhos = [r for _, r, _ in DATA[stage]]
        mean_rho = np.mean(rhos)
        std_rho = np.std(rhos)
        c = stage_colors[stage]

        for i, (seed, rho, pval) in enumerate(DATA[stage]):
            stage_cell = STAGE_LABELS[stage] if i == 1 else ''
            rows.append([stage_cell, str(seed), f'{rho:.3f}', fmt_p(pval)])
            row_colors.append([c, c, c, c])

        # Mean ± std summary row
        rows.append(['', 'mean ± std',
                     f'{mean_rho:.3f} ± {std_rho:.3f}', ''])
        row_colors.append(['#F5F5F5', '#F5F5F5', '#F5F5F5', '#F5F5F5'])

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

    # Bold header
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight='bold', color='white')
        table[0, j].set_facecolor('#37474F')

    # Bold mean rows and summary rows
    mean_row_indices = [4, 8, 12]  # after each group of 3
    for ri in mean_row_indices:
        for j in range(len(col_labels)):
            table[ri, j].set_text_props(fontstyle='italic', color='#424242')

    fig.suptitle('Claim 1 — Spearman ρ: CCE Score vs Oracle Q* Importance',
                 fontsize=12, fontweight='bold', y=0.97)

    out_path = Path(__file__).parents[5] / 'docs' / 'figures' / 'real' / 'claim1' / 'frozen_lake' / 'fig_c1_rho_table.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Table saved → {out_path}')


if __name__ == '__main__':
    main()
