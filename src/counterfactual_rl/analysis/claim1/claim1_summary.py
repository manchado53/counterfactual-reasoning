"""
CLAIM 1, both environments, in one figure — the complete case.

THE CLAIM. CCE scores a state using ONLY policy rollouts: from state s, try each action,
follow the current policy, compare the resulting return distributions. It never sees the
transition table, the reward function, or Q*. The exact oracle computes, by dynamic
programming over the full MDP, how much the action choice REALLY changes the outcome:

    Oracle(s) = mean_{a != a*} [ Q*(s,a*) - Q*(s,a) ]

Claim 1 is that these two rankings agree -- i.e. a cheap rollout estimate recovers a quantity
that otherwise requires solving the MDP exactly.

  (a) what is being compared, and what each side is allowed to see
  (b) rho across training stages, both environments
  (c) precision@k -- of CCE's top k%, how many are truly in the oracle's top k%
  (d) robustness to the aggregation bug, both environments

Run:
    python -m counterfactual_rl.analysis.claim1.claim1_summary --out docs/figures/real/claim1
"""

import argparse
from pathlib import Path

import numpy as np

STAGES = ['untrained', 'mid', 'trained']

# FrozenLake 8x8 slippery, 3 seeds, all 53 non-terminal states (paper/repro checkpoints)
FL = {
    'mean': {'rho': [0.326, 0.791, 0.895], 'sd': [0.105, 0.088, 0.032],
             'prec': {5: 0.333, 10: 0.667, 20: 0.800}},
    'max':  {'rho': [0.319, 0.764, 0.888], 'sd': [0.114, 0.096, 0.031]},
    'n_seeds': 3, 'n_states': 53,
}
# Budget routing, 10 seeds, 1000 of 31,345 decision states
RT = {
    'mean': {'rho': [0.576, 0.765, 0.741], 'sd': [0.024, 0.012, 0.020],
             'prec': {5: 0.374, 10: 0.478, 20: 0.626}},
    'max':  {'rho': [0.536, 0.693, 0.667], 'sd': [0.020, 0.015, 0.018]},
    'n_seeds': 10, 'n_states': 1000,
}
FLC, RTC, INK, GREY = '#1B5FA8', '#B4600F', '#122236', '#C9D2DC'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 4.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.1, 1.1, 1.0], wspace=0.32)

    # (a) what is compared
    ax = fig.add_subplot(gs[0, 0]); ax.axis('off')
    ax.add_patch(plt.Rectangle((0.02, 0.60), 0.44, 0.30, fc='#E9F0F8', ec=FLC, lw=1.6))
    ax.text(0.24, 0.845, 'CCE score', ha='center', fontsize=11.5, weight='bold', color=FLC)
    ax.text(0.24, 0.71, 'roll out the policy\ntry each action\ncompare outcomes',
            ha='center', fontsize=9, color=INK)
    ax.add_patch(plt.Rectangle((0.54, 0.60), 0.44, 0.30, fc='#FBF0E4', ec=RTC, lw=1.6))
    ax.text(0.76, 0.845, 'exact oracle', ha='center', fontsize=11.5, weight='bold', color=RTC)
    ax.text(0.76, 0.71, 'solve the whole MDP\nQ*(s,a*) - Q*(s,a)\naveraged over actions',
            ha='center', fontsize=9, color=INK)
    ax.annotate('', xy=(0.54, 0.75), xytext=(0.46, 0.75),
                arrowprops=dict(arrowstyle='<->', lw=2, color=INK))
    ax.text(0.5, 0.53, 'do they RANK states the same way?', ha='center',
            fontsize=10.5, weight='bold', color=INK)
    ax.text(0.02, 0.42, 'CCE never sees:', fontsize=9.5, weight='bold', color=INK,
            va='top')
    ax.text(0.05, 0.345, '· the transition table\n· the reward function\n· Q*',
            fontsize=9.5, color='#48586B', va='top', linespacing=1.5)
    ax.text(0.02, 0.15, 'so agreement is not circular —\na cheap rollout estimate recovers\n'
                        'something that otherwise needs\nthe MDP solved exactly.',
            fontsize=9.5, color=INK, style='italic', va='top', linespacing=1.5)
    ax.set_title('(a) What Claim 1 compares', fontsize=11.5, loc='left')

    # (b) rho
    ax = fig.add_subplot(gs[0, 1])
    xs = np.arange(3)
    for D, c, lab in ((FL, FLC, f"FrozenLake  ({FL['n_seeds']} seeds, {FL['n_states']} states)"),
                      (RT, RTC, f"routing  ({RT['n_seeds']} seeds, {RT['n_states']} states)")):
        ci = [1.96 * s / np.sqrt(D['n_seeds']) for s in D['mean']['sd']]
        ax.errorbar(xs, D['mean']['rho'], yerr=ci, fmt='o-', color=c, lw=2, ms=8,
                    capsize=4, label=lab)
    ax.axhline(0, color='k', lw=.8)
    ax.text(0.02, 0.06, 'no relationship', fontsize=8.5, color='#7C8A9B',
            transform=ax.transAxes)
    ax.set_xticks(xs); ax.set_xticklabels(STAGES, fontsize=9.5)
    ax.set_ylabel(r'Spearman $\rho$  (CCE vs oracle)')
    ax.set_ylim(0, 1.0); ax.grid(alpha=.25); ax.legend(fontsize=8, loc='lower right')
    ax.set_title('(b) Agreement rises with training', fontsize=11.5, loc='left')

    # (c) precision@k
    ax = fig.add_subplot(gs[0, 2])
    ks = [5, 10, 20]; w = 0.34
    ax.bar(np.arange(3) - w/2, [FL['mean']['prec'][k] * 100 for k in ks], w,
           color=FLC, label='FrozenLake')
    ax.bar(np.arange(3) + w/2, [RT['mean']['prec'][k] * 100 for k in ks], w,
           color=RTC, label='routing')
    ax.plot(np.arange(3), ks, 'k--', marker='_', ms=22, lw=1.5, label='random chance')
    for i, k in enumerate(ks):
        ax.text(i - w/2, FL['mean']['prec'][k]*100 + 1.5,
                f"{FL['mean']['prec'][k]/(k/100):.1f}x", ha='center', fontsize=8.5, color=FLC)
        ax.text(i + w/2, RT['mean']['prec'][k]*100 + 1.5,
                f"{RT['mean']['prec'][k]/(k/100):.1f}x", ha='center', fontsize=8.5, color=RTC)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels([f'top {k}%' for k in ks], fontsize=9.5)
    ax.set_ylabel('% truly in the oracle top-k')
    ax.set_title('(c) precision@k, trained policy', fontsize=11.5, loc='left')
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis='y')

    # (d) robustness
    ax = fig.add_subplot(gs[0, 3])
    for off, D, c, lab in ((-0.17, FL, FLC, 'FrozenLake'), (0.17, RT, RTC, 'routing')):
        ax.bar(xs + off - 0.08, D['max']['rho'], 0.16, color=GREY,
               label='max (shipped)' if off < 0 else None)
        ax.bar(xs + off + 0.08, D['mean']['rho'], 0.16, color=c,
               label=f'{lab} (mean)')
    ax.set_xticks(xs); ax.set_xticklabels(STAGES, fontsize=9.5)
    ax.set_ylabel(r'Spearman $\rho$'); ax.set_ylim(0, 1.0)
    ax.set_title('(d) Robust to the aggregation fix', fontsize=11.5, loc='left')
    ax.legend(fontsize=7.5, loc='upper left'); ax.grid(alpha=.25, axis='y')

    fig.suptitle('CLAIM 1 — a rollout-only score recovers the exact oracle\'s ranking, '
                 'in two unrelated domains', fontsize=13, y=1.03)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / 'fig_claim1_summary.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    print(f"wrote {p}")


if __name__ == '__main__':
    main()
