"""Generate all Claim 2 figures from computed metrics."""

import os
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ALG_ORDER = ['DQN-Uniform', 'DQN+PER', 'DQN+CCE-only', 'CCE+TD (add)', 'CCE+TD (mul)']
COLORS = {
    'DQN-Uniform':   '#7f7f7f',
    'DQN+PER':       '#2196F3',
    'DQN+CCE-only':  '#FF9800',
    'CCE+TD (add)':  '#4CAF50',
    'CCE+TD (mul)':  '#E91E63',
}


def _alg_order(algs):
    return [a for a in ALG_ORDER if a in algs] + [a for a in algs if a not in ALG_ORDER]


def fig1_iqm_curves(
    iqm_by_env: Dict[str, Dict],
    eval_steps_by_env: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    out_path: str,
):
    """Fig 1: IQM learning curves, one panel per environment."""
    envs = list(iqm_by_env.keys())
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 4), sharey=False)
    if len(envs) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, envs):
        iqm_data = iqm_by_env[env_name]
        steps = eval_steps_by_env[env_name]
        threshold = thresholds.get(env_name)

        for alg in _alg_order(iqm_data):
            iqm_vals, ci_lo, ci_hi = iqm_data[alg]
            x = steps[:len(iqm_vals)]
            color = COLORS.get(alg, 'black')
            ax.plot(x, iqm_vals, label=alg, color=color, linewidth=1.5)
            ax.fill_between(x, ci_lo, ci_hi, alpha=0.15, color=color)

        if threshold is not None:
            ax.axhline(threshold, color='red', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_title(env_name)
        ax.set_xlabel('Steps')
        ax.set_ylabel('IQM Win Rate')
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(ALG_ORDER),
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def fig2_final_iqm(
    final_iqm_by_env: Dict[str, Dict],
    out_path: str,
):
    """Fig 2: Final IQM bar chart per environment."""
    envs = list(final_iqm_by_env.keys())
    fig, axes = plt.subplots(1, len(envs), figsize=(4 * len(envs), 4))
    if len(envs) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, envs):
        data = final_iqm_by_env[env_name]
        algs = _alg_order(data)
        points = [data[a][0] for a in algs]
        err_lo = [data[a][0] - data[a][1] for a in algs]
        err_hi = [data[a][2] - data[a][0] for a in algs]
        colors = [COLORS.get(a, 'gray') for a in algs]
        y = np.arange(len(algs))

        ax.barh(y, points, xerr=[err_lo, err_hi], color=colors, alpha=0.8, capsize=4)
        if 'DQN+PER' in data:
            ax.axvline(data['DQN+PER'][0], color='blue', linestyle=':', linewidth=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels(algs)
        ax.set_xlabel('Final IQM Win Rate')
        ax.set_title(env_name)
        ax.set_xlim(0, 1.05)
        ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def fig3_steps_to_threshold(
    thresh_by_env: Dict[str, Dict],
    out_path: str,
):
    """Fig 3: Steps-to-threshold lollipop chart."""
    envs = list(thresh_by_env.keys())
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 4))
    if len(envs) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, envs):
        data = thresh_by_env[env_name]
        algs = _alg_order(data)
        medians = np.array([data[a][0] for a in algs])
        iqrs = np.array([data[a][1] for a in algs])
        y = np.arange(len(algs))
        colors = [COLORS.get(a, 'gray') for a in algs]

        finite_mask = np.isfinite(medians)
        for i, (alg, med, iqr, color) in enumerate(zip(algs, medians, iqrs, colors)):
            if np.isfinite(med):
                ax.plot(med, i, 'o', color=color, markersize=8)
                ax.hlines(i, med - iqr / 2, med + iqr / 2, colors=color, linewidth=3, alpha=0.6)
                ax.annotate(f'{med/1000:.0f}k', (med, i), textcoords='offset points',
                            xytext=(5, 3), fontsize=8)
            else:
                ax.text(0.95, i / len(algs), '∞', transform=ax.transAxes,
                        ha='right', va='center', color=color)

        ax.set_yticks(y)
        ax.set_yticklabels(algs)
        ax.set_xlabel('Steps to Threshold')
        ax.set_title(env_name)
        ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def fig4_prob_improvement(
    prob_by_env: Dict[str, Dict],
    out_path: str,
):
    """Fig 4: P(algorithm > DQN+PER) bar chart."""
    envs = list(prob_by_env.keys())
    fig, axes = plt.subplots(1, len(envs), figsize=(4 * len(envs), 3))
    if len(envs) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, envs):
        data = prob_by_env[env_name]
        algs = _alg_order(data)
        points = [data[a][0] for a in algs]
        err_lo = [data[a][0] - data[a][1] for a in algs]
        err_hi = [data[a][2] - data[a][0] for a in algs]
        colors = [COLORS.get(a, 'gray') for a in algs]
        y = np.arange(len(algs))

        ax.barh(y, points, xerr=[err_lo, err_hi], color=colors, alpha=0.8, capsize=4)
        ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels(algs)
        ax.set_xlabel('P(alg > DQN+PER)')
        ax.set_title(env_name)
        ax.set_xlim(0, 1.0)
        ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def fig5a_wallclock_breakdown(
    wallclock_by_env: Dict[str, Dict],
    out_path: str,
):
    """Fig 5a: Stacked bar — training vs. scoring hours per algorithm per environment."""
    envs = list(wallclock_by_env.keys())
    fig, axes = plt.subplots(1, len(envs), figsize=(4 * len(envs), 4))
    if len(envs) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, envs):
        data = wallclock_by_env[env_name]
        algs = _alg_order(data)
        training = [data[a]['training_hours'] for a in algs]
        scoring = [data[a]['scoring_hours'] for a in algs]
        totals = [t + s for t, s in zip(training, scoring)]
        x = np.arange(len(algs))
        ref = totals[algs.index('DQN-Uniform')] if 'DQN-Uniform' in algs else 1.0

        ax.bar(x, training, label='Training', color='#2196F3', alpha=0.8)
        ax.bar(x, scoring, bottom=training, label='Scoring overhead', color='#F44336', alpha=0.8)
        for i, (tot, alg) in enumerate(zip(totals, algs)):
            mult = tot / ref if ref > 0 else 1.0
            ax.text(i, tot + 0.01, f'{mult:.1f}×', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(algs, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('Hours')
        ax.set_title(env_name)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_length_curves(
    length_by_env: Dict[str, Dict],
    eval_steps_by_env: Dict[str, np.ndarray],
    out_path: str,
):
    """Avg episode length IQM curves, one panel per environment.

    Note: FrozenLake lengths include failed episodes (all-episode avg_length).
    """
    envs = list(length_by_env.keys())
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 4), sharey=False)
    if len(envs) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, envs):
        data = length_by_env[env_name]
        steps = eval_steps_by_env[env_name]
        for alg in _alg_order(data):
            iqm_vals, ci_lo, ci_hi = data[alg]
            x = steps[:len(iqm_vals)]
            color = COLORS.get(alg, 'black')
            ax.plot(x, iqm_vals, label=alg, color=color, linewidth=1.5)
            ax.fill_between(x, ci_lo, ci_hi, alpha=0.15, color=color)
        ax.set_title(env_name)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Avg episode length (steps)')
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(ALG_ORDER),
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


_COMP_COLORS = {
    'env':                          '#9E9E9E',
    'action':                       '#BDBDBD',
    'collect':                      '#757575',
    'buffer.add':                   '#FF9800',
    'update':                       '#2196F3',
    'update.q_update':              '#1565C0',
    'eval':                         '#4CAF50',
    'update.scoring.sample':        '#FFCDD2',
    'update.scoring.rollouts':      '#EF9A9A',
    'update.scoring.metrics':       '#E57373',
    'update.scoring.beam':          '#EF5350',
    'update.scoring.stack':         '#F44336',
    'update.scoring.buffer_update': '#C62828',
}
_COMP_ORDER = list(_COMP_COLORS.keys())


def fig5b_wallclock_to_threshold(
    wc_thresh_by_env: Dict[str, Dict],
    out_path: str,
):
    """Fig 5b: Wall-clock hours to threshold, one lollipop panel per environment."""
    envs = list(wc_thresh_by_env.keys())
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 4))
    if len(envs) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, envs):
        data = wc_thresh_by_env[env_name]
        algs = _alg_order(data)
        wc_hours = [data[a][0] for a in algs]
        y = np.arange(len(algs))
        colors = [COLORS.get(a, 'gray') for a in algs]

        ref_h = data.get('DQN+PER', (np.inf,))[0]

        for i, (alg, h, color) in enumerate(zip(algs, wc_hours, colors)):
            if np.isfinite(h):
                ax.plot(h, i, 'o', color=color, markersize=8)
                ax.hlines(i, 0, h, colors=color, linewidth=2, alpha=0.6)
                ax.annotate(f'{h:.2f}h', (h, i), textcoords='offset points',
                            xytext=(5, 3), fontsize=8)
            else:
                ax.text(0.97, (i + 0.5) / len(algs), '∞', transform=ax.transAxes,
                        ha='right', va='center', color=color, fontsize=12)

        if np.isfinite(ref_h):
            ax.axvline(ref_h, color=COLORS.get('DQN+PER', '#2196F3'),
                       linestyle='--', linewidth=1.5, alpha=0.7)

        ax.set_yticks(y)
        ax.set_yticklabels(algs)
        ax.set_xlabel('Wall-clock hours to threshold')
        ax.set_title(env_name)
        ax.set_xlim(left=0)
        ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def fig5c_component_breakdown(
    components_by_env: Dict[str, Dict],
    out_path: str,
):
    """Fig 5c: Per-component timing stacked bar, one panel per environment.

    Renders whatever components exist in each env's timing data.
    Segments ≥5% of an algorithm's total are labelled with their percentage.
    """
    envs = list(components_by_env.keys())
    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, envs):
        data = components_by_env[env_name]  # {alg: {comp: hours}}
        algs = _alg_order(data)

        # Sorted by canonical order, unknowns last
        all_comps = sorted(
            {c for alg_d in data.values() for c in alg_d},
            key=lambda c: _COMP_ORDER.index(c) if c in _COMP_ORDER else len(_COMP_ORDER),
        )

        y = np.arange(len(algs))
        lefts = np.zeros(len(algs))

        for comp in all_comps:
            vals = np.array([data[a].get(comp, 0.0) for a in algs])
            color = _COMP_COLORS.get(comp, '#E0E0E0')
            ax.barh(y, vals, left=lefts, color=color, alpha=0.85, label=comp)

            totals = np.array([sum(data[a].values()) for a in algs])
            for i, (val, tot) in enumerate(zip(vals, totals)):
                if tot > 0 and val / tot >= 0.05:
                    ax.text(lefts[i] + val / 2, i, f'{val/tot:.0%}',
                            ha='center', va='center', fontsize=7,
                            color='white', fontweight='bold')
            lefts += vals

        ax.set_yticks(y)
        ax.set_yticklabels(algs)
        ax.set_xlabel('Hours')
        ax.set_title(env_name)
        ax.legend(loc='lower right', fontsize=7, ncol=2)
        ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_allies_curves(
    allies_by_env: Dict[str, Dict],
    eval_steps_by_env: Dict[str, np.ndarray],
    out_path: str,
):
    """Appendix: Allies alive IQM curves for SMAX environments."""
    envs = list(allies_by_env.keys())
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 4), sharey=False)
    if len(envs) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, envs):
        data = allies_by_env[env_name]
        steps = eval_steps_by_env[env_name]
        for alg in _alg_order(data):
            vals = data[alg]
            color = COLORS.get(alg, 'black')
            ax.plot(steps[:len(vals)], vals, label=alg, color=color, linewidth=1.5)
        ax.set_title(env_name)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Mean Allies Alive')
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(ALG_ORDER),
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_wdl_table(
    wdl_by_alg: Dict[str, Tuple[float, float, float]],
    env_name: str,
    out_path: str,
):
    """Appendix: W/D/L stacked bar at convergence for chess."""
    algs = _alg_order(wdl_by_alg)
    wins = [wdl_by_alg[a][0] for a in algs]
    draws = [wdl_by_alg[a][1] for a in algs]
    losses = [wdl_by_alg[a][2] for a in algs]
    x = np.arange(len(algs))

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(x, wins,   label='Win',  color='#4CAF50', alpha=0.85)
    ax.bar(x, draws,  label='Draw', color='#FF9800', alpha=0.85, bottom=wins)
    ax.bar(x, losses, label='Loss', color='#F44336', alpha=0.85,
           bottom=[w + d for w, d in zip(wins, draws)])
    ax.set_xticks(x)
    ax.set_xticklabels(algs, rotation=15, ha='right')
    ax.set_ylabel('Fraction')
    ax.set_title(f'W/D/L at Convergence — {env_name}')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")
