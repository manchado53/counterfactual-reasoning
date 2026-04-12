"""Bootstrap sample efficiency analysis for SMAX experiments.

Reads completed training runs from metrics.log files and uses bootstrapping
to compute P(consequence-DQN > baseline) at each training checkpoint.
Produces one 3-panel figure per metric (win_rate, avg_return by default).

Usage:
    python bootstrap_analysis.py experiments/full_algorithm_comparison_2026-04-07.json
    python bootstrap_analysis.py experiments/my_exp.json --scenario 3m --focus-checkpoint 10000
    python bootstrap_analysis.py experiments/my_exp.json --metrics win_rate
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Reuse parsing utilities from summarize_experiment.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from summarize_experiment import parse_metrics_log, load_manifests, config_label, config_key

# ---------------------------------------------------------------------------
# Bootstrap core
# ---------------------------------------------------------------------------

def bootstrap_prob_a_beats_b(a_data, b_data, n_bootstrap, rng):
    """P(mean of resample from A > mean of resample from B).

    Vectorized: draws all resamples in two (n_bootstrap, n_seeds) arrays,
    computes means along axis=1, then compares — no Python loop.

    Args:
        a_data: array of shape (n_seeds,)
        b_data: array of shape (n_seeds,)
        n_bootstrap: number of resampling iterations
        rng: numpy Generator

    Returns:
        Scalar float in [0, 1]
    """
    n_a, n_b = len(a_data), len(b_data)
    a_idx = rng.integers(0, n_a, size=(n_bootstrap, n_a))
    b_idx = rng.integers(0, n_b, size=(n_bootstrap, n_b))
    a_means = a_data[a_idx].mean(axis=1)
    b_means = b_data[b_idx].mean(axis=1)
    return (a_means > b_means).mean()


def bootstrap_ci_on_prob(a_data, b_data, n_outer, n_bootstrap, rng):
    """90% CI on the P(A>B) estimate via an outer bootstrap over seeds.

    Vectorized outer loop: resamples all n_outer seed-sets at once,
    then calls the vectorized inner bootstrap for each.

    Args:
        a_data: array of shape (n_seeds,)
        b_data: array of shape (n_seeds,)
        n_outer: outer resampling iterations (controls CI width)
        n_bootstrap: inner iterations per outer step
        rng: numpy Generator

    Returns:
        (lo, hi) tuple — 90% CI bounds
    """
    n_a, n_b = len(a_data), len(b_data)
    # Draw all outer resamples at once: (n_outer, n_seeds)
    a_outer = a_data[rng.integers(0, n_a, size=(n_outer, n_a))]
    b_outer = b_data[rng.integers(0, n_b, size=(n_outer, n_b))]
    probs = np.array([
        bootstrap_prob_a_beats_b(a_outer[k], b_outer[k], n_bootstrap, rng)
        for k in range(n_outer)
    ])
    return float(np.percentile(probs, 5)), float(np.percentile(probs, 95))


def get_bootstrap_dist(data, n_bootstrap, rng):
    """Distribution of bootstrap means (for histogram in Panel 2)."""
    n = len(data)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    return data[idx].mean(axis=1)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_groups(manifest_paths, scenario_filter):
    """Load runs from manifests and group by (label, scenario).

    Returns:
        groups: dict mapping (label, scenario) -> list of row-lists (one per seed)
        missing: number of runs with no metrics.log found
    """
    all_runs = load_manifests(manifest_paths)

    # runs/ is relative to the shared/ directory (parent of evals/)
    shared_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = os.path.join(shared_dir, 'runs')

    from collections import defaultdict
    groups = defaultdict(list)
    missing = 0

    for job_id, overrides in all_runs.items():
        scenario = overrides.get('scenario', 'unknown')
        if scenario_filter and scenario != scenario_filter:
            continue
        metrics_path = os.path.join(runs_dir, job_id, 'metrics.log')
        if not os.path.exists(metrics_path):
            print(f"  WARNING: missing {metrics_path}")
            missing += 1
            continue
        rows = parse_metrics_log(metrics_path)
        if not rows:
            print(f"  WARNING: empty metrics.log for job {job_id}")
            missing += 1
            continue
        label = config_label(overrides)
        groups[(label, scenario)].append(rows)

    return groups, missing


def extract_metric_series(rows_list, metric):
    """Extract (episodes, values_per_seed) aligned to common checkpoints.

    Args:
        rows_list: list of row-lists (one per seed)
        metric: 'win_rate' or 'avg_return'

    Returns:
        episodes: 1-D array of episode numbers (common across seeds)
        values: 2-D array of shape (n_seeds, n_checkpoints)
    """
    all_episodes = [[r['episode'] for r in rows] for rows in rows_list]
    all_values   = [[r[metric]   for r in rows] for rows in rows_list]

    min_len = min(len(e) for e in all_episodes)
    episodes = np.array(all_episodes[0][:min_len])
    values   = np.array([v[:min_len] for v in all_values])
    return episodes, values

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    'win_rate':   'Win rate (%)',
    'avg_return': 'Avg return',
}

COLORS = plt.cm.tab10.colors


def make_figure(groups, scenario, metric, focus_checkpoint,
                n_bootstrap, n_outer, stride, rng, manifest_paths, treatment_labels):
    """Produce and save one 4-panel bootstrap figure for a given metric.

    Layout:
        Panel 1 (top-left)  — Learning curves, all seeds
        Panel 2 (top-right) — Bootstrap distributions at focus checkpoint
        Panel 3 (bot-left)  — P(additive consequence > baseline) over training
        Panel 4 (bot-right) — P(multiplicative consequence > baseline) over training
    """

    scene_labels = sorted(set(lbl for (lbl, scen) in groups if scen == scenario))
    baseline_labels = [l for l in scene_labels if 'consequence' not in l.lower()]

    if not baseline_labels:
        print(f"  No baseline runs found for scenario '{scenario}' — skipping.")
        return

    # Warn if N is low
    for label in scene_labels:
        n = len(groups[(label, scenario)])
        if n < 10:
            print(f"  WARNING: Only N={n} seeds for '{label}' — "
                  f"bootstrap CI bands will be wide. "
                  f"Consider running more seeds for tighter estimates.")

    color_map = {lbl: COLORS[i % len(COLORS)] for i, lbl in enumerate(scene_labels)}
    metric_label = METRIC_LABELS.get(metric, metric)

    # ── Figure layout: 2×2 grid ────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0f0f0f')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                           top=0.90, bottom=0.07, left=0.06, right=0.97)
    ax_curves  = fig.add_subplot(gs[0, 0])
    ax_dist    = fig.add_subplot(gs[0, 1])
    ax_prob_p3 = fig.add_subplot(gs[1, 0])   # additive
    ax_prob_p4 = fig.add_subplot(gs[1, 1])   # multiplicative

    for ax in [ax_curves, ax_dist, ax_prob_p3, ax_prob_p4]:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    n_seeds_str = ', '.join(f"N={len(groups[(l, scenario)])}" for l in scene_labels)
    fig.suptitle(
        f"Bootstrap Sample Efficiency — SMAX {scenario} | {metric_label}  ({n_seeds_str})",
        color='white', fontsize=13, fontweight='bold'
    )

    # ── Panel 1: Learning curves ───────────────────────────────────────────
    print(f"  [Panel 1] Building learning curves...", flush=True)
    focus_ep_k = focus_checkpoint / 1000

    for label in scene_labels:
        rows_list = groups[(label, scenario)]
        episodes, values = extract_metric_series(rows_list, metric)
        ep_k = episodes / 1000
        color = color_map[label]
        for seed_vals in values:
            ax_curves.plot(ep_k, seed_vals, color=color, alpha=0.18, linewidth=1)
        ax_curves.plot(ep_k, values.mean(axis=0), color=color, linewidth=2.5, label=label)

    ax_curves.axvline(x=focus_ep_k, color='yellow', linestyle='--', alpha=0.6, linewidth=1)
    ax_curves.text(focus_ep_k + 0.3, 2, f'{int(focus_checkpoint/1000)}k\n(zoom →)',
                   color='yellow', fontsize=8, va='bottom')
    ax_curves.set_xlabel('Training episodes (thousands)')
    ax_curves.set_ylabel(metric_label)
    ax_curves.set_title('Panel 1 — Learning curves (all seeds)')
    ax_curves.legend(facecolor='#111', labelcolor='white', framealpha=0.8, fontsize=7)
    if metric == 'win_rate':
        ax_curves.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))

    # ── Panel 2: Bootstrap distributions at focus checkpoint ───────────────
    print(f"  [Panel 2] Bootstrap distributions at ep {focus_checkpoint:,} "
          f"({n_bootstrap:,} iterations)...", flush=True)

    def values_at_checkpoint(label):
        rows_list = groups[(label, scenario)]
        episodes, values = extract_metric_series(rows_list, metric)
        idx = np.argmin(np.abs(episodes - focus_checkpoint))
        return values[:, idx], episodes[idx]

    boot_dists = {}
    actual_ep = None
    for label in scene_labels:
        vals, ep = values_at_checkpoint(label)
        boot_dists[label] = (vals, get_bootstrap_dist(vals, n_bootstrap, rng))
        actual_ep = ep

    bins = np.linspace(
        min(d.min() for _, d in boot_dists.values()),
        max(d.max() for _, d in boot_dists.values()),
        55
    )
    for label in scene_labels:
        vals, dist = boot_dists[label]
        ax_dist.hist(dist, bins=bins, color=color_map[label], alpha=0.65,
                     label=label, density=True)

    # Annotate P for all treatment variants vs each baseline
    annotation_lines = []
    for t_label in treatment_labels:
        mixing = 'mult' if 'mult' in t_label else 'add'
        _, t_dist = boot_dists[t_label]
        for b_label in baseline_labels:
            _, b_dist = boot_dists[b_label]
            p = (t_dist > b_dist).mean()
            short_b = b_label.split('(')[0].strip()
            annotation_lines.append(f'P({mixing} > {short_b}) = {p:.2f}')

    ax_dist.text(0.97, 0.97, '\n'.join(annotation_lines),
                 transform=ax_dist.transAxes, ha='right', va='top',
                 color='white', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#222', edgecolor='#555'))
    ax_dist.set_xlabel(f'Bootstrap mean {metric_label} @ ep {actual_ep:,}')
    ax_dist.set_ylabel('Density')
    ax_dist.set_title(f'Panel 2 — Bootstrap distributions at {int(focus_checkpoint/1000)}k episodes')
    ax_dist.legend(facecolor='#111', labelcolor='white', framealpha=0.8, fontsize=7)
    if metric == 'win_rate':
        ax_dist.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))

    # ── Panels 3 & 4: P(treatment > baseline) — one panel per consequence variant ──
    panel_axes = {t_label: ax for t_label, ax in zip(treatment_labels,
                                                      [ax_prob_p3, ax_prob_p4])}
    panel_nums = {t_label: n for t_label, n in zip(treatment_labels, [3, 4])}

    line_colors = [COLORS[(i + 2) % len(COLORS)] for i in range(len(baseline_labels))]

    for t_label, ax_prob in panel_axes.items():
        mixing_name = 'Multiplicative' if 'mult' in t_label else 'Additive'
        panel_num = panel_nums[t_label]
        print(f"  [Panel {panel_num}] P({mixing_name} consequence > baseline)...", flush=True)

        ax_prob.axhline(y=0.5, color='white', linestyle='--', alpha=0.4, linewidth=1,
                        label='Chance (P = 0.50)')
        ax_prob.axhline(y=0.8, color='yellow', linestyle=':', alpha=0.3, linewidth=1)
        ax_prob.text(0.01, 0.82, 'P=0.80', transform=ax_prob.transAxes,
                     color='yellow', fontsize=7)

        t_rows = groups[(t_label, scenario)]
        t_episodes, t_values = extract_metric_series(t_rows, metric)

        for b_label, lc in zip(baseline_labels, line_colors):
            b_rows = groups[(b_label, scenario)]
            b_episodes, b_values = extract_metric_series(b_rows, metric)

            common_eps = np.intersect1d(t_episodes, b_episodes)[::stride]
            t_vals_aligned = t_values[:, np.isin(t_episodes, common_eps)]
            b_vals_aligned = b_values[:, np.isin(b_episodes, common_eps)]

            probs, ci_lo, ci_hi = [], [], []
            n_checkpoints = len(common_eps)
            print(f"    vs {b_label} — {n_checkpoints} checkpoints...", flush=True)

            for i in range(n_checkpoints):
                p = bootstrap_prob_a_beats_b(t_vals_aligned[:, i], b_vals_aligned[:, i],
                                             n_bootstrap, rng)
                lo, hi = bootstrap_ci_on_prob(t_vals_aligned[:, i], b_vals_aligned[:, i],
                                              n_outer, n_bootstrap=500, rng=rng)
                probs.append(p)
                ci_lo.append(lo)
                ci_hi.append(hi)
                if (i + 1) % max(1, n_checkpoints // 10) == 0 or (i + 1) == n_checkpoints:
                    pct = (i + 1) / n_checkpoints * 100
                    print(f"      [{pct:5.1f}%] ep {common_eps[i]:6,}  P={p:.3f}  "
                          f"90% CI=[{lo:.2f}, {hi:.2f}]", flush=True)

            probs = np.array(probs)
            ci_lo = np.array(ci_lo)
            ci_hi = np.array(ci_hi)
            ep_k  = common_eps / 1000
            short_b = b_label.split('(')[0].strip()

            ax_prob.fill_between(ep_k, ci_lo, ci_hi, color=lc, alpha=0.2)
            ax_prob.plot(ep_k, probs, color=lc, linewidth=2, marker='o', markersize=4,
                         label=f'P(consequence > {short_b})')

            focus_idx = np.argmin(np.abs(common_eps - focus_checkpoint))
            ax_prob.annotate(
                f'CI @ {int(focus_checkpoint/1000)}k:\n'
                f'[{ci_lo[focus_idx]:.2f}, {ci_hi[focus_idx]:.2f}]',
                xy=(ep_k[focus_idx], probs[focus_idx]),
                xytext=(ep_k[focus_idx] + max(ep_k) * 0.05, probs[focus_idx] - 0.13),
                color='white', fontsize=7,
                arrowprops=dict(arrowstyle='->', color='white', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#222', edgecolor='#888')
            )

        ax_prob.set_xlabel('Training episodes (thousands)')
        ax_prob.set_ylabel('P(consequence wins)')
        ax_prob.set_title(
            f'Panel {panel_num} — {mixing_name}: P(consequence > baseline)\n'
            f'Shaded = 90% CI (outer bootstrap, N={len(t_rows)} seeds)'
        )
        ax_prob.legend(facecolor='#111', labelcolor='white', framealpha=0.8, fontsize=8)
        ax_prob.set_ylim(0.0, 1.05)

    # ── Save ──────────────────────────────────────────────────────────────
    manifest_dir = os.path.dirname(os.path.abspath(manifest_paths[0]))
    base_name = os.path.splitext(os.path.basename(manifest_paths[0]))[0]
    save_path = os.path.join(manifest_dir, f'{base_name}_{scenario}_{metric}_bootstrap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap sample efficiency analysis for SMAX experiments.'
    )
    parser.add_argument('manifests', nargs='+', help='Manifest JSON files')
    parser.add_argument('--focus-checkpoint', type=int, default=10000,
                        help='Episode checkpoint to zoom in on in Panel 2 (default: 10000)')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Number of inner bootstrap iterations (default: 10000)')
    parser.add_argument('--n-outer', type=int, default=200,
                        help='Number of outer bootstrap iterations for CI (default: 200)')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Filter to a single scenario, e.g. "3m" (default: all)')
    parser.add_argument('--metrics', nargs='+', default=['win_rate', 'avg_return'],
                        choices=['win_rate', 'avg_return'],
                        help='Metrics to analyze (default: win_rate avg_return)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Use every Nth checkpoint for Panel 3 (default: 1 = all checkpoints). '
                             'Higher = fewer checkpoints but faster.')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for reproducibility (default: 42)')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"\nLoading runs from {len(args.manifests)} manifest(s)...")
    groups, missing = load_groups(args.manifests, args.scenario)
    if missing:
        print(f"  {missing} run(s) missing or incomplete.")

    scenarios = sorted(set(scen for (_, scen) in groups.keys()))
    if not scenarios:
        print("No completed runs found. Check manifest paths and runs/ directory.")
        return

    print(f"Scenarios: {scenarios}")
    print(f"Focus checkpoint: episode {args.focus_checkpoint:,}")
    print(f"Metrics: {args.metrics}")
    print(f"Inner bootstrap iterations: {args.n_bootstrap:,}")
    print(f"Outer bootstrap iterations (CI): {args.n_outer}\n")

    for scenario in scenarios:
        scene_labels = sorted(set(lbl for (lbl, scen) in groups.keys() if scen == scenario))
        treatment_labels = [l for l in scene_labels if 'consequence' in l.lower()]
        if not treatment_labels:
            print(f"No consequence-DQN runs found for scenario '{scenario}' — skipping.")
            continue
        for metric in args.metrics:
            print(f"--- {scenario} | {metric} ---")
            make_figure(
                groups=groups,
                scenario=scenario,
                metric=metric,
                focus_checkpoint=args.focus_checkpoint,
                n_bootstrap=args.n_bootstrap,
                n_outer=args.n_outer,
                stride=args.stride,
                rng=rng,
                manifest_paths=args.manifests,
                treatment_labels=treatment_labels,
            )


if __name__ == '__main__':
    main()
