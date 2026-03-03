"""Quantitative analysis of experiment results: ranking, statistical tests, recommendation.

Usage:
    python analyze_experiment.py experiments/mu_sweep_2026-03-02.json
    python analyze_experiment.py experiments/mu_sweep_2026-03-02.json --last-n 50
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from summarize_experiment import (
    parse_metrics_log,
    config_key,
    config_label,
    load_manifests,
)

try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def last_n_scalars(rows, n):
    """Average win_rate and avg_return over the final n logged rows."""
    tail = rows[-n:]
    win_rates = [r['win_rate'] for r in tail]
    returns = [r['avg_return'] for r in tail]
    return np.mean(win_rates), np.mean(returns)


def save_table_png(ranked, pairwise_results, last_n, save_path, scenario=None):
    """Render ranking table and pairwise p-values as a PNG."""
    n_configs = len(ranked)

    # Build short labels for the pairwise matrix — show only what varies
    all_labels = [stats['label'] for _, stats in ranked]

    def make_short_labels(labels):
        """Extract the parts that differ across labels."""
        import re
        if len(labels) <= 1:
            return labels[:]
        # Extract parenthesized content from each label
        parts_list = []
        for lab in labels:
            m = re.search(r'\((.+)\)', lab)
            parts_list.append(m.group(1).split(', ') if m else [lab])
        # Find which parts vary
        n_parts = min(len(p) for p in parts_list)
        varying = []
        for i in range(n_parts):
            vals = set(p[i] for p in parts_list if i < len(p))
            if len(vals) > 1:
                varying.append(i)
        if not varying:
            # Nothing varies inside parens — use full label
            return labels[:]
        # Build short labels from varying parts only
        short = []
        for parts in parts_list:
            short.append(', '.join(parts[i] for i in varying if i < len(parts)))
        return short

    short_labels = make_short_labels(all_labels)

    # Build ranking rows
    rank_headers = ['#', 'Config', 'Win Rate', 'Avg Return', 'Seeds']
    rank_rows = []
    for i, (key, stats) in enumerate(ranked, 1):
        wr_mean = stats['win_rates'].mean()
        wr_std = stats['win_rates'].std()
        ret_mean = stats['returns'].mean()
        ret_std = stats['returns'].std()
        rank_rows.append([
            str(i),
            stats['label'],
            f"{wr_mean:.1f} \u00b1 {wr_std:.1f}%",
            f"{ret_mean:.2f} \u00b1 {ret_std:.2f}",
            str(stats['n_seeds']),
        ])

    # Build pairwise matrix rows
    has_pairwise = len(pairwise_results) > 0

    # Figure layout: ranking table on top, pairwise matrix below
    n_tables = 2 if has_pairwise else 1
    fig_height = 0.6 + 0.4 * len(rank_rows) + (0.6 + 0.4 * n_configs if has_pairwise else 0) + 0.8
    fig, axes = plt.subplots(n_tables, 1, figsize=(10, fig_height),
                             gridspec_kw={'height_ratios': [len(rank_rows) + 1.5] + ([n_configs + 1.5] if has_pairwise else [])})
    if n_tables == 1:
        axes = [axes]

    # --- Ranking table ---
    ax = axes[0]
    ax.axis('off')
    title_prefix = f'{scenario} — ' if scenario else ''
    ax.set_title(f'{title_prefix}Rankings (by win_rate, last {last_n} checkpoints)', fontsize=13, fontweight='bold', pad=12)

    table = ax.table(
        cellText=rank_rows,
        colLabels=rank_headers,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    # Widen the Config column
    table.auto_set_column_width(col=list(range(len(rank_headers))))

    # Style header
    for j in range(len(rank_headers)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Highlight best row
    for j in range(len(rank_headers)):
        table[1, j].set_facecolor('#d5f5e3')

    # Alternate row colors
    for i in range(2, len(rank_rows) + 1):
        color = '#f9f9f9' if i % 2 == 0 else '#ffffff'
        for j in range(len(rank_headers)):
            table[i, j].set_facecolor(color)

    # --- Pairwise p-value matrix ---
    if has_pairwise:
        ax2 = axes[1]
        ax2.axis('off')
        ax2.set_title('Pairwise Comparisons (Mann-Whitney U, win_rate)', fontsize=13, fontweight='bold', pad=12)

        labels = short_labels
        # Build matrix
        pval_matrix = [['' for _ in range(n_configs)] for _ in range(n_configs)]
        for (i, j), pval in pairwise_results.items():
            pval_str = f"{pval:.4f}"
            pval_matrix[i][j] = pval_str
            pval_matrix[j][i] = pval_str
        for i in range(n_configs):
            pval_matrix[i][i] = '—'

        table2 = ax2.table(
            cellText=pval_matrix,
            rowLabels=labels,
            colLabels=labels,
            cellLoc='center',
            loc='center',
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1, 1.6)
        table2.auto_set_column_width(col=list(range(n_configs)))

        # Style header row and column
        for j in range(n_configs):
            cell = table2[0, j]
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold', fontsize=8)
        for i in range(n_configs):
            cell = table2[i + 1, -1]
            cell.set_facecolor('#34495e')
            cell.set_text_props(color='white', fontweight='bold', fontsize=8)

        # Color-code p-values
        for i in range(n_configs):
            for j in range(n_configs):
                cell = table2[i + 1, j]
                if i == j:
                    cell.set_facecolor('#ecf0f1')
                elif pval_matrix[i][j]:
                    pval = float(pval_matrix[i][j])
                    if pval < 0.05:
                        cell.set_facecolor('#d5f5e3')  # green = significant
                    elif pval < 0.10:
                        cell.set_facecolor('#fdebd0')  # orange = marginal
                    else:
                        cell.set_facecolor('#ffffff')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved {save_path}")


def analyze_scenario(scenario, groups, last_n):
    """Analyze a single scenario. Returns (ranked, pairwise_results)."""
    # Compute per-seed scalars for each config
    config_stats = {}
    for key, run_list in groups.items():
        label = config_label(run_list[0][0])
        seed_win_rates = []
        seed_returns = []
        for overrides, rows in run_list:
            effective_n = min(last_n, len(rows))
            wr, ret = last_n_scalars(rows, effective_n)
            seed_win_rates.append(wr)
            seed_returns.append(ret)
        config_stats[key] = {
            'win_rates': np.array(seed_win_rates),
            'returns': np.array(seed_returns),
            'label': label,
            'n_seeds': len(run_list),
        }

    # Sort by mean win_rate descending
    ranked = sorted(config_stats.items(),
                    key=lambda x: x[1]['win_rates'].mean(), reverse=True)

    # Print ranking table
    print(f"\n--- [{scenario}] Rankings (by win_rate, last {last_n} checkpoints) ---")
    print(f" {'#':>2}  {'config':<35} {'win_rate':>18} {'avg_return':>18}")
    for i, (key, stats) in enumerate(ranked, 1):
        wr_mean = stats['win_rates'].mean()
        wr_std = stats['win_rates'].std()
        ret_mean = stats['returns'].mean()
        ret_std = stats['returns'].std()
        wr_str = f"{wr_mean:.1f} \u00b1 {wr_std:.1f}%"
        ret_str = f"{ret_mean:.2f} \u00b1 {ret_std:.2f}"
        print(f" {i:>2}  {stats['label']:<35} {wr_str:>18} {ret_str:>18}")

    # Pairwise Mann-Whitney U tests
    pairwise_results = {}  # (i, j) -> p-value
    print(f"\n--- [{scenario}] Pairwise Comparisons (Mann-Whitney U, win_rate) ---")
    if not HAS_SCIPY:
        print("  [scipy not installed — skipping statistical tests]")
        print("  Install with: pip install scipy")
    elif len(ranked) < 2:
        print("  Only one config — no comparisons needed.")
    else:
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                key_a, stats_a = ranked[i]
                key_b, stats_b = ranked[j]
                try:
                    stat, pval = mannwhitneyu(
                        stats_a['win_rates'],
                        stats_b['win_rates'],
                        alternative='two-sided',
                    )
                    pairwise_results[(i, j)] = pval
                    sig = " *" if pval < 0.05 else ""
                    print(f"  {stats_a['label']} vs {stats_b['label']}:  p={pval:.4f}{sig}")
                except ValueError as e:
                    print(f"  {stats_a['label']} vs {stats_b['label']}:  test failed ({e})")

    # Recommendation
    best_key, best_stats = ranked[0]
    print(f"\n[{scenario}] Best: {best_stats['label']}  "
          f"(win_rate {best_stats['win_rates'].mean():.1f}%, "
          f"n={best_stats['n_seeds']} seeds)")

    if HAS_SCIPY and len(ranked) >= 2:
        runner_key, runner_stats = ranked[1]
        try:
            _, pval = mannwhitneyu(
                best_stats['win_rates'],
                runner_stats['win_rates'],
                alternative='greater',
            )
            if pval < 0.05:
                print(f"  Statistically significant vs runner-up "
                      f"({runner_stats['label']}, p={pval:.4f})")
            else:
                print(f"  NOT statistically significant vs runner-up "
                      f"({runner_stats['label']}, p={pval:.4f})")
                print(f"  (With only {best_stats['n_seeds']} seeds per config, "
                      f"significance is hard to achieve)")
        except ValueError:
            pass

    return ranked, pairwise_results


def analyze(manifest_paths, last_n):
    all_runs = load_manifests(manifest_paths)

    # Group runs by (config_key, scenario)
    groups = defaultdict(list)  # (config_key, scenario) -> [(overrides, rows)]
    completed = 0
    missing = 0

    for job_id, overrides in all_runs.items():
        metrics_path = os.path.join('runs', job_id, 'metrics.log')
        if not os.path.exists(metrics_path):
            missing += 1
            continue
        rows = parse_metrics_log(metrics_path)
        if not rows:
            missing += 1
            continue
        completed += 1
        scenario = overrides.get('scenario', 'unknown')
        key = config_key(overrides)
        groups[(key, scenario)].append((overrides, rows))

    print(f"\n=== Experiment Analysis ===")
    print(f"Runs: {completed} completed, {missing} missing")

    if not groups:
        print("No data to analyze.")
        return

    # Identify scenarios
    scenarios = sorted(set(scen for (_, scen) in groups.keys()))

    manifest_dir = os.path.dirname(os.path.abspath(manifest_paths[0]))
    base_name = os.path.splitext(os.path.basename(manifest_paths[0]))[0]

    for scenario in scenarios:
        # Collect groups for this scenario (keyed by config only)
        scenario_groups = {}
        for (key, scen), run_list in groups.items():
            if scen == scenario:
                scenario_groups[key] = run_list

        ranked, pairwise_results = analyze_scenario(scenario, scenario_groups, last_n)

        # Save per-scenario PNG
        suffix = f"_{scenario}" if len(scenarios) > 1 else ""
        save_path = os.path.join(manifest_dir, f"{base_name}_analysis{suffix}.png")
        save_table_png(ranked, pairwise_results, last_n, save_path, scenario)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Quantitative analysis of experiment results')
    parser.add_argument('manifests', nargs='+', help='Manifest JSON files')
    parser.add_argument('--last-n', type=int, default=50,
                        help='Number of final checkpoints to average (default: 50)')
    args = parser.parse_args()

    analyze(args.manifests, args.last_n)
