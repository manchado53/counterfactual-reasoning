"""Summarize experiment results: print table + generate learning curve plots.

Usage:
    python summarize_experiment.py experiments/metric_sweep_2026-02-24.json
    python summarize_experiment.py experiments/uniform_dqn_*.json experiments/metric_sweep_*.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_metrics_log(path):
    """Parse a metrics.log file into a list of row dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            # Header line (contains 'episode')
            if 'episode' in line and 'win_rate' in line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                rows.append({
                    'episode': int(parts[0]),
                    'epsilon': float(parts[1]),
                    'win_rate': float(parts[2].rstrip('%')),
                    'avg_allies': float(parts[3]),
                    'avg_return': float(parts[4]),
                    'avg_length': float(parts[5]),
                })
            except (ValueError, IndexError):
                continue
    return rows


def config_key(overrides):
    """Create a hashable key from config overrides, excluding seed."""
    key_parts = []
    for k in sorted(overrides.keys()):
        if k == 'seed':
            continue
        key_parts.append(f"{k}={overrides[k]}")
    return tuple(key_parts)


def config_label(overrides):
    """Create a human-readable label for a config group."""
    algo = overrides.get('algorithm', 'dqn')
    metric = overrides.get('consequence_metric', '')
    mu = overrides.get('mu', '')

    if algo == 'consequence-dqn' and metric:
        label = f"consequence ({metric}"
        if mu != '':
            label += f", mu={mu}"
        label += ")"
    elif algo == 'dqn':
        label = 'DQN + PER'
    elif algo == 'dqn-uniform':
        label = 'DQN (uniform)'
    else:
        label = algo

    return label


def load_manifests(manifest_paths):
    """Load and merge multiple manifest files."""
    all_runs = {}  # job_id -> overrides
    for path in manifest_paths:
        with open(path) as f:
            manifest = json.load(f)
        all_runs.update(manifest)
    return all_runs


def summarize(manifest_paths):
    all_runs = load_manifests(manifest_paths)

    # Group by config (excluding seed) and scenario
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

    # Print summary header
    print(f"\nRuns: {completed} completed, {missing} missing/incomplete")
    print(f"{'':->80}")

    # Print summary table
    print(f"\n{'scenario':<12} {'algorithm':<22} {'win_rate':>20} {'avg_return':>20}")
    print(f"{'':->12} {'':->22} {'':->20} {'':->20}")

    for (key, scenario), run_list in sorted(groups.items()):
        # Get final win_rate and avg_return for each seed
        final_win_rates = []
        final_returns = []
        for overrides, rows in run_list:
            final_win_rates.append(rows[-1]['win_rate'])
            final_returns.append(rows[-1]['avg_return'])

        mean_wr = np.mean(final_win_rates)
        std_wr = np.std(final_win_rates)
        mean_ret = np.mean(final_returns)
        std_ret = np.std(final_returns)

        label = config_label(run_list[0][0])
        wr_str = f"{mean_wr:.1f} +/- {std_wr:.1f}%"
        ret_str = f"{mean_ret:.2f} +/- {std_ret:.2f}"

        print(f"{scenario:<12} {label:<22} {wr_str:>20} {ret_str:>20}")

    # Generate learning curve plots
    scenarios = sorted(set(s for (_, s) in groups.keys()))
    for scenario in scenarios:
        plot_learning_curves(groups, scenario, manifest_paths)


def plot_learning_curves(groups, scenario, manifest_paths):
    """Generate learning curve plot for a single scenario."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    colors = plt.cm.tab10.colors
    color_idx = 0

    for (key, scen), run_list in sorted(groups.items()):
        if scen != scenario:
            continue

        label = config_label(run_list[0][0])
        color = colors[color_idx % len(colors)]
        color_idx += 1

        # Collect timeseries across seeds
        all_episodes = []
        all_win_rates = []
        all_returns = []

        for overrides, rows in run_list:
            episodes = [r['episode'] for r in rows]
            win_rates = [r['win_rate'] for r in rows]
            returns = [r['avg_return'] for r in rows]
            all_episodes.append(episodes)
            all_win_rates.append(win_rates)
            all_returns.append(returns)

        # Find common episode range (truncate to shortest)
        min_len = min(len(ep) for ep in all_episodes)
        episodes = all_episodes[0][:min_len]
        wr_array = np.array([wr[:min_len] for wr in all_win_rates])
        ret_array = np.array([ret[:min_len] for ret in all_returns])

        # Win rate plot
        ax = axes[0]
        mean_wr = wr_array.mean(axis=0)
        std_wr = wr_array.std(axis=0)
        ax.plot(episodes, mean_wr, color=color, label=label, linewidth=1.5)
        ax.fill_between(episodes, mean_wr - std_wr, mean_wr + std_wr,
                        color=color, alpha=0.15)

        # Return plot
        ax = axes[1]
        mean_ret = ret_array.mean(axis=0)
        std_ret = ret_array.std(axis=0)
        ax.plot(episodes, mean_ret, color=color, label=label, linewidth=1.5)
        ax.fill_between(episodes, mean_ret - std_ret, mean_ret + std_ret,
                        color=color, alpha=0.15)

    axes[0].set_ylabel('Win Rate (%)')
    axes[0].set_title(f'Learning Curves — {scenario}')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Avg Return')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save next to the first manifest file
    manifest_dir = os.path.dirname(os.path.abspath(manifest_paths[0]))
    base_name = os.path.splitext(os.path.basename(manifest_paths[0]))[0]
    save_path = os.path.join(manifest_dir, f"{base_name}_{scenario}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize experiment results')
    parser.add_argument('manifests', nargs='+', help='Manifest JSON files')
    args = parser.parse_args()

    summarize(args.manifests)
