"""Analyze training time breakdown from SLURM log files.

Usage:
    python analyze_timing.py logs/train_227863.out
    python analyze_timing.py logs/train_227863.out logs/train_227864.out
    python analyze_timing.py logs/train_*.out
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(path):
    """Parse a SLURM log file for scoring timing and training progress."""
    scoring_passes = []
    wall_time = None

    with open(path) as f:
        for line in f:
            # Parse [Scoring] lines
            m = re.search(
                r'\[Scoring\]\s+sample=([\d.]+)s\s+beam=([\d.]+)s\s+'
                r'stack=([\d.]+)s\s+rollouts=([\d.]+)s\s+metrics=([\d.]+)s\s+'
                r'update=([\d.]+)s\s+total=([\d.]+)s\s+\(B=(\d+)\)',
                line,
            )
            if m:
                scoring_passes.append({
                    'sample': float(m.group(1)),
                    'beam': float(m.group(2)),
                    'stack': float(m.group(3)),
                    'rollouts': float(m.group(4)),
                    'metrics': float(m.group(5)),
                    'update': float(m.group(6)),
                    'total': float(m.group(7)),
                    'batch': int(m.group(8)),
                })

            # Parse wall time from tqdm progress bar (last occurrence wins)
            wt = re.search(r'\[(\d+):(\d+):(\d+)<', line)
            if wt:
                h, m_val, s = int(wt.group(1)), int(wt.group(2)), int(wt.group(3))
                wall_time = h * 3600 + m_val * 60 + s

    return scoring_passes, wall_time


def analyze_log(path):
    """Analyze a single log file. Returns dict of results."""
    scoring_passes, wall_time = parse_log(path)

    if not scoring_passes:
        return None

    # Skip first pass (JIT warmup)
    jit_pass = scoring_passes[0]
    steady = scoring_passes[1:]

    if not steady:
        return None

    components = ['beam', 'stack', 'rollouts', 'metrics']
    result = {
        'path': path,
        'n_passes': len(scoring_passes),
        'n_steady': len(steady),
        'jit_total': jit_pass['total'],
        'wall_time': wall_time,
        'batch_size': steady[0]['batch'],
    }

    for c in components:
        vals = np.array([d[c] for d in steady])
        result[f'{c}_mean'] = vals.mean()
        result[f'{c}_std'] = vals.std()
        result[f'{c}_sum'] = vals.sum()

    totals = np.array([d['total'] for d in steady])
    result['total_mean'] = totals.mean()
    result['total_sum'] = totals.sum()
    result['scoring_sum'] = totals.sum() + jit_pass['total']

    # Time series for plotting
    result['timeseries'] = {c: [d[c] for d in steady] for c in components}
    result['timeseries']['total'] = [d['total'] for d in steady]

    return result


def print_report(results):
    """Print text report for all analyzed logs."""
    for r in results:
        name = os.path.basename(r['path'])
        wall = r['wall_time']
        wall_str = f"{wall//3600}h {(wall%3600)//60}m {wall%60}s" if wall else "unknown"
        non_scoring = wall - r['scoring_sum'] if wall else None

        print(f"\n{'='*65}")
        print(f"  {name}")
        print(f"{'='*65}")
        print(f"  Wall time:       {wall_str}")
        print(f"  Scoring passes:  {r['n_passes']} ({r['n_steady']} steady + 1 JIT)")
        print(f"  JIT warmup:      {r['jit_total']:.1f}s")
        print(f"  Batch size:      {r['batch_size']} transitions/pass")
        print()

        # Overall breakdown
        print(f"  --- Overall Breakdown ---")
        items = [
            ('Training (non-scoring)', non_scoring),
            ('Scoring: rollouts (GPU)', r['rollouts_sum']),
            ('Scoring: metrics (CPU)', r['metrics_sum']),
            ('Scoring: stack (pytree)', r['stack_sum']),
            ('Scoring: beam search', r['beam_sum']),
        ]
        for label, secs in items:
            if secs is None:
                continue
            pct = secs / wall * 100 if wall else 0
            bar_len = max(1, int(pct / 2.5))
            bar = '\u2588' * bar_len
            print(f"  {label:<28} {bar:<40} {pct:5.1f}%  ({secs/3600:.2f}h)")

        print()
        print(f"  --- Per Scoring Pass (steady-state avg) ---")
        components = ['beam', 'stack', 'rollouts', 'metrics']
        for c in components:
            mean = r[f'{c}_mean']
            pct = mean / r['total_mean'] * 100
            bar_len = max(1, int(pct / 3))
            bar = '\u2588' * bar_len
            print(f"  {c:<12} {mean:.3f}s \u00b1 {r[f'{c}_std']:.3f}s  {bar:<33} {pct:5.1f}%")
        print(f"  {'total':<12} {r['total_mean']:.3f}s")
        print()


def save_plot(results, save_path):
    """Save timing breakdown as a PNG with pie chart + time series."""
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n), squeeze=False)

    colors = {
        'Training': '#3498db',
        'rollouts': '#e74c3c',
        'metrics': '#f39c12',
        'stack': '#2ecc71',
        'beam': '#9b59b6',
    }

    for i, r in enumerate(results):
        name = os.path.basename(r['path'])
        wall = r['wall_time'] or r['scoring_sum']
        non_scoring = wall - r['scoring_sum']

        # --- Pie chart ---
        ax = axes[i, 0]
        sizes = [
            max(0, non_scoring),
            r['rollouts_sum'],
            r['metrics_sum'],
            r['stack_sum'],
            r['beam_sum'],
        ]
        labels = [
            f"Training\n({non_scoring/3600:.2f}h)",
            f"Rollouts\n({r['rollouts_sum']/3600:.2f}h)",
            f"Metrics\n({r['metrics_sum']/3600:.2f}h)",
            f"Stack\n({r['stack_sum']/3600:.2f}h)",
            f"Beam\n({r['beam_sum']/3600:.2f}h)",
        ]
        pie_colors = [colors['Training'], colors['rollouts'], colors['metrics'],
                      colors['stack'], colors['beam']]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.75,
            textprops={'fontsize': 9},
        )
        for t in autotexts:
            t.set_fontsize(8)
        ax.set_title(f'{name}\nTotal: {wall/3600:.2f}h', fontsize=11, fontweight='bold')

        # --- Time series ---
        ax2 = axes[i, 1]
        x = np.arange(len(r['timeseries']['rollouts']))
        components = ['metrics', 'rollouts', 'stack', 'beam']
        bottom = np.zeros(len(x))
        for c in components:
            vals = np.array(r['timeseries'][c])
            ax2.bar(x, vals, bottom=bottom, color=colors[c], label=c, width=1.0, alpha=0.8)
            bottom += vals

        ax2.set_xlabel('Scoring Pass')
        ax2.set_ylabel('Time (s)')
        ax2.set_title('Per-Pass Breakdown Over Training', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.set_xlim(0, len(x))
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze training time breakdown')
    parser.add_argument('logs', nargs='+', help='SLURM log files')
    parser.add_argument('-o', '--output', default=None,
                        help='Output PNG path (default: timing_breakdown.png in logs dir)')
    args = parser.parse_args()

    results = []
    for path in args.logs:
        r = analyze_log(path)
        if r:
            results.append(r)
        else:
            print(f"Warning: no scoring data in {path}", file=sys.stderr)

    if not results:
        print("No scoring data found in any log file.", file=sys.stderr)
        sys.exit(1)

    print_report(results)

    save_path = args.output
    if not save_path:
        log_dir = os.path.dirname(os.path.abspath(args.logs[0]))
        save_path = os.path.join(log_dir, 'timing_breakdown.png')
    save_plot(results, save_path)
