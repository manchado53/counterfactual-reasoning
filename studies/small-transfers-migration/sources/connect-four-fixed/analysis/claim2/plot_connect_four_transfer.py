"""Plot actual fixed-foe logs. Never fill missing tails or merge unlike runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from counterfactual_rl.agents.connect_four.transfer_run import effective_config


def load_runs(paths):
    runs, seen, contract = [], set(), None
    for path in paths:
        p = json.loads((path / 'provenance.json').read_text())
        cfg = p['config']
        key = cfg['arm'], cfg['seed']
        if cfg != effective_config(*key):
            raise ValueError(f'Unexpected effective configuration for {key}')
        if key in seen:
            raise ValueError(f'Duplicate arm/seed: {key}')
        seen.add(key)
        arm_fields = ('arm', 'seed', 'mu', 'priority_mixing', 'cce_enabled',
                      'normalise_priorities', 'score_on_insert', 'active_features', 'inactive_keys')
        base = {k: v for k, v in cfg.items() if k not in arm_fields}
        identity = dict(config=base, hashes=p['source_hashes'], packages=p['packages'],
                        mode=p['mode'], bank=p['eval_bank_hash'])
        if contract is not None and identity != contract:
            raise ValueError('Mixed source, package, config, mode, or evaluation bank')
        contract = identity
        log = path / 'eval.jsonl'
        rows = []
        if log.exists():
            lines = log.read_text().splitlines(keepends=True)
            for i, line in enumerate(lines):
                # A live writer may have a partial final line. Never invent its values.
                if i == len(lines) - 1 and not line.endswith('\n'):
                    break
                row = json.loads(line)
                if (row['arm'], row['seed']) != key:
                    raise ValueError('Log identity differs from provenance')
                rows.append(row)
        steps = [r['steps'] for r in rows]
        if any(a >= b for a, b in zip(steps, steps[1:])):
            raise ValueError('Duplicate or unsorted evaluation steps')
        runs.append((key, rows))
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    runs = load_runs(args.runs)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    arms = [a for a in ('uniform','per','cce_mul','cce_add','cce_only') if any(key[0] == a for key, _ in runs)]
    colors = dict(zip(arms, plt.get_cmap('tab10').colors))
    for ax, role in zip(axes, (None, 0, 1)):
        for arm in arms:
            series = []
            for (a, seed), rows in runs:
                if a != arm or not rows:
                    continue
                values = {r['steps']: (r if role is None else r[f'role_{role}'])['win_rate'] for r in rows}
                series.append(values)
                ax.plot(list(values), list(values.values()), color=colors[arm], alpha=.75,
                        linewidth=1., marker='o', markersize=4, label=f'{arm} seed {seed}')
            if len(series) == 2:
                common = sorted(set(series[0]) & set(series[1]))
                ys = np.array([[s[x] for x in common] for s in series])
                ax.plot(common, ys.mean(0), color=colors[arm], label=f'{arm} mean', linewidth=2)
                ax.plot(common, np.median(ys, axis=0), color=colors[arm], linestyle='--',
                        label=f'{arm} median', linewidth=1)
        ax.set_title('Equal-role mean' if role is None else ('Moves first' if role == 0 else 'Moves second'))
        ax.set_xlabel('Learner decisions')
        ax.set_ylim(-.03, 1.03)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Win rate against frozen foe')
    axes[-1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1., 1.))
    counts = ', '.join(f'{a}: {sum(bool(rows) for (arm, _), rows in runs if arm == a)} seeds' for a in arms)
    mode = json.loads((args.runs[0] / 'provenance.json').read_text())['mode']
    phase = 'Smoke checks (not full training)' if mode == 'smoke' else 'Full training (live)'
    fig.suptitle(phase + '\nDots are actual checks; two fixed role paths, 500 repeats each\n' + counts)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches='tight')


if __name__ == '__main__':
    main()
