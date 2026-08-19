"""
Claim 2 analysis for the routing sweep.

Final score alone cannot separate the arms here: plain DQN already reaches ~1.0 of
optimal, so every arm ceilings out (the same trap the graded-slip sweep hit). So the
primary metric is SPEED — episodes to first reach a bar — with area-under-curve as a
second view and final score reported for honesty.

Run:
    python -m counterfactual_rl.analysis.claim2.cvrp_sweep --runs-dir <path/to/runs>
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

ARMS = ['uniform', 'per', 'cceonly', 'cceadd', 'ccemul']
ARM_LABEL = {
    'uniform': 'DQN-Uniform', 'per': 'DQN+PER', 'cceonly': 'DQN+CCE-only',
    'cceadd': 'CCE+TD (add)', 'ccemul': 'CCE+TD (mul)',
}
THRESHOLDS = (0.90, 0.95, 0.99)


def read_curve(path: Path):
    """(episodes, opt_ratio) from a metrics.log, ignoring the comment header."""
    eps, vals = [], []
    for line in path.read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            eps.append(int(parts[0])); vals.append(float(parts[3]))
        except ValueError:
            continue  # header row
    return np.array(eps), np.array(vals)


def steps_to(eps, vals, thr):
    """First episode at which the curve reaches thr; None if it never does."""
    hit = np.flatnonzero(vals >= thr)
    return int(eps[hit[0]]) if hit.size else None


def collect(runs_dir: Path):
    pat = re.compile(r'^c2_(\w+?)_n(\d+)_s(\d+)$')
    data = {}
    for d in sorted(runs_dir.glob('c2_*')):
        m = pat.match(d.name)
        f = d / 'metrics.log'
        if not m or not f.exists():
            continue
        arm, tag, seed = m.group(1), m.group(2), int(m.group(3))
        noise = {'00': 0.0, '015': 0.15}.get(tag, float('nan'))
        eps, vals = read_curve(f)
        if eps.size < 5:
            continue
        data.setdefault(noise, {}).setdefault(arm, []).append(
            dict(seed=seed, eps=eps, vals=vals))
    return data


def boot_p_better(a, b, n=10000, seed=0):
    """P(a random draw from a beats one from b), stratified bootstrap."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float('nan')
    ai = rng.integers(0, a.size, (n, a.size))
    bi = rng.integers(0, b.size, (n, b.size))
    wins = (a[ai][:, :, None] > b[bi][:, None, :]).mean(axis=(1, 2))
    return float(wins.mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args(argv)

    data = collect(args.runs_dir)
    if not data:
        print(f'no c2_* runs found under {args.runs_dir}')
        return 1

    summary = {}
    for noise in sorted(data):
        print('\n' + '=' * 78)
        print(f'TRAFFIC = {noise}     '
              + ('(CCE score is ~constant here -> predict a tie)' if noise == 0.0
                 else '(CCE score is graded here -> CCE has something to work with)'))
        print('=' * 78)

        arms = data[noise]
        finals, aucs = {}, {}
        for arm in ARMS:
            runs = arms.get(arm, [])
            if not runs:
                continue
            n_tail = max(1, len(runs[0]['vals']) // 10)
            finals[arm] = np.array([r['vals'][-n_tail:].mean() for r in runs])
            aucs[arm] = np.array([r['vals'].mean() for r in runs])

        # ── speed ────────────────────────────────────────────────────────
        print(f"\n{'arm':<15}{'seeds':>6}", end='')
        for t in THRESHOLDS:
            print(f'{f"eps to {t:.2f}":>15}', end='')
        print(f"{'AUC':>8}{'final':>9}")
        print('-' * 78)
        for arm in ARMS:
            runs = arms.get(arm, [])
            if not runs:
                continue
            print(f'{ARM_LABEL[arm]:<15}{len(runs):>6}', end='')
            for t in THRESHOLDS:
                hits = [steps_to(r['eps'], r['vals'], t) for r in runs]
                got = [h for h in hits if h is not None]
                if not got:
                    print(f'{"never":>15}', end='')
                else:
                    med = int(np.median(got))
                    miss = len(hits) - len(got)
                    lbl = f'{med}' + (f' ({miss}x never)' if miss else '')
                    print(f'{lbl:>15}', end='')
            print(f'{aucs[arm].mean():>8.3f}{finals[arm].mean():>9.4f}')

        # ── vs PER ───────────────────────────────────────────────────────
        if 'per' in finals:
            print(f"\n{'vs DQN+PER':<15}{'P(better final)':>18}{'P(better AUC)':>16}"
                  f"{'speed vs PER @0.95':>22}")
            print('-' * 78)
            per_hits = [steps_to(r['eps'], r['vals'], 0.95) for r in arms['per']]
            per_med = np.median([h for h in per_hits if h is not None]) if any(
                h is not None for h in per_hits) else None
            for arm in ARMS:
                if arm == 'per' or arm not in finals:
                    continue
                pf = boot_p_better(finals[arm], finals['per'])
                pa = boot_p_better(aucs[arm], aucs['per'])
                hits = [steps_to(r['eps'], r['vals'], 0.95) for r in arms[arm]]
                got = [h for h in hits if h is not None]
                if got and per_med:
                    d = np.median(got) - per_med
                    spd = f'{d:+.0f} eps ({"faster" if d < 0 else "slower"})'
                else:
                    spd = 'n/a'
                print(f'{ARM_LABEL[arm]:<15}{pf:>18.3f}{pa:>16.3f}{spd:>22}')
            print('\n  P > 0.5 favours the arm; 0.5 = tie. '
                  'Lower "eps to" = fewer episodes = better.')

        summary[str(noise)] = {
            arm: dict(n=len(arms.get(arm, [])),
                      final_mean=float(finals[arm].mean()) if arm in finals else None,
                      final_std=float(finals[arm].std()) if arm in finals else None,
                      auc_mean=float(aucs[arm].mean()) if arm in aucs else None)
            for arm in ARMS if arm in arms
        }

    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
        print(f'\nsummary saved -> {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
