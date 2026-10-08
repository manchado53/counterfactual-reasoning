"""
Compare CVRP arms at MATCHED GRADIENT STEPS, which is the only fair axis here.

Arms do not run at the same speed. CCE spends 50-130x more compute per gradient step
on scoring than PER does, so at any wall-clock moment -- and at any episode count --
the CCE arms have taken far fewer steps. Comparing final numbers, or comparing at equal
episodes, credits PER for a head start it got from being cheap.

This reads every run in a manifest, interpolates each arm's eval curve onto a common
gradient-step grid, and reports the comparison only where every arm has actually
reached that many steps.

NOISE FLOOR. uniform vs PER was replicated twice at nspu=512 and came out 7.345/7.323
then 7.385/7.400 -- the sign flipped. At one seed, a difference under ~0.05 tour length
is noise. Differences below that are printed as "tie" regardless of sign.

    python -m counterfactual_rl.agents.cvrp_neural.compare_arms \
        --manifest <path/to/manifest.json> [--at 20000]
"""

import argparse
import glob
import json
import os

import numpy as np

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
NOISE = 0.05
OPT, THEIRS = 6.10, 6.36


def curve(job, nspu):
    m = os.path.join(RUNS, job, 'metrics.log')
    if not os.path.exists(m):
        return None
    ep, tour = [], []
    for line in open(m):
        if line.startswith('#'):
            continue
        p = line.split()
        if not p or p[0] == 'episode':
            continue
        ep.append(int(p[0])); tour.append(-float(p[5]))
    if len(ep) < 2:
        return None
    # ~24 decisions per n=20 episode; the same conversion parse_logs uses
    return np.array(ep) * 24.0 / nspu, np.array(tour)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', action='append', required=True,
                    help='repeatable; pass several to compare across budgets')
    ap.add_argument('--at', type=int, default=None,
                    help='gradient-step budget to compare at (default: the largest '
                         'budget every arm has reached)')
    ap.add_argument('--smooth', type=int, default=5,
                    help='trailing evals averaged at the comparison point; a single '
                         'eval is noisy enough to invent a winner')
    args = ap.parse_args()

    arms = {}
    for pattern in args.manifest:
        for path in glob.glob(pattern):
            for jid, cfg in json.load(open(path)).items():
                c = curve(jid, cfg.get('n_steps_per_update', 512))
                if c is None:
                    continue
                name = cfg.get('_arm', cfg.get('algorithm', '?'))
                if cfg.get('algorithm') == 'consequence-dqn':
                    name += f" h{cfg.get('cf_horizon', '?')}"
                name += f" [nspu{cfg.get('n_steps_per_update')}]"
                arms[name] = dict(job=jid, gs=c[0], tour=c[1], reached=c[0][-1])

    if not arms:
        raise SystemExit('no runs found for those manifests')

    common = min(a['reached'] for a in arms.values())
    at = args.at or int(common)
    print(f"comparing at {at:,} gradient steps "
          f"(every arm has reached {common:,.0f})\n")
    if at > common:
        print(f"  WARNING: {at:,} exceeds what some arms have reached; those are "
              f"extrapolated and must not be read as results.\n")

    rows = []
    for name, a in sorted(arms.items()):
        if a['reached'] < at * 0.98:
            rows.append((name, a['job'], None, a['reached']))
            continue
        # trailing mean of the evals at or before `at`
        idx = np.searchsorted(a['gs'], at)
        lo = max(0, idx - args.smooth)
        rows.append((name, a['job'], float(np.mean(a['tour'][lo:idx + 1])), a['reached']))

    done = [r for r in rows if r[2] is not None]
    best = min((r[2] for r in done), default=None)
    print(f"{'arm':<30}{'job':>8}{'tour':>9}{'gap':>8}{'vs best':>10}")
    for name, job, t, reached in sorted(rows, key=lambda r: (r[2] is None, r[2] or 0)):
        if t is None:
            print(f"{name:<30}{job:>8}{'--':>9}{'':>8}   only {reached:,.0f} steps so far")
            continue
        d = t - best
        tag = 'best' if d == 0 else ('tie' if abs(d) < NOISE else f'{d:+.3f}')
        print(f"{name:<30}{job:>8}{t:>9.3f}{100*(t-OPT)/OPT:>7.1f}%{tag:>10}")

    print(f"\n  RP-DQN greedy {THEIRS} (gap 4.26%) · optimal {OPT}")
    print(f"  differences under {NOISE} at one seed are noise -- shown as 'tie'")


if __name__ == '__main__':
    main()
