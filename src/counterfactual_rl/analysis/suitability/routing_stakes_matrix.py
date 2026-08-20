"""
GATE 1 — does giving routing cliffs actually change the stakes distribution?

Answered entirely from the exact oracle. No training, no cluster, minutes not days.

WHY THIS GATE EXISTS
--------------------
Budget routing produced a clean Claim-2 null over ~2,000 runs, and the cause turned out to be a
property of the environment we could have measured beforehand for free: how much the action
choice actually changes the outcome, per decision state.

    band          meaning
    barely  <5%   the choice is nearly free
    MIDDLE 5-50%  the choice matters a bit -- the useless region
    critical>50%  the choice decides the run

    FrozenLake det   50.9 /  0.0 / 49.1   <- the only env CCE wins on: NOTHING in the middle
    budget routing   10.9 / 75.2 / 13.9   <- a smooth ramp: nothing to isolate

A replay method that ranks states and replays the top slice needs a CLUMP to isolate. On a smooth
ramp the cut is arbitrary and the ordering carries little beyond noise. So the question for
Option A is not "is it harder" but "did the middle band collapse".

PASS: some configuration puts the middle band well under 25%.
FAIL: none of them move it -> STOP. That is a real, publishable result about when CCE applies,
      and it costs nothing further to establish.

Run:
    python -m counterfactual_rl.analysis.suitability.routing_stakes_matrix \
        --budget-mult 0.80 --capacity 10 --out docs/figures/real/suitability
"""

import argparse
import json
from pathlib import Path

import numpy as np

from counterfactual_rl.envs.routing_budget import BudgetRoutingEnv
from counterfactual_rl.analysis.claim1.cvrp.budget_oracle import compute_oracle, stakes

CONFIGS = [
    ("baseline (today)",            dict()),
    ("windows x3",                  dict(time_windows=True, n_windowed=3)),
    ("windows all",                 dict(time_windows=True, n_windowed=99)),
    ("strand, terminal",            dict(allow_stranding=True, reward_shape='terminal')),
    ("strand, stepwise",            dict(allow_stranding=True, reward_shape='stepwise')),
    ("windows x3 + strand term",    dict(time_windows=True, n_windowed=3,
                                         allow_stranding=True, reward_shape='terminal')),
    ("windows x3 + strand step",    dict(time_windows=True, n_windowed=3,
                                         allow_stranding=True, reward_shape='stepwise')),
    ("windows all + strand term",   dict(time_windows=True, n_windowed=99,
                                         allow_stranding=True, reward_shape='terminal')),
]

REFERENCE = [
    ("FrozenLake 8x8 det   (CCE WINS)", 50.9, 0.0, 49.1),
    ("FrozenLake 8x8 slip  (null)",     24.5, 62.3, 13.2),
]


def bands(st):
    """Share of decision states in each band, normalised by the per-env max spread."""
    st = np.asarray(st, float)
    if st.size == 0 or st.max() <= 0:
        return float('nan'), float('nan'), float('nan')
    rel = st / st.max()
    return (100 * (rel < 0.05).mean(),
            100 * ((rel >= 0.05) & (rel <= 0.50)).mean(),
            100 * (rel > 0.50).mean())


def measure(budget_mult, capacity, window_width, instance_kw=None):
    rows = []
    for label, kw in CONFIGS:
        try:
            env = BudgetRoutingEnv(budget_mult=budget_mult, capacity=capacity,
                                   dist_scale=10, window_width=window_width,
                                   **(instance_kw or {}), **kw)
            V, Q = compute_oracle(env)
            st = stakes(env, Q)
            decision = st[env.action_masks_np.sum(1) >= 2]
            lo, mid, hi = bands(decision)
            rows.append(dict(
                label=label, states=int(env.n_states), decision=int(decision.size),
                failures=int(env.n_failure_states),
                optimal=float(V[env.start_states[0]]),
                barely=lo, middle=mid, critical=hi,
                windowed=list(map(int, env.windowed_customers)),
            ))
        except Exception as e:
            rows.append(dict(label=label, error=f"{type(e).__name__}: {e}"))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget-mult', type=float, default=0.80)
    ap.add_argument('--capacity', type=int, default=10)
    ap.add_argument('--window-width', type=int, default=6)
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    rows = measure(args.budget_mult, args.capacity, args.window_width)

    print(f"\nGATE 1 - stakes distribution, budget {args.budget_mult:.2f}x, "
          f"capacity {args.capacity}, window width {args.window_width}\n")
    print(f"{'configuration':<28} {'states':>9} {'fail':>6} {'opt':>4} "
          f"{'barely':>7} {'MIDDLE':>7} {'crit':>7}")
    print("-" * 78)
    for label, lo, mid, hi in REFERENCE:
        print(f"{label:<28} {'-':>9} {'-':>6} {'-':>4} {lo:>6.1f}% {mid:>6.1f}% {hi:>6.1f}%")
    print("-" * 78)
    for r in rows:
        if 'error' in r:
            print(f"{r['label']:<28} {r['error']}")
            continue
        print(f"{r['label']:<28} {r['states']:>9,} {r['failures']:>6,} "
              f"{int(round(r['optimal'])):>4} "
              f"{r['barely']:>6.1f}% {r['middle']:>6.1f}% {r['critical']:>6.1f}%")

    good = [r for r in rows if 'error' not in r and r['middle'] < 25.0]
    base = next((r for r in rows if r['label'] == 'baseline (today)'), None)
    print()
    if base and 'middle' in base:
        print(f"baseline middle band: {base['middle']:.1f}%")
    if good:
        best = min(good, key=lambda r: r['middle'])
        print(f"GATE 1 PASSED - '{best['label']}' puts the middle band at "
              f"{best['middle']:.1f}% (target <25%).")
    else:
        ok = [r for r in rows if 'error' not in r]
        if ok:
            best = min(ok, key=lambda r: r['middle'])
            print(f"GATE 1 FAILED - best configuration is '{best['label']}' at "
                  f"{best['middle']:.1f}%, still above the 25% target.")
        print("STOP: do not build a trainer or spend a sweep. Routing lacking pivotal "
              "decisions is itself the result.")

    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        p = out / 'routing_stakes_matrix.json'
        p.write_text(json.dumps(dict(budget_mult=args.budget_mult, capacity=args.capacity,
                                     window_width=args.window_width, rows=rows), indent=2))
        print(f"\nwrote {p}")
    return rows


if __name__ == '__main__':
    main()
