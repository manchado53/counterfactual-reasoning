"""Option B CLI — precision@k + ESS from runs' `sampling.npz`, graded vs the EXACT FrozenLake
oracle. The validation claim is a CONTRAST: precision@k(WIN run) >> precision@k(STALE run).

Usage:
    python -m counterfactual_rl.analysis.suitability.run_realized_sampling \
        WIN=src/.../runs/<job_W> STALE=src/.../runs/<job_S>

Each arg is LABEL=run_dir (or just run_dir). Reads map/slippery/gamma from the run's checkpoint
so the oracle matches the env the agent trained in.
"""
import argparse
import json
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from counterfactual_rl.envs.frozen_lake import FrozenLakeEnv
from counterfactual_rl.analysis.claim1.frozen_lake.oracle import compute_oracle
from counterfactual_rl.analysis.suitability.envs import qstar_spread_exact
from counterfactual_rl.analysis.suitability.rollout_sweep import compute_realized_sampling

KS = (0.05, 0.10, 0.20)

# Repo root → default output folder (docs/figures/suitability), so results auto-land there.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
_OUT_DIR = os.path.join(_REPO, 'docs', 'figures', 'suitability')


def _run_config(run_dir: str) -> dict:
    for name in ('last.pkl', 'best.pkl'):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            with open(p, 'rb') as f:
                d = pickle.load(f)
            return d.get('config', d) if isinstance(d, dict) else {}
    return {}


def analyze(run_dir: str) -> dict:
    cfg = _run_config(run_dir)
    map_name = cfg.get('map_name', '8x8')
    is_slippery = bool(cfg.get('is_slippery', False))
    gamma = float(cfg.get('gamma', 0.99))

    _, oracle, non_terminal = compute_oracle(map_name=map_name, is_slippery=is_slippery, gamma=gamma)
    nt = np.array(non_terminal, dtype=np.int64)
    mean_gap = np.array([oracle[s] for s in non_terminal], dtype=np.float64)   # paper truth (headline)

    res = compute_realized_sampling(run_dir, nt, mean_gap, ks=KS)
    res['config'] = {'map': map_name, 'is_slippery': is_slippery, 'gamma': gamma}
    return res


def _headline(r):
    """Supply-normalized cumulative oversampling stats, or None."""
    return r.get('cumulative') if r.get('supply_normalized') else None


def _save_figure(results, fig_path):
    """Per-run: supply-normalized Spearman(oversampling,stakes) vs RAW Spearman (left),
    and mean oversampling on top-10% stakes states with the fair-share line at 1.0 (right)."""
    have = {k: v for k, v in results.items() if not v.get('empty') and _headline(v)}
    if not have:
        return
    labels = list(have)
    x = np.arange(len(labels)); w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    raw = [have[l]['raw']['spearman'] for l in labels]
    norm = [_headline(have[l])['spearman'] for l in labels]
    ax1.bar(x - w/2, raw, w, label='raw draws', color='#bbb')
    ax1.bar(x + w/2, norm, w, label='supply-normalized', color='#2196F3')
    ax1.axhline(0, color='k', lw=.5); ax1.set_ylim(-1, 1)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel('Spearman(score, stakes)'); ax1.legend(fontsize=8)
    ax1.set_title('Does it track stakes? (raw vs supply-normalized)')
    over = [_headline(have[l])['mean_oversampling_top10stakes'] for l in labels]
    ax2.bar(x, over, color='#4CAF50')
    ax2.axhline(1.0, color='r', lw=1, ls='--', label='fair share (=1)')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('mean oversampling on top-10% stakes'); ax2.legend(fontsize=8)
    ax2.set_title('Over-drilling of high-stakes states')
    fig.suptitle('Option B — supply-normalized replay sampling', fontweight='bold')
    fig.tight_layout(); fig.savefig(fig_path, dpi=130, bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='LABEL=run_dir (or run_dir)')
    ap.add_argument('--out', default=os.path.join(_OUT_DIR, 'realized_sampling.json'),
                    help='JSON summary path (default: docs/figures/suitability/)')
    ap.add_argument('--fig', default=os.path.join(_OUT_DIR, 'realized_sampling.png'),
                    help='figure path (default: docs/figures/suitability/)')
    args = ap.parse_args()

    results = {}
    for item in args.runs:
        label, _, run_dir = item.partition('=')
        if not run_dir:
            label, run_dir = os.path.basename(label.rstrip('/')), label
        print(f"\n=== {label}  ({run_dir}) ===")
        r = analyze(run_dir)
        results[label] = r
        if r.get('empty'):
            print(f"  EMPTY: {r.get('reason', 'no draws logged')}  ({r.get('path')})")
            continue
        print(f"  config: {r['config']}   total_draws={r['total_draws']}  eval_states={r['n_eval_states']}")
        print(f"  RAW  Spearman(draws,stakes)={r['raw']['spearman']:+.3f}  precision@k={r['raw']['precision_at_k']}")
        h = _headline(r)
        if h:
            print(f"  NORM Spearman(oversampling,stakes)={h['spearman']:+.3f}  precision@k={h['precision_at_k']}")
            print(f"       mean oversampling on top-10% stakes = {h['mean_oversampling_top10stakes']:.2f}  (1.0=fair)")
            lw = r.get('late_window')
            if lw:
                print(f"       late-window Spearman={lw['spearman']:+.3f} (from update {lw['from_update']})")
        else:
            print("  NO supply normalization — `adds` missing (old npz); RERUN to get the fix.")

    # contrast: supply-normalized headline (Spearman + top-10% oversampling)
    have = {k: v for k, v in results.items() if not v.get('empty') and _headline(v)}
    if len(have) >= 2:
        print(f"\n--- CONTRAST (supply-normalized) ---")
        print(f"  {'cond':<12}{'Spearman':>10}{'meanOver@top10':>16}")
        for k in sorted(have, key=lambda kk: -_headline(have[kk])['spearman']):
            h = _headline(have[k])
            print(f"  {k:<12}{h['spearman']:>+10.3f}{h['mean_oversampling_top10stakes']:>16.2f}")

    # Auto-save summary + figure into docs/figures/suitability/.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(f"\nwrote {args.out}")
    _save_figure(results, args.fig)
    if os.path.exists(args.fig):
        print(f"wrote {args.fig}")


if __name__ == '__main__':
    main()
