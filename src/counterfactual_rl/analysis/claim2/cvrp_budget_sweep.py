"""
Claim-2 analysis for the BUDGET DIAL sweep.

Reads runs named  bd_{arm}_b{budget*100}_c{capacity}_s{seed}  and asks one question:

    does CCE's advantage over PER TRACK the dial?

Metrics (all on opt_ratio = customers served / max servable, in [0, 1]):
  AUC       mean of the eval curve — the sample-efficiency summary (area under curve)
  final     mean of the last 10% of evals — where it ended up
  ep@thr    first episode reaching a bar — raw speed
  P(>PER)   stratified-bootstrap probability a random seed of this arm beats a random
            PER seed on AUC. 0.5 = indistinguishable.

The headline plot is P(>PER) vs budget_mult, one line per arm, one panel per capacity.
The registered prediction is an INVERTED U (advantage peaks mid-dial): looser budgets
concentrate the stakes but destroy headroom, tighter budgets keep headroom but flatten
the stakes. Flat or monotone contradicts it.

Run:
    python -m counterfactual_rl.analysis.claim2.cvrp_budget_sweep \
        --runs-dir src/counterfactual_rl/agents/cvrp/runs --out docs/figures/real/claim2
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ['uniform', 'per', 'cceonly', 'cceadd', 'ccemul']
ARM_LABEL = {
    'uniform': 'DQN-Uniform', 'per': 'DQN+PER', 'cceonly': 'DQN+CCE-only',
    'cceadd': 'CCE+TD (add)', 'ccemul': 'CCE+TD (mul)',
}
RUN_RE = re.compile(r'^bd_(\w+?)_b(\d+)_c(\d+)_s(\d+)$')
RUN_RE_I = re.compile(r'^bdi_([a-z0-9]+)_(\w+?)_b(\d+)_c(\d+)_s(\d+)$')
THRESHOLDS = (0.75, 0.90, 1.00)


def read_curve(path: Path):
    """(episodes, opt_ratio) from a metrics.log, skipping the comment header."""
    eps, vals = [], []
    for line in path.read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            eps.append(int(parts[0]))
            vals.append(float(parts[3]))
        except ValueError:
            continue          # the column header row
    return np.array(eps), np.array(vals)


def arm_order(data):
    """Known arms first, then any tuning-probe arms discovered in the run names."""
    found = {a for cell in data.values() for a in cell}
    extra = sorted(found - set(ARMS))
    return [a for a in ARMS if a in found] + extra


def parse_name(name):
    """-> (instance, arm, budget_mult, capacity, seed) or None."""
    m = RUN_RE.match(name)
    if m:
        return ('default', m.group(1), int(m.group(2)) / 100.0, int(m.group(3)), int(m.group(4)))
    m = RUN_RE_I.match(name)
    if m:
        return (m.group(1), m.group(2), int(m.group(3)) / 100.0,
                int(m.group(4)), int(m.group(5)))
    return None


def collect(runs_dir: Path):
    """{(instance, budget_mult, capacity): {arm: [ {seed, eps, vals}, ... ]}}"""
    data = defaultdict(lambda: defaultdict(list))
    for d in sorted(list(runs_dir.glob('bd_*')) + list(runs_dir.glob('bdi_*'))):
        parsed = parse_name(d.name)
        f = d / 'metrics.log'
        if not parsed or not f.exists():
            continue
        inst, arm, b, cap, seed = parsed
        eps, vals = read_curve(f)
        if eps.size < 5:
            continue
        data[(inst, b, cap)][arm].append(dict(seed=seed, eps=eps, vals=vals))
    return data


def curve_stats(runs):
    """Per-seed AUC, final score, and episodes-to-threshold."""
    auc, final, thr = [], [], {t: [] for t in THRESHOLDS}
    for r in runs:
        v, e = r['vals'], r['eps']
        auc.append(float(v.mean()))
        tail = max(1, len(v) // 10)
        final.append(float(v[-tail:].mean()))
        for t in THRESHOLDS:
            hit = np.flatnonzero(v >= t - 1e-9)
            thr[t].append(int(e[hit[0]]) if hit.size else None)
    return np.array(auc), np.array(final), thr


def boot_p_better(a, b, n=10000, seed=0):
    """P(a random draw from a beats one from b); ties count half."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float('nan')
    ai = rng.integers(0, a.size, (n, a.size))
    bi = rng.integers(0, b.size, (n, b.size))
    x, y = a[ai][:, :, None], b[bi][:, None, :]
    wins = ((x > y).mean(axis=(1, 2)) + 0.5 * (x == y).mean(axis=(1, 2)))
    return float(wins.mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--out', default=None, help='directory for the summary json + figure')
    args = ap.parse_args(argv)

    data = collect(Path(args.runs_dir))
    if not data:
        raise SystemExit(f"no bd_* runs found under {args.runs_dir}")

    summary = {}
    for (inst, b, cap) in sorted(data):
        arms = data[(inst, b, cap)]
        print(f"\n=== {inst}   budget {b:.2f}x   capacity {cap} "
              f"=====================================")
        print(f"{'arm':<14} {'n':>3} {'AUC':>8} {'final':>8} "
              f"{'ep@0.90':>9} {'P(>PER,AUC)':>12}")
        per_auc = curve_stats(arms['per'])[0] if 'per' in arms else np.array([])
        row = {}
        for arm in arm_order(data):
            if arm not in arms:
                continue
            auc, final, thr = curve_stats(arms[arm])
            hits = [x for x in thr[0.90] if x is not None]
            ep90 = f"{np.median(hits):.0f}" if hits else "never"
            p = boot_p_better(auc, per_auc) if arm != 'per' and per_auc.size else float('nan')
            print(f"{ARM_LABEL.get(arm, arm):<14} {auc.size:>3} {auc.mean():>8.4f} "
                  f"{final.mean():>8.4f} {ep90:>9} "
                  f"{('--' if arm == 'per' else f'{p:.3f}'):>12}")
            row[arm] = dict(n=int(auc.size), auc=float(auc.mean()), auc_std=float(auc.std()),
                            final=float(final.mean()), final_std=float(final.std()),
                            ep90=(float(np.median(hits)) if hits else None),
                            p_beats_per=(None if arm == 'per' else float(p)))
        summary[f"{inst}_b{b:.2f}_c{cap}"] = row

    # ── the dial verdict ─────────────────────────────────────────────────────
    print("\n\n=== DOES THE ADVANTAGE TRACK THE DIAL? (P(arm > PER) on AUC) ===")
    cells = sorted({(i, c) for (i, _, c) in data})
    for inst, cap in cells:
        budgets = sorted({b for (i, b, c) in data if c == cap and i == inst})
        print(f"\n{inst}, capacity {cap}")
        print(f"{'arm':<14} " + " ".join(f"{b:>7.2f}x" for b in budgets))
        for arm in arm_order(data):
            if arm == 'per':
                continue
            row_cells = []
            for b in budgets:
                r = summary.get(f"{inst}_b{b:.2f}_c{cap}", {}).get(arm)
                row_cells.append(f"{r['p_beats_per']:>8.3f}"
                                 if r and r['p_beats_per'] is not None else f"{'--':>8}")
            print(f"{ARM_LABEL.get(arm, arm):<14} " + " ".join(row_cells))
    print("\nreading: 0.5 = indistinguishable from PER. >0.5 = better. "
          "Registered prediction = a PEAK in the middle of the dial.")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'cvrp_budget_sweep_summary.json').write_text(json.dumps(summary, indent=2))
        _plot(summary, data, out / 'fig_c2_cvrp_budget_dial.png')
        print(f"\nwrote {out / 'cvrp_budget_sweep_summary.json'}")
    return summary


def _plot(summary, data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cells = sorted({(i, c) for (i, _, c) in data})
    fig, axes = plt.subplots(1, len(cells), figsize=(5.5 * len(cells), 4.5), squeeze=False)
    for ax, (inst, cap) in zip(axes[0], cells):
        budgets = sorted({b for (i, b, c) in data if c == cap and i == inst})
        for arm in arm_order(data):
            if arm == 'per':
                continue
            ys = []
            for b in budgets:
                r = summary.get(f"{inst}_b{b:.2f}_c{cap}", {}).get(arm)
                ys.append(r['p_beats_per'] if r and r['p_beats_per'] is not None else np.nan)
            ax.plot(budgets, ys, marker='o', label=ARM_LABEL.get(arm, arm))
        ax.axhline(0.5, color='k', ls='--', lw=1, label='= PER')
        ax.set_xlabel('budget multiple of optimal tour  (the DIAL)')
        ax.set_ylabel('P(beats PER) on AUC')
        ax.set_title(f'{inst}, capacity {cap}')
        ax.set_ylim(0, 1)
        ax.grid(alpha=.3)
    axes[0][-1].legend(fontsize=8)
    fig.suptitle('CCE vs PER across the budget dial — does the advantage track stakes?')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == '__main__':
    main()
