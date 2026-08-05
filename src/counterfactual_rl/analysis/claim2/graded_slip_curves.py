"""Learning curves across the whole graded-slip axis, from both sweeps.

The slip axis was covered by two sweeps with different designs:

    2026-08-03  claim2_graded_slip        5 levels, 10 seeds  (0.0, 0.166, 0.333, 0.5, 0.666)
    2026-08-04  claim2_graded_slip_dense  7 levels, 20 seeds  (0.0 ... 0.133)

They are NOT pooled. Seeds 0-9 appear in both, so pooling would double count, and a
spot check showed 10 of 58 overlapping (level, arm, seed) cells disagree anyway —
same config, same seed, different outcome — because the runs are not bit
reproducible on GPU and outcomes here are bimodal, so a tiny float difference flips
an entire seed. Each level therefore comes from exactly one sweep: the dense one
where it exists (more seeds), the coarse one otherwise.

Curves are plotted as the MEAN win rate across seeds, not IQM. In this regime a
seed either escapes the dead basin or never does, so the per-seed distribution is
bimodal and IQM's middle 50% collapses to zero while a quarter of the seeds are
sitting at 1.0 — it reports a floor that is not there. The mean tracks
"escape rate x how good the escapees are", which is what these curves are about.

Run:
    PYTHONPATH=<worktree>/src python -m counterfactual_rl.analysis.claim2.graded_slip_curves \
        --out docs/figures/graded_slip_curves
"""

import argparse
import json
import os
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .parse_logs import load_manifest, filter_complete_runs

ALG_ORDER = ["DQN-Uniform", "DQN+PER", "DQN+CCE-only", "CCE+TD (mul)"]
COLORS = {
    "DQN-Uniform": "#888888",
    "DQN+PER": "#1f77b4",
    "DQN+CCE-only": "#ff7f0e",
    "CCE+TD (mul)": "#d62728",
}
EXP = "src/counterfactual_rl/agents/frozen_lake/experiments/2026-08"
SWEEPS = {
    "dense": f"{EXP}/claim2_graded_slip_dense_2026-08-04/RECOVERED_partial.json",
    "coarse": f"{EXP}/claim2_graded_slip_2026-08-03/claim2_graded_slip_2026-08-03.json",
}
# Levels the dense sweep completed at 20 seeds; everything else falls back to coarse.
DENSE_LEVELS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.133]


def _boot_mean_ci(arr, reps=2000, seed=0):
    """Bootstrap 95% CI of the across-seed mean, per checkpoint. arr: (n_seeds, T)."""
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    idx = rng.integers(0, n, size=(reps, n))
    means = arr[idx].mean(axis=1)          # (reps, T)
    return np.percentile(means, [2.5, 97.5], axis=0)


def collect(repo_root):
    """Return {slip: (source, {alg: (steps, mean_curve, lo, hi, n_seeds)})}."""
    manifests = {}
    for name, rel in SWEEPS.items():
        path = os.path.join(repo_root, rel)
        with open(path) as f:
            m = json.load(f)
        kept, dropped = filter_complete_runs(m)
        print(f"{name}: kept {len(kept)}/{len(m)} (dropped {len(dropped)} incomplete)")
        manifests[name] = kept

    # Which sweep owns each level: dense where it finished all 4 arms, else coarse.
    levels = {}
    for name in ("dense", "coarse"):
        for cfg in manifests[name].values():
            p = round(float(cfg["slip_prob"]), 4)
            if name == "dense" and p not in DENSE_LEVELS:
                continue          # dense 0.166 is partial (2 of 4 arms) — skip it
            levels.setdefault(p, name)

    out = {}
    tmpdir = tempfile.mkdtemp(prefix="slipcurves_")
    for p, source in sorted(levels.items()):
        sub = {j: c for j, c in manifests[source].items()
               if abs(float(c["slip_prob"]) - p) < 1e-9}
        sub_path = os.path.join(tmpdir, f"{source}_{p:.3f}.json")
        with open(sub_path, "w") as f:
            json.dump(sub, f)
        data = load_manifest(sub_path)
        per_alg = {}
        for alg, arr3 in data["raw"].items():
            arr = arr3[:, 0, :]                       # (n_seeds, T)
            steps = np.asarray(data["eval_steps"][alg], dtype=float)[:arr.shape[1]]
            mean = arr.mean(axis=0)
            lo, hi = _boot_mean_ci(arr)
            per_alg[alg] = {
                "steps": steps, "mean": mean, "lo": lo, "hi": hi,
                "n": arr.shape[0], "raw": arr,
            }
        out[p] = (source, per_alg)
    return out


def plot(curves, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    slips = sorted(curves)
    ncol = 4
    nrow = int(np.ceil(len(slips) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.2 * nrow),
                             sharey=True, squeeze=False)

    for i, p in enumerate(slips):
        ax = axes[i // ncol][i % ncol]
        source, per_alg = curves[p]
        for alg in ALG_ORDER:
            if alg not in per_alg:
                continue
            d = per_alg[alg]
            ax.plot(d["steps"], d["mean"], color=COLORS[alg], lw=1.8,
                    label=f"{alg} (n={d['n']})")
            ax.fill_between(d["steps"], d["lo"], d["hi"],
                            color=COLORS[alg], alpha=0.15, lw=0)
        tag = "20 seeds" if source == "dense" else "10 seeds"
        ax.set_title(f"slip = {p:g}   ({tag})", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        if i % ncol == 0:
            ax.set_ylabel("mean win rate")
        if i // ncol == nrow - 1:
            ax.set_xlabel("environment steps")

    for j in range(len(slips), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    handles = [plt.Line2D([], [], color=COLORS[a], lw=2.5, label=a) for a in ALG_ORDER]
    fig.legend(handles=handles, loc="lower right", ncol=1, fontsize=10,
               bbox_to_anchor=(0.99, 0.06), frameon=True)
    fig.suptitle("FrozenLake 8x8 — learning curves across the slip axis  "
                 "(mean over seeds, 95% bootstrap CI)\n"
                 "slip 0.0-0.133 from the 20-seed dense sweep, 0.166-0.666 from the "
                 "10-seed coarse sweep; sweeps not pooled",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "fig_learning_curves_all_slips.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_escape(curves, out_dir, threshold=0.5):
    """Fraction of seeds above `threshold` over time — the basin-escape view.

    One panel per algorithm, one line per slip level. This is the metric the
    2026-08-04 result turned on: outcomes are bimodal, so "how many seeds got
    out" is the quantity, and the mean win rate only blurs it.
    """
    slips = sorted(curves)
    algs = ["DQN+PER", "CCE+TD (mul)"]
    fig, axes = plt.subplots(1, len(algs), figsize=(6.5 * len(algs), 4.6),
                             sharey=True)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(slips)))
    for ax, alg in zip(np.atleast_1d(axes), algs):
        for c, p in zip(cmap, slips):
            _, per_alg = curves[p]
            if alg not in per_alg:
                continue
            d = per_alg[alg]
            frac = (d["raw"] >= threshold).mean(axis=0)
            ax.plot(d["steps"], frac, color=c, lw=1.8, label=f"slip {p:g}")
        ax.set_title(f"{alg} — seeds above a {threshold:g} win rate")
        ax.set_xlabel("environment steps")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    np.atleast_1d(axes)[0].set_ylabel(f"fraction of seeds >= {threshold:g}")
    np.atleast_1d(axes)[-1].legend(fontsize=7, ncol=2, title="slip")
    fig.suptitle("Basin escape over training — how many seeds get out, by noise level",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "fig_escape_rate_over_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", default="docs/figures/graded_slip_curves")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    curves = collect(args.repo_root)
    p1 = plot(curves, args.out)
    p2 = plot_escape(curves, args.out, threshold=args.threshold)
    print(f"Wrote:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
