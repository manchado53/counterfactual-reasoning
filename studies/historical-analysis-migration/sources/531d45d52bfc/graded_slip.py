"""Graded-stochasticity analysis for FrozenLake.

Groups a `claim2_graded_slip*` manifest by `slip_prob`, computes the paper's
final-IQM and P(improvement) metrics per slip level (reusing compute_metrics),
and plots how CCE+TD(mul)'s advantage over DQN+PER changes with environment
noise. Theorem 3 predicts the advantage GROWS as slip_prob falls.

Three things this does that the first (2026-08-03) version did not:

1. DROPS RUNS THAT DIED MID-TRAINING. load_manifest forward-fills short seeds,
   which is right for an early-stopped winner (fill ~1.0) and wrong for a killed
   run (fill at its dying win rate, often 0.0). 14/200 runs died in the first
   sweep and were silently counted as seeds that never learned.

2. PLOTS AGAINST NOISE, NOT JUST SLIP. Theorem 3 is a statement about noise.
   Outcome probabilities are [p/2, 1-p, p/2], so entropy PEAKS at p=2/3
   ([1/3,1/3,1/3], H=1.099) and FALLS after it (p=1 -> [.5,0,.5], H=0.693).
   Slip and noise therefore stop agreeing above p=2/3, and those levels are
   drawn as detached markers so the reader never reads them as one curve.

3. MEASURES SPEED, NOT ONLY FINAL SCORE. Between roughly slip 0.25 and 0.5 every
   arm ends at ~0.96 — slip aids exploration, so the task gets easy and final
   IQM saturates. Steps-to-threshold still separates arms under that ceiling.

Run (as worktree code):
    PYTHONPATH=<worktree>/src python -m counterfactual_rl.analysis.claim2.graded_slip \
        --manifest <path>/claim2_graded_slip_*.json --out <out_dir>
"""

import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .parse_logs import load_manifest, filter_complete_runs
from .compute_metrics import final_iqm, prob_improvement, steps_to_threshold, iqm_curves

MUL = "CCE+TD (mul)"
PER = "DQN+PER"
ALG_ORDER = ["DQN-Uniform", "DQN+PER", "DQN+CCE-only", "CCE+TD (mul)"]
COLORS = {
    "DQN-Uniform": "#888888",
    "DQN+PER": "#1f77b4",
    "DQN+CCE-only": "#ff7f0e",
    "CCE+TD (mul)": "#d62728",
}

# Above this slip the intended action is no longer the most likely outcome and
# entropy starts falling again, so these levels are a separate probe rather than
# a continuation of the noise curve.
NOISE_PEAK = 2.0 / 3.0


def slip_entropy(p):
    """Shannon entropy (nats) of the outcome distribution [p/2, 1-p, p/2].

    This is the actual 'how noisy is the environment' axis. It rises from 0 at
    p=0 to its maximum ln(3)=1.099 at p=2/3, then falls back to ln(2)=0.693 at
    p=1 where the intended action becomes impossible.
    """
    d = np.array([p / 2.0, 1.0 - p, p / 2.0], dtype=float)
    d = d[d > 0]
    return float(-(d * np.log(d)).sum())


def _group_by_slip(manifest):
    """Return {slip_prob: {job_id: cfg}} from a manifest dict."""
    groups = {}
    for job_id, cfg in manifest.items():
        p = float(cfg.get("slip_prob", 2.0 / 3.0))
        groups.setdefault(p, {})[job_id] = cfg
    return dict(sorted(groups.items()))


def _load_cache(npz_path):
    """Reverse of build_graded_slip_cache.py: rebuild the per-slip parsed arrays
    from the .npz so figures regenerate WITHOUT the raw run dirs.

    Returns (raw_by_slip, evalsteps_by_slip, n_submitted, dropped).
    """
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    raw_by_slip, evalsteps_by_slip = {}, {}
    for p in meta["slips"]:
        k = f"{p:.3f}"
        labels = meta["groups"][k]["labels"]
        raw_by_slip[float(p)] = {a: z[f"slip{k}_raw_{i}"] for i, a in enumerate(labels)}
        evalsteps_by_slip[float(p)] = {a: z[f"slip{k}_steps_{i}"] for i, a in enumerate(labels)}
    return (raw_by_slip, evalsteps_by_slip,
            meta.get("n_submitted"), meta.get("dropped_runs", []))


def _split_main_probe(slips):
    """Split slip levels into the noise curve (<= 2/3) and the probe (> 2/3)."""
    main = [p for p in slips if p <= NOISE_PEAK + 1e-9]
    probe = [p for p in slips if p > NOISE_PEAK + 1e-9]
    return main, probe


def analyze(source, out_dir, reps=50000, thresholds=(0.5, 0.9),
            drop_incomplete=True, curve_reps=5000, from_cache=False):
    os.makedirs(out_dir, exist_ok=True)

    # --- load the parsed per-slip arrays, from the .npz cache or the raw runs ---
    raw_by_slip = {}        # slip -> {alg: (n_seeds, 1, n_checkpoints)}
    evalsteps_by_slip = {}  # slip -> {alg: (n_checkpoints,)}
    if from_cache:
        raw_by_slip, evalsteps_by_slip, n_submitted, dropped = _load_cache(source)
        print(f"Loaded cache {source}: {len(raw_by_slip)} slip levels "
              f"({n_submitted} submitted, {len(dropped)} dead runs dropped)")
    else:
        with open(source) as f:
            manifest = json.load(f)
        n_submitted = len(manifest)
        dropped = []
        if drop_incomplete:
            manifest, dropped = filter_complete_runs(manifest)
            print(f"Completeness filter: kept {len(manifest)}/{n_submitted}, "
                  f"dropped {len(dropped)}")
            for d in dropped:
                if d.get("reason") == "died mid-training":
                    print(f"  drop {d['job_id']}: died at episode {d['last_episode']}"
                          f"/{d['n_episodes']} with win rate {d['last_win_rate']:.2f}")
                else:
                    print(f"  drop {d['job_id']}: {d['reason']}")
        # Scratch sub-manifests go to a system temp dir we always remove — never
        # into out_dir (which used to leave gradedslip_*/ litter in the figures folder).
        tmpdir = tempfile.mkdtemp(prefix="gradedslip_")
        try:
            for p, sub in _group_by_slip(manifest).items():
                sub_path = os.path.join(tmpdir, f"slip_{p:.3f}.json")
                with open(sub_path, "w") as f:
                    json.dump(sub, f)
                data = load_manifest(sub_path)
                if not data["raw"]:
                    print(f"[slip {p:.3f}] no runs parsed, skipping")
                    continue
                raw_by_slip[p] = data["raw"]
                evalsteps_by_slip[p] = data["eval_steps"]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    n_used = (n_submitted - len(dropped)) if n_submitted is not None else None

    # --- metrics per slip level (identical whether sourced from cache or runs) ---
    slips = []
    iqm_by_slip = {}      # slip -> {alg: (pt, lo, hi)}
    pimp_by_slip = {}     # slip -> P(mul > PER) (pt, lo, hi)
    steps_by_slip = {}    # slip -> {threshold: {alg: (median, iqr, n_censored)}}
    seeds_by_slip = {}    # slip -> {alg: n_seeds}

    for p in sorted(raw_by_slip):
        raw = raw_by_slip[p]
        fi = final_iqm(raw, reps=reps)
        pi = prob_improvement(raw, baseline=PER, reps=reps)
        st = {t: steps_to_threshold(raw, evalsteps_by_slip[p], t) for t in thresholds}

        slips.append(p)
        iqm_by_slip[p] = fi
        pimp_by_slip[p] = pi.get(MUL)
        steps_by_slip[p] = st
        seeds_by_slip[p] = {a: int(raw[a].shape[0]) for a in raw}

        tag = "PROBE" if p > NOISE_PEAK + 1e-9 else "     "
        print(f"[slip {p:.3f}  H={slip_entropy(p):.3f} {tag}] "
              f"n_seeds={seeds_by_slip[p]}")
        for a in ALG_ORDER:
            if a in fi:
                pt, lo, hi = fi[a]
                line = f"    {a:14s} IQM {pt:.3f} [{lo:.3f}, {hi:.3f}]"
                for t in thresholds:
                    med, iqr, cens = st[t][a]
                    med_txt = "never" if not np.isfinite(med) else f"{med:,.0f}"
                    line += f"   t{t:g}: {med_txt} ({cens} censored)"
                print(line)
        if MUL in fi and PER in fi:
            adv = fi[MUL][0] - fi[PER][0]
            pp = pimp_by_slip[p]
            pptxt = f"{pp[0]:.2f}" if pp else "n/a"
            print(f"    -> advantage(mul-PER) = {adv:+.3f}   P(mul>PER) = {pptxt}")

    slips = sorted(slips)
    main, probe = _split_main_probe(slips)

    # ---- Figure 1: final IQM vs slip, one line per algorithm ----
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for a in ALG_ORDER:
        xs = [p for p in main if a in iqm_by_slip[p]]
        if xs:
            ys = [iqm_by_slip[p][a][0] for p in xs]
            los = [iqm_by_slip[p][a][1] for p in xs]
            his = [iqm_by_slip[p][a][2] for p in xs]
            ax.plot(xs, ys, "-o", color=COLORS[a], label=a, lw=2, ms=4)
            ax.fill_between(xs, los, his, color=COLORS[a], alpha=0.15)
        pxs = [p for p in probe if a in iqm_by_slip[p]]
        if pxs:
            ax.plot(pxs, [iqm_by_slip[p][a][0] for p in pxs], "s--",
                    color=COLORS[a], lw=1.2, ms=5, alpha=0.75)
    if probe:
        ax.axvline(NOISE_PEAK, color="k", lw=0.8, ls=":")
        ax.text(NOISE_PEAK, 0.02, "  noise peak\n  (probe beyond)", fontsize=7,
                va="bottom", ha="left")
    ax.set_xlabel("slip probability  (0 = deterministic, 2/3 = max noise)")
    ax.set_ylabel("Final IQM win rate")
    ax.set_title("FrozenLake 8x8 — final IQM vs environment noise")
    ax.invert_xaxis()  # noise decreasing left->right so the CCE gap 'opens' rightward
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    f1 = os.path.join(out_dir, "fig_iqm_vs_slip.png")
    fig.tight_layout(); fig.savefig(f1, dpi=150); plt.close(fig)

    # ---- Figure 2: advantage + P(mul>PER) vs slip (tests Theorem 3) ----
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.3))

    def _adv(p):
        return iqm_by_slip[p][MUL][0] - iqm_by_slip[p][PER][0]

    has = [p for p in slips if MUL in iqm_by_slip[p] and PER in iqm_by_slip[p]]
    mh = [p for p in has if p <= NOISE_PEAK + 1e-9]
    ph = [p for p in has if p > NOISE_PEAK + 1e-9]
    axa.axhline(0, color="k", lw=0.8, ls="--")
    axa.plot(mh, [_adv(p) for p in mh], "-o", color=COLORS[MUL], lw=2, ms=4)
    if ph:
        axa.plot(ph, [_adv(p) for p in ph], "s--", color=COLORS[MUL], lw=1.2,
                 ms=6, alpha=0.75, label="probe (past noise peak)")
        axa.axvline(NOISE_PEAK, color="k", lw=0.8, ls=":")
        axa.legend(fontsize=8)
    axa.set_xlabel("slip probability")
    axa.set_ylabel("IQM(CCE-mul) - IQM(PER)")
    axa.set_title("Advantage vs noise\n(Thm 3: grows as slip falls)")
    axa.invert_xaxis(); axa.grid(alpha=0.3)

    px = [p for p in slips if pimp_by_slip.get(p)]
    mpx = [p for p in px if p <= NOISE_PEAK + 1e-9]
    ppx = [p for p in px if p > NOISE_PEAK + 1e-9]
    axb.axhline(0.5, color="k", lw=0.8, ls="--")
    if mpx:
        axb.plot(mpx, [pimp_by_slip[p][0] for p in mpx], "-o",
                 color=COLORS[MUL], lw=2, ms=4)
        axb.fill_between(mpx, [pimp_by_slip[p][1] for p in mpx],
                         [pimp_by_slip[p][2] for p in mpx],
                         color=COLORS[MUL], alpha=0.15)
    if ppx:
        axb.errorbar(ppx, [pimp_by_slip[p][0] for p in ppx],
                     yerr=[[pimp_by_slip[p][0] - pimp_by_slip[p][1] for p in ppx],
                           [pimp_by_slip[p][2] - pimp_by_slip[p][0] for p in ppx]],
                     fmt="s", color=COLORS[MUL], ms=6, alpha=0.75, capsize=3)
        axb.axvline(NOISE_PEAK, color="k", lw=0.8, ls=":")
    axb.set_ylim(0, 1)
    axb.set_xlabel("slip probability")
    axb.set_ylabel("P(CCE-mul > DQN+PER)")
    axb.set_title("Probability of improvement vs noise")
    axb.invert_xaxis(); axb.grid(alpha=0.3)
    f2 = os.path.join(out_dir, "fig_advantage_vs_slip.png")
    fig.tight_layout(); fig.savefig(f2, dpi=150); plt.close(fig)

    # ---- Figure 3: the same advantage against ACTUAL noise (entropy) ----
    # Theorem 3 is about noise, and slip stops tracking noise past p=2/3, so the
    # probe levels land back on the low-noise side here. If the advantage is
    # driven by noise they should lift; if it is driven by determinism they stay
    # flat. Probe points are drawn detached — the mapping p -> H is not injective.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.plot([slip_entropy(p) for p in mh], [_adv(p) for p in mh], "-o",
            color=COLORS[MUL], lw=2, ms=4, label="slip <= 2/3 (noise curve)")
    if ph:
        ax.plot([slip_entropy(p) for p in ph], [_adv(p) for p in ph], "s",
                color="#7f7f7f", ms=8, label="slip > 2/3 (probe: noise falls again)")
        for p in ph:
            ax.annotate(f"p={p:g}", (slip_entropy(p), _adv(p)),
                        textcoords="offset points", xytext=(6, 5), fontsize=7)
    ax.set_xlabel("environment noise — entropy of [p/2, 1-p, p/2]  (nats)")
    ax.set_ylabel("IQM(CCE-mul) - IQM(PER)")
    ax.set_title("Advantage vs actual noise\n(probe separates 'needs low noise' "
                 "from 'needs determinism')")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    f3 = os.path.join(out_dir, "fig_advantage_vs_noise.png")
    fig.tight_layout(); fig.savefig(f3, dpi=150); plt.close(fig)

    # ---- Figure 4: steps-to-threshold — speed under the saturation ceiling ----
    fig, axes = plt.subplots(1, len(thresholds),
                             figsize=(5.5 * len(thresholds), 4.3), squeeze=False)
    for ax, t in zip(axes[0], thresholds):
        for a in ALG_ORDER:
            xs, ys, cens, tots = [], [], [], []
            for p in slips:
                if a not in steps_by_slip[p][t]:
                    continue
                med, _, c = steps_by_slip[p][t][a]
                if np.isfinite(med):
                    xs.append(p); ys.append(med); cens.append(c)
                    tots.append(seeds_by_slip[p].get(a, 0))
            if xs:
                ax.plot(xs, ys, "-o", color=COLORS[a], label=a, lw=2, ms=4)
                # A point where most seeds never got there is not a fair speed
                # comparison; mark it rather than dropping it silently.
                for x, y, c, n_tot in zip(xs, ys, cens, tots):
                    if n_tot and c > n_tot / 2:
                        ax.plot([x], [y], "x", color="k", ms=8, mew=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("slip probability")
        ax.set_ylabel(f"median env steps to win rate >= {t:g}")
        ax.set_title(f"Speed to {t:g} win rate  (lower = faster)\n"
                     "x = majority of seeds never reached it")
        ax.invert_xaxis(); ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    f4 = os.path.join(out_dir, "fig_steps_to_threshold.png")
    fig.tight_layout(); fig.savefig(f4, dpi=150); plt.close(fig)

    # ---- Figure 5: IQM learning curves, one panel per slip (paper Fig-4 style) ----
    # Watch the training dynamics change as noise rises. Each alg's raw curves are
    # forward-filled to a common length so early-stopped winners hold at their
    # final value (same convention as the paper's Fig 4).
    curve_slips = [p for p in slips if p in raw_by_slip]
    f5 = None
    if curve_slips:
        ncol = min(3, len(curve_slips))
        nrow = int(np.ceil(len(curve_slips) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow),
                                 squeeze=False)
        flat = [ax for row in axes for ax in row]
        for ax, p in zip(flat, curve_slips):
            raw_p = raw_by_slip[p]
            T = max(a.shape[2] for a in raw_p.values())
            padded = {}
            for k, arr in raw_p.items():
                if arr.shape[2] < T:
                    fill = np.repeat(arr[:, :, -1:], T - arr.shape[2], axis=2)
                    arr = np.concatenate([arr, fill], axis=2)
                padded[k] = arr
            print(f"  learning curves: slip {p:.3f} (T={T} checkpoints)")
            curves = iqm_curves(padded, reps=curve_reps)
            steps_map = evalsteps_by_slip[p]
            xref = max(steps_map.values(), key=lambda s: len(s))
            for a in ALG_ORDER:
                if a not in curves:
                    continue
                iqm_v, lo, hi = curves[a]
                x = xref[:len(iqm_v)]
                ax.plot(x, iqm_v, color=COLORS[a], lw=2, label=a)
                ax.fill_between(x, lo, hi, color=COLORS[a], alpha=0.15)
            tag = "  [PROBE]" if p > NOISE_PEAK + 1e-9 else ""
            ax.set_title(f"slip {p:g}  (noise H={slip_entropy(p):.2f}){tag}")
            ax.set_xlabel("environment steps")
            ax.set_ylabel("IQM win rate")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
        for ax in flat[len(curve_slips):]:
            ax.axis("off")
        handles, labels = flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   fontsize=9, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle("IQM win-rate learning curves by slip level "
                     "(FrozenLake 8x8, dead runs dropped)", y=1.0, fontsize=13)
        f5 = os.path.join(out_dir, "fig_learning_curves_by_slip.png")
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        fig.savefig(f5, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- summary json ----
    summary = {
        "slips": slips,
        "noise_entropy": {f"{p:.3f}": slip_entropy(p) for p in slips},
        "noise_peak_slip": NOISE_PEAK,
        "n_submitted": n_submitted,
        "n_used": n_used,
        "dropped_runs": dropped,
        "seeds_per_cell": {f"{p:.3f}": seeds_by_slip[p] for p in slips},
        "final_iqm": {f"{p:.3f}": iqm_by_slip[p] for p in slips},
        "prob_improve_mul_vs_per": {f"{p:.3f}": pimp_by_slip.get(p) for p in slips},
        "steps_to_threshold": {
            f"{p:.3f}": {f"{t:g}": steps_by_slip[p][t] for t in thresholds}
            for p in slips
        },
    }
    with open(os.path.join(out_dir, "graded_slip_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    wrote = [f1, f2, f3, f4]
    if f5:
        wrote.append(f5)
    wrote.append(os.path.join(out_dir, "graded_slip_summary.json"))
    print("\nWrote:\n  " + "\n  ".join(wrote))
    return summary


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="job_id->config manifest; reads the raw runs")
    src.add_argument("--from-cache", dest="from_cache_path",
                     help="paper/repro/cache/claim2_graded_slip.npz; rebuilds figures "
                          "WITHOUT the raw runs")
    ap.add_argument("--out", default="docs/figures/graded_slip")
    ap.add_argument("--reps", type=int, default=50000)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.9],
                    help="win-rate thresholds for the steps-to-threshold figure")
    ap.add_argument("--keep-incomplete", action="store_true",
                    help="do NOT drop runs that died mid-training (reproduces the "
                         "2026-08-03 numbers, which forward-filled killed seeds)")
    ap.add_argument("--curve-reps", type=int, default=5000,
                    help="bootstrap reps for the per-slip learning-curve panels "
                         "(cheaper than --reps since it runs at every checkpoint)")
    args = ap.parse_args()
    source = args.from_cache_path or args.manifest
    analyze(source, args.out, reps=args.reps,
            thresholds=tuple(args.thresholds),
            drop_incomplete=not args.keep_incomplete,
            curve_reps=args.curve_reps,
            from_cache=bool(args.from_cache_path))


if __name__ == "__main__":
    main()
