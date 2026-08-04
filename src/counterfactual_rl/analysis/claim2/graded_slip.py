"""Graded-stochasticity analysis for FrozenLake.

Groups a `claim2_graded_slip` manifest by `slip_prob`, computes the paper's
final-IQM and P(improvement) metrics per slip level (reusing compute_metrics),
and plots how CCE+TD(mul)'s advantage over DQN+PER changes with environment
noise. Theorem 3 predicts the advantage GROWS as slip_prob falls.

Run (as worktree code):
    PYTHONPATH=<worktree>/src python -m counterfactual_rl.analysis.claim2.graded_slip \
        --manifest <path>/claim2_graded_slip_*.json --out <out_dir>
"""

import argparse
import json
import os
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .parse_logs import load_manifest
from .compute_metrics import final_iqm, prob_improvement

MUL = "CCE+TD (mul)"
PER = "DQN+PER"
COLORS = {
    "DQN-Uniform": "#888888",
    "DQN+PER": "#1f77b4",
    "DQN+CCE-only": "#ff7f0e",
    "CCE+TD (mul)": "#d62728",
}


def _group_by_slip(manifest_path):
    """Return {slip_prob: {job_id: cfg}} from a manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    groups = {}
    for job_id, cfg in manifest.items():
        p = float(cfg.get("slip_prob", 2.0 / 3.0))
        groups.setdefault(p, {})[job_id] = cfg
    return dict(sorted(groups.items()))


def analyze(manifest_path, out_dir, reps=50000):
    os.makedirs(out_dir, exist_ok=True)
    groups = _group_by_slip(manifest_path)
    tmpdir = tempfile.mkdtemp(prefix="gradedslip_", dir=out_dir)

    slips = []
    iqm_by_slip = {}      # slip -> {alg: (pt, lo, hi)}
    pimp_by_slip = {}     # slip -> P(mul > PER) (pt, lo, hi)

    for p, sub in groups.items():
        sub_path = os.path.join(tmpdir, f"slip_{p:.3f}.json")
        with open(sub_path, "w") as f:
            json.dump(sub, f)
        data = load_manifest(sub_path)
        raw = data["raw"]
        if not raw:
            print(f"[slip {p:.3f}] no runs parsed, skipping")
            continue
        fi = final_iqm(raw, reps=reps)
        pi = prob_improvement(raw, baseline=PER, reps=reps)
        slips.append(p)
        iqm_by_slip[p] = fi
        pimp_by_slip[p] = pi.get(MUL)
        seeds = {a: raw[a].shape[0] for a in raw}
        print(f"[slip {p:.3f}] n_seeds={seeds}")
        for a in ["DQN-Uniform", PER, "DQN+CCE-only", MUL]:
            if a in fi:
                pt, lo, hi = fi[a]
                print(f"    {a:14s} IQM {pt:.3f} [{lo:.3f}, {hi:.3f}]")
        if MUL in fi and PER in fi:
            adv = fi[MUL][0] - fi[PER][0]
            pp = pimp_by_slip[p]
            pptxt = f"{pp[0]:.2f}" if pp else "n/a"
            print(f"    -> advantage(mul-PER) = {adv:+.3f}   P(mul>PER) = {pptxt}")

    slips = sorted(slips)

    # ---- Figure 1: final IQM vs slip, one line per algorithm ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in ["DQN-Uniform", PER, "DQN+CCE-only", MUL]:
        ys, los, his = [], [], []
        xs = []
        for p in slips:
            if a in iqm_by_slip[p]:
                pt, lo, hi = iqm_by_slip[p][a]
                xs.append(p); ys.append(pt); los.append(lo); his.append(hi)
        if xs:
            ax.plot(xs, ys, "-o", color=COLORS[a], label=a, lw=2)
            ax.fill_between(xs, los, his, color=COLORS[a], alpha=0.15)
    ax.set_xlabel("slip probability  (0 = deterministic, 0.666 = full slip)")
    ax.set_ylabel("Final IQM win rate")
    ax.set_title("FrozenLake 8x8 — final IQM vs environment noise")
    ax.invert_xaxis()  # noise decreasing left->right so the CCE gap 'opens' rightward
    ax.grid(alpha=0.3); ax.legend()
    f1 = os.path.join(out_dir, "fig_iqm_vs_slip.png")
    fig.tight_layout(); fig.savefig(f1, dpi=150); plt.close(fig)

    # ---- Figure 2: advantage + P(mul>PER) vs slip (tests Theorem 3) ----
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.3))
    adv = [iqm_by_slip[p][MUL][0] - iqm_by_slip[p][PER][0]
           for p in slips if MUL in iqm_by_slip[p] and PER in iqm_by_slip[p]]
    advx = [p for p in slips if MUL in iqm_by_slip[p] and PER in iqm_by_slip[p]]
    axa.axhline(0, color="k", lw=0.8, ls="--")
    axa.plot(advx, adv, "-o", color=COLORS[MUL], lw=2)
    axa.set_xlabel("slip probability"); axa.set_ylabel("IQM(CCE-mul) - IQM(PER)")
    axa.set_title("Advantage vs noise\n(Thm 3: grows as slip falls)")
    axa.invert_xaxis(); axa.grid(alpha=0.3)

    px = [p for p in slips if pimp_by_slip.get(p)]
    pp = [pimp_by_slip[p][0] for p in px]
    plo = [pimp_by_slip[p][1] for p in px]
    phi = [pimp_by_slip[p][2] for p in px]
    axb.axhline(0.5, color="k", lw=0.8, ls="--")
    axb.plot(px, pp, "-o", color=COLORS[MUL], lw=2)
    axb.fill_between(px, plo, phi, color=COLORS[MUL], alpha=0.15)
    axb.set_ylim(0, 1); axb.set_xlabel("slip probability")
    axb.set_ylabel("P(CCE-mul > DQN+PER)")
    axb.set_title("Probability of improvement vs noise")
    axb.invert_xaxis(); axb.grid(alpha=0.3)
    f2 = os.path.join(out_dir, "fig_advantage_vs_slip.png")
    fig.tight_layout(); fig.savefig(f2, dpi=150); plt.close(fig)

    # ---- summary json ----
    summary = {
        "slips": slips,
        "final_iqm": {f"{p:.3f}": iqm_by_slip[p] for p in slips},
        "prob_improve_mul_vs_per": {f"{p:.3f}": pimp_by_slip.get(p) for p in slips},
    }
    with open(os.path.join(out_dir, "graded_slip_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote:\n  {f1}\n  {f2}\n  {os.path.join(out_dir, 'graded_slip_summary.json')}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="docs/figures/graded_slip")
    ap.add_argument("--reps", type=int, default=50000)
    args = ap.parse_args()
    analyze(args.manifest, args.out, reps=args.reps)


if __name__ == "__main__":
    main()
