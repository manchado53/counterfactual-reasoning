"""Top-level entry point for Claim 2 analysis.

Usage:
    python -m counterfactual_rl.analysis.claim2.run_analysis \\
        --manifest path/to/manifest.json \\
        --env smax_3m \\
        --threshold 0.60 \\
        --out docs/figures/real/

Supported --env values:
    smax_3m     threshold 0.60
    smax_8m     threshold 0.55
    chess       threshold TBD (set after pilot)
    frozen_lake threshold TBD (set after pilot)

The manifest JSON maps SLURM job_id → config override dict,
as produced by run_experiments.py in each agent directory.
"""

import argparse
import os

import numpy as np

from .parse_logs import load_manifest
from .compute_metrics import compute_all
from .plot_figures import (
    fig1_iqm_curves,
    fig2_final_iqm,
    fig3_steps_to_threshold,
    fig4_prob_improvement,
    fig5a_wallclock_breakdown,
    fig5b_wallclock_to_threshold,
    fig5c_component_breakdown,
    fig_length_curves,
    fig_allies_curves,
    fig_wdl_table,
)

ENV_THRESHOLDS = {
    'smax_3m':     0.60,
    'smax_8m':     0.55,
    'chess':       None,       # set after pilot
    'frozen_lake': None,       # set after pilot
}


def main():
    parser = argparse.ArgumentParser(description='Claim 2 analysis pipeline')
    parser.add_argument('--manifest', required=True, help='Path to manifest JSON')
    parser.add_argument('--env', required=True, choices=list(ENV_THRESHOLDS.keys()),
                        help='Environment name')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override pre-registered threshold (use for pilot-derived values)')
    parser.add_argument('--out', default='docs/figures/real',
                        help='Output directory for figures')
    parser.add_argument('--reps', type=int, default=50000,
                        help='Bootstrap resamples (default 50000)')
    args = parser.parse_args()

    threshold = args.threshold if args.threshold is not None else ENV_THRESHOLDS[args.env]
    if threshold is None:
        raise ValueError(
            f"No threshold registered for '{args.env}'. "
            f"Run pilot first, then pass --threshold VALUE."
        )

    os.makedirs(args.out, exist_ok=True)
    env_label = args.env.replace('_', ' ').title()

    print(f"Loading manifest: {args.manifest}")
    data = load_manifest(args.manifest)
    raw = data['raw']
    raw_length = data.get('raw_length')
    eval_steps = data['eval_steps']
    run_dirs = data.get('run_dirs')

    print(f"Algorithms: {list(raw.keys())}")
    for alg, arr in raw.items():
        print(f"  {alg}: {arr.shape[0]} seeds, {arr.shape[2]} checkpoints")

    print(f"\nComputing metrics (threshold={threshold}, reps={args.reps})...")
    results = compute_all(
        raw, eval_steps, threshold,
        raw_length=raw_length,
        run_dirs=run_dirs,
        reps=args.reps,
    )

    # ── Print steps-to-threshold table ────────────────────────────────────────
    print(f"\nSteps-to-threshold ({env_label}, threshold={threshold:.0%}):")
    for alg, (med, iqr, n_cens) in results['steps_thresh'].items():
        med_str = f"{med/1000:.1f}k" if np.isfinite(med) else "∞"
        print(f"  {alg:<22} median={med_str}  IQR={iqr/1000:.1f}k  censored={n_cens}")

    print(f"\nFinal IQM ({env_label}):")
    for alg, (pt, lo, hi) in results['final_iqm'].items():
        print(f"  {alg:<22} {pt:.3f}  [{lo:.3f}, {hi:.3f}]")

    print(f"\nP(alg > DQN+PER) ({env_label}):")
    for alg, (pt, lo, hi) in results['prob_improve'].items():
        print(f"  {alg:<22} {pt:.3f}  [{lo:.3f}, {hi:.3f}]")

    # ── Figures ────────────────────────────────────────────────────────────────
    env_name = env_label

    fig1_iqm_curves(
        {env_name: results['iqm_curves']},
        {env_name: next(iter(eval_steps.values()))},
        {env_name: threshold},
        os.path.join(args.out, f'fig1_iqm_{args.env}.png'),
    )

    fig2_final_iqm(
        {env_name: results['final_iqm']},
        os.path.join(args.out, f'fig2_final_iqm_{args.env}.png'),
    )

    fig3_steps_to_threshold(
        {env_name: results['steps_thresh']},
        os.path.join(args.out, f'fig3_steps_thresh_{args.env}.png'),
    )

    fig4_prob_improvement(
        {env_name: results['prob_improve']},
        os.path.join(args.out, f'fig4_prob_improve_{args.env}.png'),
    )

    # Allies alive (SMAX only)
    if data['env_type'] == 'smax' and data['raw_allies']:
        avg_allies_iqm = {}
        for alg, arr in data['raw_allies'].items():
            avg_allies_iqm[alg] = arr[:, 0, :].mean(axis=0)  # simple mean across seeds
        fig_allies_curves(
            {env_name: avg_allies_iqm},
            {env_name: next(iter(eval_steps.values()))},
            os.path.join(args.out, f'fig_allies_{args.env}.png'),
        )

    # W/D/L table (chess only)
    if data['env_type'] == 'chess' and data['raw_wdl']:
        wdl_at_conv = {}
        for alg, arr in data['raw_wdl'].items():
            cutoff = int(0.9 * arr.shape[1])
            mean_wdl = arr[:, cutoff:, :].mean(axis=(0, 1))
            wdl_at_conv[alg] = (float(mean_wdl[0]), float(mean_wdl[1]), float(mean_wdl[2]))
        fig_wdl_table(
            wdl_at_conv,
            env_name,
            os.path.join(args.out, f'fig_wdl_{args.env}.png'),
        )

    if results['wallclock']:
        fig5a_wallclock_breakdown(
            {env_name: results['wallclock']},
            os.path.join(args.out, f'fig5a_wallclock_{args.env}.png'),
        )

    if results.get('wallclock_thresh'):
        fig5b_wallclock_to_threshold(
            {env_name: results['wallclock_thresh']},
            os.path.join(args.out, f'fig5b_wallclock_thresh_{args.env}.png'),
        )

    if results.get('wallclock_components'):
        fig5c_component_breakdown(
            {env_name: results['wallclock_components']},
            os.path.join(args.out, f'fig5c_components_{args.env}.png'),
        )

    if results.get('iqm_length_curves'):
        fig_length_curves(
            {env_name: results['iqm_length_curves']},
            {env_name: next(iter(eval_steps.values()))},
            os.path.join(args.out, f'fig_length_{args.env}.png'),
        )

    print(f"\nAll figures saved to {args.out}")


if __name__ == '__main__':
    main()
