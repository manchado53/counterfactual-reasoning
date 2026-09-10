"""Check the smoke gates for the greedy-coverage sweep (plans/jaxnav-greedy-coverage.md).

Every number the launch decision rests on is DERIVED until this runs. In particular
"~17% coverage" is 0.86% x 20, and the same derivation already reads ~2x high against
the measured 0.86%. scored_frac in ess.jsonl is the truth.

    python -m counterfactual_rl.analysis.claim2.greedy_smoke_gates   (or run directly)
"""
import json
import os
import subprocess
import sys

import numpy as np

WT = ("/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/"
      ".claude/worktrees/research+cce-robotics-transfer")
RUNS = os.path.join(WT, "src/counterfactual_rl/agents/jax_nav/runs")
MANIFEST = os.path.join(WT, "docs/figures/real/claim2/jaxnav/data/manifest_greedy_smoke.json")

# H=60 baseline for the terminal-reason mix, measured over 73 instrumented runs
GOAL_AT_H60 = 9.2


def resolve(manifest):
    """SLURM array tasks write output under JobIDRaw, not <array>_<task>."""
    arrays = sorted({v["array_id"] for v in manifest.values()})
    out = subprocess.run(["sacct", "-j", ",".join(arrays), "-X", "-n", "-P",
                          "--format=JobID,JobIDRaw,State"],
                         capture_output=True, text=True).stdout
    raw, state = {}, {}
    for line in out.strip().splitlines():
        p = line.split("|")
        if len(p) >= 3:
            raw[p[0]], state[p[0]] = p[1], p[2].split()[0]
    return raw, state


def last_ess(rid):
    p = os.path.join(RUNS, rid, "ess.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return rows[-1] if rows else None


def main():
    manifest = json.load(open(MANIFEST))
    raw, state = resolve(manifest)
    print(f"{'arm':<18}{'state':<11}{'scored_frac':>12}{'expected':>10}"
          f"{'zero%':>8}{'goal%':>8}{'alive%':>8}{'crash%':>8}")
    print("-" * 84)
    failures = []
    for key in sorted(manifest):
        v = manifest[key]
        jid = v["job_id"]
        st = state.get(jid, "?")
        rid = raw.get(jid, jid)
        d = last_ess(rid)
        arm = v["arm"]
        exp = 0.86 * v["n_score_sample"] / 64.0
        if d is None:
            print(f"{arm:<18}{st:<11}{'no ess.jsonl':>12}{exp:>9.2f}%")
            failures.append(f"{arm}: no ess.jsonl (state {st})")
            continue
        sf = 100 * d.get("scored_frac", float("nan"))
        n = max(1, d.get("scored_n") or d.get("size") or 1)
        zero = 100.0 * d.get("score_n_zero", 0) / n
        goal = 100 * d.get("rollout_goal_frac", float("nan"))
        alive = 100 * d.get("rollout_alive_frac", float("nan"))
        crash = 100 * d.get("rollout_crash_frac", float("nan"))
        print(f"{arm:<18}{st:<11}{sf:>11.2f}%{exp:>9.2f}%"
              f"{zero:>7.1f}%{goal:>7.1f}%{alive:>7.1f}%{crash:>7.1f}%")

        # gates
        if st not in ("COMPLETED", "RUNNING"):
            failures.append(f"{arm}: state {st}")
        if not np.isfinite(sf) or sf <= 0:
            failures.append(f"{arm}: scored_frac not positive")
        if zero > 95:
            failures.append(f"{arm}: {zero:.0f}% of scores are exactly 0 -> degenerate")
        if arm.startswith("greedy_full"):
            if np.isfinite(alive) and alive > 1.0:
                failures.append(f"{arm}: rollout_alive_frac {alive:.1f}% > 1% "
                                f"-- H=200 does NOT guarantee termination. KILL THE ARM.")
            # NOT a goal-rate comparison: the smoke runs 5,000 episodes at epsilon ~0.99
            # while the 9.2% H=60 figure comes from fully-trained 250k runs, so the two
            # are at different training stages and cannot be compared. The structural
            # test is the zero fraction: under sparse reward a rollout pays only on GOAL,
            # so without bootstrap the return is binary and ~90% of rollouts return 0.0
            # at ANY horizon -- running to the end turns "alive" into "timeout", which
            # pays the same 0. Measured 88.3% / 89.6% zeros on 2026-09-09.
            if zero > 50:
                failures.append(f"{arm}: {zero:.0f}% of measured scores are exactly 0 "
                                f"-- un-bootstrapped return is binary under sparse "
                                f"reward. KILL THE ARM.")

    print()
    if failures:
        print("GATE FAILURES:")
        for f in failures:
            print(f"   - {f}")
    else:
        print("ALL GATES PASS.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
