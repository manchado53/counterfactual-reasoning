"""
Step 3 verdict — did prioritising by measured u beat uniform replay?

Reads the arms submitted by `utility_sampler.py`, compares learning speed and
final performance, and states the verdict plainly.

Two comparisons, and they answer different questions:

  utility-b1.0  vs  uniform        Is u real? If a sampler drawing proportional
                                   to measured replay utility cannot beat uniform,
                                   then u is a number without a claim to the name,
                                   and the covariance results in steps 2, 4 and 5
                                   are measurements of nothing.

  utility-b0.25 vs  utility-b1.0   What does the deployed exponent cost? Step 1
                                   showed beta=0.25 caps the priority spread at
                                   3.17x. If B beats uniform and C does not, the
                                   shaping is destroying a signal that works.

u is an oracle quantity requiring exact Q*, so arm B is a ceiling, not a
deployable method: the best any utility-tracking priority could achieve.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

LOG_DIR = os.path.expanduser("~/theorem3_logs")
ARM_ORDER = ["uniform", "utility-b1.0", "utility-b0.25"]


def eval_curve(metrics_log):
    """(episode, win_rate) from a run's metrics.log body."""
    rows = []
    for line in open(metrics_log):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if not p or not p[0].isdigit():
            continue
        try:
            rows.append((int(p[0]), float(p[3].rstrip("%")) / 100.0))
        except (IndexError, ValueError):
            continue
    return rows


def collect(log_dir=LOG_DIR):
    """One record per finished job, from the RESULT line each run prints."""
    out = []
    for f in sorted(glob.glob(os.path.join(log_dir, "*.out"))):
        m = re.search(r"^RESULT (\{.*\})\s*$", open(f).read(), re.M)
        if not m:
            continue
        rec = json.loads(m.group(1))
        log = os.path.join(rec["run_dir"], "metrics.log")
        if not os.path.exists(log):
            continue
        curve = eval_curve(log)
        if not curve:
            continue
        ep = np.array([c[0] for c in curve])
        wr = np.array([c[1] for c in curve])
        tail = max(1, len(wr) // 10)
        rec["final_wr"] = float(np.mean(wr[-tail:]))
        rec["best_wr"] = float(wr.max())
        rec["auc"] = float(np.trapezoid(wr, ep) / (ep[-1] - ep[0])) if len(ep) > 1 else 0.0
        for thr in (0.5, 0.9):
            hit = np.where(wr >= thr)[0]
            rec[f"ep_to_{thr}"] = int(ep[hit[0]]) if len(hit) else None
        rec["job"] = os.path.basename(f).split(".")[0]
        out.append(rec)
    return out


def _fmt_speed(vals):
    got = [v for v in vals if v is not None]
    if not got:
        return "never"
    return f"{int(np.median(got))} ({len(got)}/{len(vals)})"


def report(records):
    by_arm = {}
    for r in records:
        by_arm.setdefault(r["arm"], []).append(r)

    print(f"{'arm':>14} {'n':>3} {'final win rate':>22} {'AUC':>16} "
          f"{'eps to 0.5':>16} {'eps to 0.9':>16}")
    stats = {}
    for arm in ARM_ORDER:
        rs = by_arm.get(arm, [])
        if not rs:
            print(f"{arm:>14}   0   (no finished runs yet)")
            continue
        fw = np.array([r["final_wr"] for r in rs])
        auc = np.array([r["auc"] for r in rs])
        stats[arm] = dict(final=fw, auc=auc)
        print(f"{arm:>14} {len(rs):>3} "
              f"{np.median(fw):>10.3f} [{fw.min():.2f},{fw.max():.2f}] "
              f"{np.median(auc):>10.3f} [{auc.min():.2f},{auc.max():.2f}] "
              f"{_fmt_speed([r['ep_to_0.5'] for r in rs]):>16} "
              f"{_fmt_speed([r['ep_to_0.9'] for r in rs]):>16}")

    print()
    if "uniform" in stats and "utility-b1.0" in stats:
        _verdict("Is u real?", "utility-b1.0", "uniform", stats,
                 pass_msg="u carries real replay utility. The covariance results "
                          "in steps 2, 4 and 5 measure something meaningful.",
                 fail_msg="u does NOT behave like utility. Prioritising by it is "
                          "no better than uniform, so the covariances computed "
                          "against it in steps 2, 4 and 5 are not evidence about "
                          "which priority signal is better. That would invalidate "
                          "the interpretation of this whole experiment.")
    if "utility-b1.0" in stats and "utility-b0.25" in stats:
        _verdict("What does beta=0.25 cost?", "utility-b1.0", "utility-b0.25", stats,
                 pass_msg="the deployed exponent measurably weakens a working "
                          "signal -- direct support for revisiting beta.",
                 fail_msg="no measurable difference between the exponents here.")


def _verdict(question, arm_a, arm_b, stats, pass_msg, fail_msg):
    a, b = stats[arm_a], stats[arm_b]
    try:
        from scipy.stats import mannwhitneyu
        p_final = mannwhitneyu(a["final"], b["final"], alternative="greater").pvalue
        p_auc = mannwhitneyu(a["auc"], b["auc"], alternative="greater").pvalue
    except Exception:
        p_final = p_auc = float("nan")
    d_final = float(np.median(a["final"]) - np.median(b["final"]))
    d_auc = float(np.median(a["auc"]) - np.median(b["auc"]))
    better = d_final > 0 and d_auc > 0
    print(f"--- {question}   ({arm_a} vs {arm_b}) ---")
    print(f"    median final win rate  {np.median(a['final']):.3f} vs "
          f"{np.median(b['final']):.3f}   (diff {d_final:+.3f}, p={p_final:.3f})")
    print(f"    median AUC             {np.median(a['auc']):.3f} vs "
          f"{np.median(b['auc']):.3f}   (diff {d_auc:+.3f}, p={p_auc:.3f})")
    print(f"    -> {pass_msg if better else fail_msg}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", default=LOG_DIR)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    recs = collect(args.log_dir)
    print(f"{len(recs)} finished run(s) in {args.log_dir}\n")
    if recs:
        report(recs)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(recs, f, indent=1, default=float)
        print(f"wrote {args.json_out}")
