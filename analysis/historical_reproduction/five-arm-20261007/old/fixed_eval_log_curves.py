"""Render a reproducible JaxNav learning-curve PNG directly from Rosie logs.

Example:
  PYTHONPATH=src python analysis/claim2/fixed_eval_log_curves.py --array 277733 --array 278035

The script reads each ``jaxnav_logs/<array>_<task>.out`` file, finds the run
directory recorded by Slurm, then reads that run's ``eval.jsonl``.  It refuses
to mix configurations with different environment or fixed-evaluation settings.
It never forward-fills a seed after its last logged evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = Path.home() / "jaxnav_logs"
DEFAULT_OUT_DIR = ROOT / "docs" / "figures" / "real" / "claim2" / "jaxnav"
RUN_DIR_RE = re.compile(r"Run dir: (.+)")
CONFIG_KEYS = (
    "map_id", "map_size", "fill", "goal_radius", "max_steps", "sparse_reward",
    "goal_rew", "coll_rew", "epsilon_decay_episodes", "n_episodes",
    "eval_interval", "eval_episodes", "fixed_eval_seed", "buffer_capacity",
)
ARM_NAMES = {
    "dqn-uniform": "Uniform",
    "dqn": "PER",
}
ARM_ORDER = ("Uniform", "PER", "CCE Mul", "CCE Only", "CCE Add 50/50")
COLORS = {
    "Uniform": "#377eb8", "PER": "#ff7f00", "CCE Mul": "#984ea3",
    "CCE Only": "#4daf4a", "CCE Add 50/50": "#e41a1c",
}


def _arm_name(config: dict) -> str:
    algorithm = config.get("algorithm")
    if algorithm in ARM_NAMES:
        return ARM_NAMES[algorithm]
    if algorithm != "consequence-dqn":
        raise ValueError(f"unknown algorithm {algorithm!r}")
    if config.get("priority_mixing") == "multiplicative":
        return "CCE Mul"
    if config.get("priority_mixing") == "additive":
        mu = float(config.get("mu", 0.5))
        if mu == 1.0:
            return "CCE Only"
        if mu == 0.5:
            return "CCE Add 50/50"
        return f"CCE Add mu={mu:g}"
    raise ValueError(f"unknown CCE mixing mode {config.get('priority_mixing')!r}")


def _load_task(log_file: Path) -> tuple[str, int, dict, list[dict], Path]:
    """Return arm, seed, verified config, eval rows, and run directory."""
    text = log_file.read_text()
    config_line = next((line for line in text.splitlines() if line.startswith("CONFIG: ")), None)
    if config_line is None:
        raise ValueError(f"{log_file}: missing CONFIG line")
    config = json.loads(config_line.removeprefix("CONFIG: "))
    match = RUN_DIR_RE.search(text)
    if match is None:
        raise ValueError(f"{log_file}: missing Run dir line")
    run_dir = Path(match.group(1).strip())
    eval_file = run_dir / "eval.jsonl"
    if not eval_file.exists():
        raise ValueError(f"{log_file}: missing {eval_file}")
    rows = [json.loads(line) for line in eval_file.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{eval_file}: no evaluation rows")
    return _arm_name(config), int(config["seed"]), config, rows, run_dir


def _check_configs(records: list[tuple[str, int, dict, list[dict], Path]]) -> None:
    """Reject mixed environments or non-paired evaluation protocols."""
    reference = records[0][2]
    for _, _, config, _, source in records[1:]:
        changed = [key for key in CONFIG_KEYS if config.get(key) != reference.get(key)]
        if changed:
            detail = ", ".join(f"{key}: {reference.get(key)!r} != {config.get(key)!r}" for key in changed)
            raise ValueError(f"configuration mismatch in {source}: {detail}")
    if reference.get("fixed_eval_seed") is None:
        raise ValueError("this figure requires fixed_eval_seed so vertical comparisons are paired")


def _binned_summary(rows_by_seed: dict[int, list[dict]], metric: str, bin_width: int,
                    minimum_n: int):
    """Mean inside each seed/bin, then mean, median, and SEM across seeds."""
    maximum = max(row["episode"] for rows in rows_by_seed.values() for row in rows)
    centers = np.arange(bin_width // 2, maximum + bin_width, bin_width)
    out = []
    for center in centers:
        values = []
        lo, hi = center - bin_width // 2, center + bin_width // 2
        for rows in rows_by_seed.values():
            values_here = [row[metric] for row in rows
                           if lo < row["episode"] <= hi and np.isfinite(row.get(metric, np.nan))]
            if values_here:
                values.append(float(np.mean(values_here)))
        if len(values) >= minimum_n:
            # SEM is descriptive here: n=5 is too small for a strong
            # confidence-interval claim. The individual seed lines stay visible.
            sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            out.append((center, np.mean(values), np.median(values), sem, len(values)))
    return np.asarray(out, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array", required=True, action="append", help="Slurm array ID; repeat for matched arrays")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--out", type=Path, default=None, help="PNG path; default is docs/figures/real/claim2/jaxnav")
    parser.add_argument("--bin-width", type=int, default=5_000)
    parser.add_argument("--length-bin-width", type=int, default=1_000,
                        help="Goal-reaching-length summary bin width (default: 1,000)")
    parser.add_argument("--minimum-seeds", type=int, default=3,
                        help="Do not draw a summary bin with fewer seeds (default: 3)")
    args = parser.parse_args()

    log_files = [path for array_id in args.array for path in sorted(args.log_dir.glob(f"{array_id}_*.out"))]
    if not log_files:
        raise SystemExit(f"No logs found for arrays: {', '.join(args.array)}")
    records = [_load_task(path) for path in log_files]
    _check_configs(records)

    by_arm: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    source_ids: list[str] = []
    for arm, seed, _, rows, run_dir in records:
        if seed in by_arm[arm]:
            raise ValueError(f"duplicate {arm} seed {seed}")
        by_arm[arm][seed] = rows
        source_ids.append(run_dir.name)

    fig, (ax, length_ax) = plt.subplots(2, 1, figsize=(12.8, 9.4), dpi=180, sharex=True,
                                        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.10})
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.16)
    for arm in ARM_ORDER:
        seeds = by_arm.get(arm, {})
        if not seeds:
            continue
        color = COLORS[arm]
        for seed, rows in sorted(seeds.items()):
            ax.plot([r["episode"] for r in rows], [r["win_rate"] for r in rows], color=color,
                    alpha=0.18, lw=0.65, zorder=1)
        summary = _binned_summary(seeds, "win_rate", args.bin_width, args.minimum_seeds)
        if summary.size:
            x, mean, median, sem, n = summary.T
            ax.fill_between(x, np.clip(mean - sem, 0, 1), np.clip(mean + sem, 0, 1),
                            color=color, alpha=0.15, lw=0, zorder=2)
            ax.plot(x, mean, color=color, lw=2.6, label=f"{arm} mean ({len(seeds)} seeds)", zorder=3)
            ax.plot(x, median, color=color, lw=1.8, ls="--", zorder=3)
            ax.scatter(x[-1], mean[-1], color=color, s=24, zorder=4)

        for seed, rows in sorted(seeds.items()):
            x_seed = [r["episode"] for r in rows if np.isfinite(r.get("avg_length_goal", np.nan))]
            y_seed = [r["avg_length_goal"] for r in rows if np.isfinite(r.get("avg_length_goal", np.nan))]
            length_ax.plot(x_seed, y_seed, color=color, alpha=0.18, lw=0.65, zorder=1)
        length_summary = _binned_summary(seeds, "avg_length_goal", args.length_bin_width,
                                         args.minimum_seeds)
        if length_summary.size:
            x, mean, median, sem, n = length_summary.T
            length_ax.fill_between(x, np.clip(mean - sem, 0, None), mean + sem,
                                   color=color, alpha=0.15, lw=0, zorder=2)
            length_ax.plot(x, mean, color=color, lw=2.6, zorder=3)
            length_ax.plot(x, median, color=color, lw=1.8, ls="--", zorder=3)
            length_ax.scatter(x[-1], mean[-1], color=color, s=24, zorder=4)

    config = records[0][2]
    finish = int(config["n_episodes"])
    decay = int(config["epsilon_decay_episodes"])
    ax.axvline(decay, color="#555555", lw=1.0, ls="--", zorder=0)
    ax.text(decay + finish * 0.012, 0.97, "epsilon floor", color="#555555", va="top", fontsize=9)
    ax.set(xlim=(0, finish), ylim=(0, 1), ylabel="goal reach rate")
    ax.grid(alpha=0.22, lw=0.7)
    ax.legend(frameon=False, loc="lower right")
    goal_lengths = [row["avg_length_goal"] for _, _, _, rows, _ in records for row in rows
                    if np.isfinite(row.get("avg_length_goal", np.nan))]
    length_top = int(np.ceil(max(goal_lengths) / 20.0) * 20)
    length_ax.axvline(decay, color="#555555", lw=1.0, ls="--", zorder=0)
    length_ax.set(xlim=(0, finish), ylim=(0, length_top),
                  xlabel="training episodes", ylabel="steps per goal reach")
    length_ax.grid(alpha=0.22, lw=0.7)
    env = f"JaxNav {config['map_size'][0]}x{config['map_size'][1]}, fill={config['fill']}, radius={config['goal_radius']}"
    fig.suptitle("Fixed-map JaxNav learning curves", x=0.09, ha="left", fontsize=15, fontweight="bold")
    ax.set_title(f"Goal reach rate | {env} | same {config['eval_episodes']:,} evaluation maps | faint = seed; solid = mean; dashed = median; band = ±SEM", loc="left", fontsize=10)
    length_ax.set_title(f"Goal-reaching episode length | includes only episodes that reached the goal | y-axis: 0–{length_top} steps", loc="left", fontsize=10)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    fig.text(0.09, 0.065,
             f"Log snapshot: {stamp} | arrays {', '.join(args.array)} | fixed_eval_seed={config['fixed_eval_seed']} | goal-rate bin={args.bin_width:,}; goal-length bin={args.length_bin_width:,} | summary needs ≥{args.minimum_seeds} seeds | band is descriptive ±SEM",
             fontsize=8.5, color="#333333")
    fig.text(0.09, 0.037, f"Run directories read: {', '.join(sorted(source_ids))}", fontsize=7.5, color="#555555")

    out = args.out or (DEFAULT_OUT_DIR / f"fixed_eval_arrays_{'_'.join(args.array)}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor="white")
    print(out.resolve())


if __name__ == "__main__":
    main()
