"""Build matched, multi-environment replay figures from saved training attempts.

The input study file names physical run directories. No training is started here.
Only checkpoints observed in every listed run of a cell enter a comparison.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ARM_STYLE = {
    "uniform": ("Uniform", "#292929"),
    "rank_per": ("Rank PER", "#b34a49"),
    "proportional_per": ("Proportional PER", "#3274b9"),
    "cce_rank": ("CCE rank blend", "#238b68"),
    "cce_raw": ("CCE raw blend", "#b4852c"),
}
AXIS_LABEL = {"env_steps": "Environment steps", "episode": "Training episodes",
              "chunk": "Training chunks"}


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path):
    return json.loads(path.read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_run(run_spec, cell, manifest_dir):
    directory = Path(run_spec["directory"])
    if not directory.is_absolute():
        directory = manifest_dir / directory
    directory = directory.resolve()
    config_path = directory / "config.json"
    attempt_path = directory / "attempt.json"
    log_file = cell.get("log_file", "eval.jsonl")
    _require(Path(log_file).name == log_file, f"Log file must be a name: {cell['id']}")
    log_path = directory / log_file
    config_bytes = config_path.read_bytes()
    attempt_bytes = attempt_path.read_bytes()
    log_bytes = log_path.read_bytes()
    config, attempt = json.loads(config_bytes), json.loads(attempt_bytes)
    _require(attempt.get("resolved_config") == config,
             f"Saved config differs from recorded attempt: {directory}")
    _require(attempt.get("environment") == cell["environment"],
             f"Environment mismatch: {directory}")
    _require(not attempt.get("smoke"), f"Smoke run is not study evidence: {directory}")
    _require(attempt.get("eval_axis") == cell["axis"],
             f"Training axis mismatch: {directory}")
    _require(attempt.get("eval_bank_hash"), f"Missing fixed evaluation bank: {directory}")
    source_hashes = attempt.get("simulator_source_hashes", attempt.get("jaxmarl_source_hashes"))
    _require(source_hashes, f"Missing simulator source hashes: {directory}")
    _require(config.get("seed") == run_spec["seed"], f"Seed mismatch: {directory}")
    _require(attempt.get("status") in {"completed", "running", "failed"},
             f"Unknown run status: {directory}")
    step_field, metric = cell.get("step_field", cell["axis"]), cell["metric"]["field"]
    observations = {}
    previous = -1
    for line in log_bytes.decode().splitlines():
        row = json.loads(line)
        step, value = row[step_field], row[metric]
        _require(type(step) is int and step > previous,
                 f"Repeated or unordered checkpoint: {directory}")
        _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and np.isfinite(value), f"Invalid score: {directory}")
        bounds = cell["metric"].get("bounds")
        _require(bounds is None or bounds[0] <= value <= bounds[1],
                 f"Score outside declared bounds: {directory}")
        observations[step] = float(value)
        previous = step
    _require(observations, f"No evaluations: {directory}")
    method_keys = set(cell["method_config_keys"])
    common_config = {key: value for key, value in config.items()
                     if key not in method_keys and key != "seed"}
    method_config = {key: config.get(key) for key in sorted(method_keys)}
    elapsed = attempt.get("seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0:
        elapsed = None
    return {
        "arm": run_spec["arm"], "seed": run_spec["seed"], "status": attempt["status"],
        "observations": observations, "common_config": common_config,
        "method_config": method_config, "config": config,
        "eval_bank_hash": attempt["eval_bank_hash"],
        "simulator_source_hashes": source_hashes, "elapsed_seconds": elapsed,
        "source": {"directory": str(directory),
                   "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                   "attempt_sha256": hashlib.sha256(attempt_bytes).hexdigest(),
                   "eval_sha256": hashlib.sha256(log_bytes).hexdigest(),
                   "source_bundle_sha256": attempt.get("source_bundle_sha256"),
                   "recipe_sha256": attempt.get("recipe_sha256")},
    }


def analyze_study(manifest_path):
    """Return checked curves and status. Final scores require the declared budget."""
    manifest_path = Path(manifest_path)
    manifest = _json(manifest_path)
    _require(manifest.get("schema_version") == 1, "Unsupported study schema")
    _require(manifest.get("cells"), "Study needs at least one environment cell")
    output = {"schema_version": 1, "study": manifest["study"],
              "manifest_sha256": _hash(manifest_path), "cells": []}
    cell_ids = set()
    for cell in manifest["cells"]:
        _require(cell["id"] not in cell_ids, f"Duplicate cell: {cell['id']}")
        cell_ids.add(cell["id"])
        _require(cell["axis"] in AXIS_LABEL, f"Unsupported axis: {cell['id']}")
        _require(type(cell["budget"]) is int and cell["budget"] > 0,
                 f"Invalid budget: {cell['id']}")
        _require(type(cell["early_step"]) is int and 0 < cell["early_step"] < cell["budget"],
                 f"Invalid early checkpoint: {cell['id']}")
        _require(cell["metric"].get("higher_is_better") is True,
                 f"Scores must declare higher_is_better: {cell['id']}")
        _require(isinstance(cell["method_config_keys"], list) and cell["method_config_keys"],
                 f"Declare method config keys: {cell['id']}")
        _require("seed" not in cell["method_config_keys"],
                 f"Seed is checked separately: {cell['id']}")
        _require(len(cell["arms"]) >= 2 and "uniform" in cell["arms"],
                 f"Cell needs Uniform and another replay arm: {cell['id']}")
        _require(len(cell["arms"]) == len(set(cell["arms"]))
                 and set(cell["arms"]) <= ARM_STYLE.keys(), f"Unknown/repeated arm: {cell['id']}")
        declared_seeds = cell["seeds"]
        _require(declared_seeds and len(declared_seeds) == len(set(declared_seeds)),
                 f"Missing/repeated seeds: {cell['id']}")
        runs = [_read_run(spec, cell, manifest_path.parent) for spec in cell["runs"]]
        _require(all(max(run["observations"]) <= cell["budget"] for run in runs),
                 f"Run exceeds declared budget: {cell['id']}")
        expected = {(arm, seed) for arm in cell["arms"] for seed in declared_seeds}
        identities = [(run["arm"], run["seed"]) for run in runs]
        _require(len(identities) == len(set(identities)) and set(identities) == expected,
                 f"Runs must cover each declared arm and seed exactly once: {cell['id']}")
        first = runs[0]
        for run in runs[1:]:
            _require(run["common_config"] == first["common_config"],
                     f"Unmatched non-method config: {cell['id']}, {run['source']['directory']}")
            _require((run["eval_bank_hash"], run["simulator_source_hashes"])
                     == (first["eval_bank_hash"], first["simulator_source_hashes"]),
                     f"Evaluation/simulator mismatch: {cell['id']}")
        blend_windows = []
        for schedule in cell.get("blend_windows", []):
            arm = schedule.get("arm")
            _require(arm in cell["arms"] and arm != "uniform",
                     f"Blend arm is missing or Uniform: {cell['id']}")
            _require(arm not in {window["arm"] for window in blend_windows},
                     f"Repeated blend arm: {cell['id']}")
            _require(schedule.get("axis") == cell["axis"],
                     f"Blend axis differs from plot axis: {cell['id']}")
            keys = (schedule.get("start_key"), schedule.get("end_key"),
                    schedule.get("enabled_key"))
            _require(all(isinstance(key, str) and key for key in keys),
                     f"Blend needs saved config keys: {cell['id']}")
            members = [run for run in runs if run["arm"] == arm]
            start_key, end_key, enabled_key = keys
            intervals = []
            for run in members:
                config = run["config"]
                _require(config.get(enabled_key) is True,
                         f"Blend not enabled in saved run: {run['source']['directory']}")
                start, end = config.get(start_key), config.get(end_key)
                _require(type(start) is int and type(end) is int
                         and 0 <= start < end <= cell["budget"],
                         f"Invalid blend window in saved run: {run['source']['directory']}")
                intervals.append((start, end))
            _require(len(set(intervals)) == 1,
                     f"Blend window drifts across seeds: {cell['id']}, {arm}")
            blend_windows.append({"arm": arm, "start": intervals[0][0],
                                  "end": intervals[0][1], "axis": cell["axis"],
                                  "config_keys": {"start": start_key, "end": end_key,
                                                  "enabled": enabled_key}})
        for arm in cell["arms"]:
            members = [run for run in runs if run["arm"] == arm]
            _require(all(run["method_config"] == members[0]["method_config"]
                         for run in members), f"Method drift within {arm}: {cell['id']}")
        shared_steps = sorted(set.intersection(*(set(run["observations"]) for run in runs)))
        _require(len(shared_steps) >= 2 and cell["early_step"] in shared_steps,
                 f"Early checkpoint or shared evaluations missing: {cell['id']}")
        _require(max(shared_steps) <= cell["budget"],
                 f"Evaluation exceeds declared budget: {cell['id']}")
        complete = all(run["status"] == "completed" for run in runs)
        final_ready = complete and cell["budget"] in shared_steps
        curves = []
        uniform_early = None
        for arm in cell["arms"]:
            members = sorted((run for run in runs if run["arm"] == arm),
                             key=lambda run: run["seed"])
            values = np.array([[run["observations"][step] for step in shared_steps]
                               for run in members])
            median = np.median(values, axis=0)
            q1, q3 = np.percentile(values, [25, 75], axis=0)
            early = float(median[shared_steps.index(cell["early_step"])])
            early_indices = [index for index, step in enumerate(shared_steps)
                             if step <= cell["early_step"]]
            early_auc = None
            if len(early_indices) > 1:
                x = np.asarray([shared_steps[index] for index in early_indices])
                y = median[early_indices]
                early_auc = float(np.trapezoid(y, x) / (x[-1] - x[0]))
            if arm == "uniform":
                uniform_early = early
            elapsed = [run["elapsed_seconds"] for run in members]
            curves.append({"arm": arm, "label": ARM_STYLE[arm][0],
                           "n_seeds": len(members), "steps": shared_steps,
                           "median": median.tolist(), "q1": q1.tolist(), "q3": q3.tolist(),
                           "early_score": early, "early_auc": early_auc,
                           "final_score": float(median[-1]) if final_ready else None,
                           "median_wall_seconds": float(np.median(elapsed)) if all(
                               seconds is not None for seconds in elapsed) else None,
                           "method_config": members[0]["method_config"]})
        for curve in curves:
            curve["early_difference_vs_uniform"] = curve["early_score"] - uniform_early
        uniform_final = next(curve["final_score"] for curve in curves
                             if curve["arm"] == "uniform")
        for curve in curves:
            reached = next((step for step, score in zip(shared_steps, curve["median"])
                            if uniform_final is not None and score >= uniform_final), None)
            curve["steps_to_uniform_final"] = reached
            curve["equivalence_fraction"] = reached / cell["budget"] if reached is not None else None
        output["cells"].append({
            "id": cell["id"], "environment": cell["environment"],
            "title": cell.get("title", cell["environment"].replace("_", " ").title()),
            "metric": cell["metric"], "axis": cell["axis"],
            "step_field": cell.get("step_field", cell["axis"]),
            "log_file": cell.get("log_file", "eval.jsonl"),
            "budget": cell["budget"], "early_step": cell["early_step"],
            "blend_windows": blend_windows,
            "comparison_status": "final_budget" if final_ready else "interim",
            "shared_observed_steps": shared_steps,
            "eval_bank_hash": first["eval_bank_hash"],
            "simulator_source_hashes": first["simulator_source_hashes"],
            "common_config": first["common_config"], "curves": curves,
            "runs": [{key: run[key] for key in ("arm", "seed", "status", "source")}
                     for run in runs],
        })
    cross_environment = {}
    for cell in output["cells"]:
        if cell["comparison_status"] != "final_budget":
            continue
        uniform = next(curve for curve in cell["curves"] if curve["arm"] == "uniform")
        for curve in cell["curves"]:
            if curve["arm"] == "uniform":
                continue
            entry = cross_environment.setdefault(curve["arm"], {
                "arm": curve["arm"], "eligible_cells": 0,
                "early_better_cells": 0, "early_auc_better_cells": 0,
                "final_better_cells": 0})
            entry["eligible_cells"] += 1
            entry["early_better_cells"] += int(curve["early_score"] > uniform["early_score"])
            entry["early_auc_better_cells"] += int(
                curve["early_auc"] is not None and uniform["early_auc"] is not None
                and curve["early_auc"] > uniform["early_auc"])
            entry["final_better_cells"] += int(curve["final_score"] > uniform["final_score"])
    output["cross_environment_summary"] = list(cross_environment.values())
    return output


def render_study(result, output_dir):
    """Write a data snapshot, a compact summary, and PER-style small multiples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "study-results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "cell", "environment", "arm", "n_seeds", "status", "early_step",
            "early_score", "early_auc", "early_difference_vs_uniform",
            "final_step", "final_score", "steps_to_uniform_final", "equivalence_fraction",
            "median_wall_seconds"])
        writer.writeheader()
        for cell in result["cells"]:
            for curve in cell["curves"]:
                writer.writerow({"cell": cell["id"], "environment": cell["environment"],
                                 "arm": curve["arm"], "n_seeds": curve["n_seeds"],
                                 "status": cell["comparison_status"],
                                 "early_step": cell["early_step"],
                                 "early_score": curve["early_score"],
                                 "early_auc": curve["early_auc"] if curve["early_auc"] is not None else "",
                                 "early_difference_vs_uniform": curve["early_difference_vs_uniform"],
                                 "final_step": cell["budget"] if curve["final_score"] is not None else "",
                                 "final_score": curve["final_score"] if curve["final_score"] is not None else "",
                                 "steps_to_uniform_final": curve["steps_to_uniform_final"]
                                 if curve["steps_to_uniform_final"] is not None else "",
                                 "equivalence_fraction": curve["equivalence_fraction"]
                                 if curve["equivalence_fraction"] is not None else "",
                                 "median_wall_seconds": curve["median_wall_seconds"]
                                 if curve["median_wall_seconds"] is not None else ""})
    with (output_dir / "cross-environment.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "arm", "eligible_cells", "early_better_cells",
            "early_auc_better_cells", "final_better_cells"])
        writer.writeheader()
        writer.writerows(result["cross_environment_summary"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = result["cells"]
    for early_zoom in (False, True):
        fig, axes = plt.subplots(1, len(cells), figsize=(5.2 * len(cells), 4.2), squeeze=False)
        for axis, cell in zip(axes[0], cells):
            for window in cell["blend_windows"]:
                if early_zoom and window["start"] >= cell["early_step"]:
                    continue
                label, color = ARM_STYLE[window["arm"]]
                axis.axvspan(window["start"], window["end"], color=color, alpha=0.09,
                             label=f"{label} blend window", zorder=0)
                axis.axvline(window["start"], color=color, ls="--", lw=1, alpha=0.65)
                axis.axvline(window["end"], color=color, ls="--", lw=1, alpha=0.65)
            for curve in cell["curves"]:
                label, color = ARM_STYLE[curve["arm"]]
                axis.plot(curve["steps"], curve["median"], label=label, color=color, lw=2)
                if curve["n_seeds"] > 1:
                    axis.fill_between(curve["steps"], curve["q1"], curve["q3"],
                                      color=color, alpha=0.16)
            axis.axvline(cell["early_step"], color="#888888", ls=":", lw=1)
            status = "full budget" if cell["comparison_status"] == "final_budget" else "interim"
            axis.set(title=f"{cell['title']} · {len(cell['runs']) // len(cell['curves'])} seeds/arm · {status}",
                     xlabel=AXIS_LABEL[cell["axis"]], ylabel=cell["metric"]["label"])
            if early_zoom:
                axis.set_xlim(0, cell["early_step"])
            if "bounds" in cell["metric"]:
                axis.set_ylim(cell["metric"]["bounds"])
            axis.grid(alpha=0.2)
        legend = {}
        for axis in axes[0]:
            handles, labels = axis.get_legend_handles_labels()
            legend.update(zip(labels, handles))
        labels, handles = list(legend), list(legend.values())
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055),
                   ncol=min(len(labels), 5), frameon=False)
        fig.text(0.5, 0.01, "Line: median · Band: interquartile range · Shared observed checkpoints only",
                 ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        name = "learning-curves-early" if early_zoom else "learning-curves"
        fig.savefig(output_dir / f"{name}.png", dpi=180)
        fig.savefig(output_dir / f"{name}.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_study(analyze_study(args.manifest), args.output)


if __name__ == "__main__":
    main()
