"""Explicit recipe validation; no simulator import needed to validate."""
import json
from pathlib import Path
from experiments.learners.jaxnav.config import DEFAULT_CONFIG

EXTRA_KEYS = {"decay_start", "decay_end", "decay_enabled", "collector_variant", "experiment",
              "cf_action_prob_temperature", "cf_drift_probe_n", "cf_drift_probe_start_ep",
              "cf_score_chunk", "normalise_priorities", "score_on_insert"}


def load_recipe(path, task):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1 or manifest.get("environment") not in ("jaxnav", "frozenlake"):
        raise ValueError("Unsupported recipe schema/environment")
    rows = manifest["rows"]
    if len({r["task"] for r in rows}) != len(rows):
        raise ValueError("Duplicate task IDs")
    selected = [r for r in rows if r["task"] == task]
    if len(selected) != 1:
        raise ValueError(f"No unique task {task}")
    row = selected[0]
    cfg = row["config"]
    if manifest["environment"] == "frozenlake":
        from experiments.learners.frozenlake.build import validate_config
        validate_config(cfg)
        if cfg["seed"] != row["seed"] or cfg["collector_variant"] != row["collector"]:
            raise ValueError("Seed/collector mismatch")
        return row
    unknown = set(cfg) - set(DEFAULT_CONFIG) - EXTRA_KEYS
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    # Require resolved defaults: never fill missing historical settings with new defaults.
    missing = set(DEFAULT_CONFIG) - set(cfg)
    if missing:
        raise ValueError(f"Unresolved settings: {sorted(missing)}")
    if cfg["collector_variant"] not in ("old", "fixed") or cfg["collector_variant"] != row["collector"]:
        raise ValueError("Collector mismatch")
    if cfg["seed"] != row["seed"] or cfg["decay_enabled"] != row["decay"]:
        raise ValueError("Seed/schedule mismatch")
    if cfg["decay_end"] <= cfg["decay_start"]:
        raise ValueError("Invalid decay interval")
    if cfg["algorithm"] not in ("consequence-dqn", "dqn", "dqn-uniform"):
        raise ValueError("Unsupported algorithm")
    if cfg["algorithm"] == "consequence-dqn" and cfg["normalise_priorities"] != "rank":
        raise ValueError("This scheduled learner requires rank priorities")
    if set(cfg["PER_parameters"]) != {"eps", "beta", "maximum_priority"}:
        raise ValueError("Unknown or missing PER settings")
    if cfg["priority_mixing"] not in ("additive", "multiplicative"):
        raise ValueError("Unsupported mixing rule")
    if not isinstance(cfg["decay_enabled"], bool):
        raise ValueError("decay_enabled must be a boolean")
    if cfg["batch_size"] > cfg["buffer_capacity"]:
        raise ValueError("Batch exceeds buffer capacity")
    if not cfg["vectorized"]:
        raise ValueError("Only vectorized JaxNav has been migrated")
    for key in ("n_episodes", "n_envs", "collect_steps", "batch_size", "buffer_capacity", "eval_episodes"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    return row
