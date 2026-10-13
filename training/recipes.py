"""Explicit recipe validation; no simulator import needed to validate."""
import json
from pathlib import Path
from training.registry import get_environment


def load_recipe(path, task):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported recipe schema")
    adapter = get_environment(manifest.get("environment"))
    rows = manifest["rows"]
    if len({r["task"] for r in rows}) != len(rows):
        raise ValueError("Duplicate task IDs")
    selected = [r for r in rows if r["task"] == task]
    if len(selected) != 1:
        raise ValueError(f"No unique task {task}")
    return validate_row(selected[0], adapter)


def validate_row(row, adapter):
    """Validate one parsed row; batch catalogs need not reread the whole file."""
    cfg = row["config"]
    adapter.validate_config(cfg)
    if cfg["seed"] != row["seed"]:
        raise ValueError("Seed mismatch")
    if "collector" in row and cfg.get("collector_variant") != row["collector"]:
        raise ValueError("Collector mismatch")
    if "decay" in row and cfg.get("decay_enabled") != row["decay"]:
        raise ValueError("Schedule mismatch")
    return row
