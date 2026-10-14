"""Rebuild the matched 15-row procedural DoorKey benchmark recipe."""

import json
from pathlib import Path

from training.learners.doorkey.config import procedural_benchmark_config


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/doorkey/procedural-12x12/training.json"


def main():
    rows = []
    task = 0
    for arm in ("uniform", "per", "cce_decay"):
        for seed in range(5):
            config = procedural_benchmark_config(arm, seed)
            rows.append(dict(
                task=task,
                mode="train",
                gpu="h100",
                size=12,
                arm=arm,
                collector="carry-safe",
                decay=config["decay_enabled"],
                seed=seed,
                config=config,
            ))
            task += 1
    manifest = dict(
        schema_version=1,
        environment="doorkey",
        reference_recipe="experiments/jaxnav/ranked-td-decay/training.json#task=33",
        benchmark="MiniGrid-DoorKey-12x12-v0-compatible",
        rows=rows,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
