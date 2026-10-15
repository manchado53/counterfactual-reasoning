"""Rebuild the 15-row DoorKey full-state 16x16 recipe."""

import json
from pathlib import Path

from training.learners.doorkey.config import fullstate_16x16_config


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/doorkey/fullstate-16x16/training.json"


def main():
    rows = []
    for task, (arm, seed) in enumerate(
        (arm, seed)
        for arm in ("uniform", "per", "cce_decay")
        for seed in range(5)
    ):
        config = fullstate_16x16_config(arm, seed)
        rows.append(
            dict(
                task=task,
                mode="train",
                gpu="h100",
                size=16,
                arm=arm,
                collector="carry-safe",
                decay=config["decay_enabled"],
                seed=seed,
                config=config,
            )
        )
    manifest = dict(
        schema_version=1,
        environment="doorkey",
        reference="DoorKey 12x12 VLM-RB paper-parity difficulty extension",
        benchmark="MiniGrid-DoorKey-16x16-v0-fullstate-matched",
        rows=rows,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
