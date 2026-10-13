"""Rebuild the matched 30-row DoorKey transfer recipe."""
import json
from pathlib import Path

from training.learners.doorkey.config import decay_transfer_config


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/doorkey/jaxnav-decay-transfer/training.json"


def main():
    rows = []
    task = 0
    for layout in ("6x6", "6x6_lava"):
        for arm in ("uniform", "per", "cce_decay"):
            for seed in range(5):
                config = decay_transfer_config(arm, seed, layout)
                rows.append(dict(
                    task=task,
                    mode="train",
                    gpu="h100",
                    layout=layout,
                    arm=arm,
                    collector="fixed",
                    decay=config["decay_enabled"],
                    seed=seed,
                    config=config,
                ))
                task += 1
    manifest = dict(
        schema_version=1,
        environment="doorkey",
        reference_recipe="experiments/jaxnav/ranked-td-decay/training.json#task=33",
        rows=rows,
    )
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
