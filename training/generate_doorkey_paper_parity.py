"""Rebuild the 15-row DoorKey VLM-RB paper-parity recipe."""

import json
from pathlib import Path

from training.learners.doorkey.config import paper_parity_config


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/doorkey/vlmrb-paper-parity/training.json"


def main():
    rows = []
    for task, (arm, seed) in enumerate(
        (arm, seed)
        for arm in ("uniform", "per", "cce_decay")
        for seed in range(5)
    ):
        config = paper_parity_config(arm, seed)
        rows.append(
            dict(
                task=task,
                mode="train",
                gpu="h100",
                size=12,
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
        reference="arXiv:2602.01915 Appendix B/E.2/E.4",
        benchmark="MiniGrid-DoorKey-12x12-v0-vlmrb-paper-parity",
        rows=rows,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
