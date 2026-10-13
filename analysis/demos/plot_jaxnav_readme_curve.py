"""Plot the matched three-seed JaxNav README comparison from eval.jsonl."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"Uniform": "#4D4D4D", "PER": "#E68613", "CCE Only + decay": "#8E44AD"}


def source_record(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    parts = path.parts
    try:
        run_index = parts.index("runs") + 1
        reference = "/".join(parts[run_index:])
    except ValueError:
        reference = path.name
    return {"reference": reference, "sha256": digest}


def load_bins(path: Path, bin_size: int = 5000) -> dict[int, float]:
    buckets: dict[int, list[float]] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            episode = int(row["episode"])
            if episode > 200_500:
                continue
            center = min(round(episode / bin_size) * bin_size, 200_000)
            buckets.setdefault(center, []).append(float(row["win_rate"]))
    return {episode: float(np.mean(values)) for episode, values in buckets.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniform", nargs=3, required=True)
    parser.add_argument("--per", nargs=3, required=True)
    parser.add_argument("--cce", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    groups = {
        "Uniform": [load_bins(Path(path)) for path in args.uniform],
        "PER": [load_bins(Path(path)) for path in args.per],
        "CCE Only + decay": [load_bins(Path(path)) for path in args.cce],
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    summary = {}
    for label, runs in groups.items():
        episodes = sorted(set.intersection(*(set(run) for run in runs)))
        values = np.array([[run[episode] for episode in episodes] for run in runs])
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
        ax.fill_between(episodes, mean - sem, mean + sem, color=COLORS[label], alpha=0.15)
        ax.plot(episodes, mean, color=COLORS[label], linewidth=2.8, label=label)
        window = [index for index, episode in enumerate(episodes) if 100_000 <= episode <= 150_000]
        summary[label] = float(mean[window].mean())
    ax.axvspan(100_000, 150_000, color="#F3D36A", alpha=0.16)
    ax.text(125_000, 0.73, "measured CCE lead", ha="center", color="#705B16")
    ax.set(title="JaxNav fixed-map learning", xlabel="training episodes", ylabel="goal reach rate")
    ax.set(xlim=(0, 200_000), ylim=(0, 0.8))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="lower center")
    ax.text(
        0.01, 0.98,
        "same seeds 0, 1, 2  •  same 1,000 evaluation maps  •  solid = mean  •  band = ±SEM",
        transform=ax.transAxes, va="top", fontsize=9, color="#444444",
    )
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "window_episodes": [100_000, 150_000],
        "window_mean": summary,
        "sources": {
            "Uniform": [source_record(Path(path)) for path in args.uniform],
            "PER": [source_record(Path(path)) for path in args.per],
            "CCE Only + decay": [source_record(Path(path)) for path in args.cce],
        },
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
