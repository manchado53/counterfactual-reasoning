"""Rebuild five-arm means, medians and descriptive SEM from an explicit manifest.

This separate historical protocol uses (lo, hi], unlike the decay plot's [lo, hi).
The frozen expected output comes from the existing October 7 audit, not this code.
"""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from analysis.historical_reporting import checked_bytes, equality_gate

ARMS = ("Uniform", "PER", "CCE Mul", "CCE Only", "CCE Add 50/50")
COLORS = ("#377eb8", "#ff7f00", "#984ea3", "#4daf4a", "#e41a1c")


def read_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest["schema"] != 1 or manifest["protocol"] != "five-arm-20261007-v1":
        raise ValueError("Unsupported historical protocol")
    values = {arm: [] for arm in ARMS}
    seen, sources = set(), []
    for entry in manifest["runs"]:
        identity = (entry["arm"], entry["seed"])
        if identity in seen or entry["arm"] not in ARMS or entry["seed"] not in range(5):
            raise ValueError("Duplicate or invalid arm/seed")
        seen.add(identity)
        rows = [json.loads(line) for line in checked_bytes(entry, path.parent).splitlines() if line.strip()]
        if not rows or rows[-1]["episode"] < 250000:
            raise ValueError("Incomplete five-arm run")
        ep = np.array([r["episode"] for r in rows])
        y = np.array([r["win_rate"] for r in rows])
        if not np.all(np.diff(ep) > 0) or not np.all(np.isfinite(y)) or not np.all((y >= 0) & (y <= 1)):
            raise ValueError("Invalid historical evaluation rows")
        bins = []
        for lo in range(0, 250000, 5000):
            selected = [r["win_rate"] for r in rows if lo < r["episode"] <= lo + 5000]
            if not selected:
                raise ValueError("Missing historical bin")
            bins.append(np.mean(selected))
        values[entry["arm"]].append(bins)
        sources.append(dict(arm=entry["arm"], seed=entry["seed"], run=entry["origin"],
                            config=entry["config"], eval_sha256=entry["sha256"]))
    if seen != {(arm, seed) for arm in ARMS for seed in range(5)}:
        raise ValueError("Five-arm figure requires all 25 runs")
    summaries = {}
    for arm in ARMS:
        a = np.array(values[arm])
        summaries[arm] = dict(mean=a.mean(0).tolist(), median=np.median(a, axis=0).tolist(),
                              sem=(a.std(0, ddof=1) / np.sqrt(5)).tolist())
    return dict(bin_centers=np.arange(2500, 250000, 5000).astype(float).tolist(),
                summaries=summaries, runs=sources)


def freeze(visualizations, output):
    """Copy hashed raw logs and an independently saved old numeric output."""
    root, output = Path(visualizations), Path(output)
    audit_blob = (root / "jaxnav-five-arms-audit.json").read_bytes()
    audit = json.loads(audit_blob)
    inputs = []
    for row in audit["runs"]:
        blob = (Path(row["run"]) / "eval.jsonl").read_bytes()
        if hashlib.sha256(blob).hexdigest() != row["eval_sha256"]:
            raise ValueError(f"Old five-arm log changed: {row['run']}")
        inputs.append((row, blob))
    output.mkdir(parents=True, exist_ok=False)

    def save(name, blob):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        return dict(path=name, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())

    old_source = save("old/five-arm-audit.py", (root / "five-arm-audit.py").read_bytes())
    loader = Path('/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/.claude/worktrees/research+cce-robotics-transfer/analysis/claim2/fixed_eval_log_curves.py')
    save("old/fixed_eval_log_curves.py", loader.read_bytes())
    save("old/audit.json", audit_blob)
    expected = {key: audit[key] for key in ("bin_centers", "summaries", "runs")}
    reference = save("old/plotted-data.json", json.dumps(expected, indent=2, allow_nan=False).encode() + b"\n")
    runs = []
    for i, (row, blob) in enumerate(inputs):
        entry = save(f"logs/{i:03d}.jsonl.gz", gzip.compress(blob, mtime=0))
        entry.update(bytes=len(blob), sha256=row["eval_sha256"], compression="gzip",
                     arm=row["arm"], seed=row["seed"], config=row["config"], origin=row["run"])
        runs.append(entry)
    manifest = dict(schema=1, protocol="five-arm-20261007-v1", runs=runs,
                    reference=reference, old_source=old_source,
                    rules=dict(bins="(lo, hi], width 5000, 0..250000", seeds="0..4 for each arm",
                               aggregation="per-seed bin mean, equal-weight mean/median, ddof=1 SEM",
                               smoothing="none", tails="all 25 runs must cover every bin; no fill",
                               limitations=audit["limitations"],
                               differing_config_fields=audit["differing_config_fields"]))
    save("manifest.json", json.dumps(manifest, indent=2).encode() + b"\n")


def render(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12.8, 7.6), dpi=170)
    x = np.array(data["bin_centers"]) / 1000
    for arm, color in zip(ARMS, COLORS):
        row = data["summaries"][arm]
        mean, median, sem = (np.array(row[key]) for key in ("mean", "median", "sem"))
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=.08, lw=0)
        ax.plot(x, mean, color=color, lw=2.6, label=arm + " (5 seeds)")
        ax.plot(x, median, color=color, lw=1.2, ls="--", alpha=.8)
    ax.axvline(62.5, color="#777", ls=":", lw=1)
    ax.text(64, .65, "Epsilon floor", fontsize=10, color="#555")
    ax.set(xlim=(0, 250), ylim=(0, .8), xlabel="Logged training episodes (thousands)", ylabel="Goal reach rate")
    ax.grid(alpha=.18)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=10)
    fig.suptitle("JaxNav: all five replay arms", x=.085, ha="left", fontsize=18, fontweight="bold")
    ax.set_title("8x8 | fill 0.3 | radius 0.8 | sparse: goal +1, crash 0 | fixed evaluation seed 20260923", loc="left", fontsize=11, pad=12)
    fig.subplots_adjust(left=.085, right=.98, bottom=.26, top=.86)
    fig.text(.085, .13, "Solid = mean; dashed = median; light band = descriptive ±SEM (not a 95% confidence interval).\nEach point averages evaluations within a 5,000-episode bin per seed, then across all five seeds.", fontsize=10)
    fig.text(.085, .045, "Arrays 277733 + 278035 | 25 unique runs | shared logged settings checked | no filled tails\nReplay rules differ by design, including raw-TD PER vs ranked CCE signals. Historical code/package identity not certified.\nOriginal episode axis retained; these are historical runs, not a new fixed-collector rerun. Read on October 7, 2026.", fontsize=9, color="#444")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-from", type=Path)
    mode.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.freeze_from:
        freeze(args.freeze_from, args.output)
        return
    manifest = json.loads(args.manifest.read_text())
    for entry in (manifest["reference"], manifest["old_source"]):
        checked_bytes(entry, args.manifest.parent)
    data = read_manifest(args.manifest)
    report = equality_gate(data, args.manifest.parent / manifest["reference"]["path"])
    expected = json.loads((args.manifest.parent / manifest["reference"]["path"]).read_text())
    report.update(arms={arm: "pass" if data["summaries"][arm] == expected["summaries"][arm] else "fail" for arm in ARMS},
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  shared_reader_sha256=hashlib.sha256(Path(__file__).with_name("historical_reporting.py").read_bytes()).hexdigest(),
                  old_source_sha256=manifest["old_source"]["sha256"], selected_inputs=len(data["runs"]),
                  runtime=dict(python=sys.version, numpy=np.__version__),
                  scope="Historical plotted data, not historical training code/package certification")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "plotted-data.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    if report["status"] != "pass":
        raise SystemExit(1)
    render(data, args.output / "reproduced.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
