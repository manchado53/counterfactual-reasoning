"""Explicit reader for the October 7 decay figure, separate from new-run reports.

Only full, observed 5k bins count. Missing values become JSON null, never zero.
The protocol has means only: adding medians would change the historical figure.
"""
import hashlib
import argparse
import gzip
import sys
import json
from pathlib import Path

import numpy as np

GROUPS = ("Historical: old / no decay", "Old / decay", "Fixed / no decay", "Fixed / decay")
COLORS = ("#777777", "#e58606", "#2687bf", "#9c43a0")
ARMS = ("mul", "add", "only")
TITLES = ("CCE Multiply", "CCE Add 50/50", "CCE Only")
EDGES = np.arange(0, 200001, 5000)
X = (EDGES[:-1] + EDGES[1:]) / 2 / 1000


def checked_bytes(entry, root):
    path = root / entry["path"]
    compression = entry.get("compression")
    if compression not in (None, "gzip"):
        raise ValueError("Unknown compression")
    opener = gzip.open if compression == "gzip" else open
    with opener(path, "rb") as stream:
        blob = stream.read(entry["bytes"])
    if len(blob) != entry["bytes"] or hashlib.sha256(blob).hexdigest() != entry["sha256"]:
        raise ValueError(f"Source hash/length mismatch: {path}")
    return blob


def _records(entry, root):
    rows = []
    for line in checked_bytes(entry, root).splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue
        if entry["format"] == "progress":
            if not line.startswith(b"PROGRESS "):
                continue
            line = line[len(b"PROGRESS "):]
        row = json.loads(line)
        e, y = row["episode"], row["win_rate"]
        if not np.isfinite(e) or (rows and e <= rows[-1]["episode"]):
            raise ValueError("Invalid/repeated episode")
        if not np.isfinite(y) or not 0 <= y <= 1:
            raise ValueError("Invalid win_rate")
        if entry["format"] == "progress":
            for key in ("seed", "arm", "task", "eval_bank_hash"):
                if row[key] != entry[key]:
                    raise ValueError(f"Progress identity mismatch: {key}")
        rows.append(row)
    if len(rows) != entry["count"]:
        raise ValueError("Record count mismatch")
    return rows


def _bins(rows):
    ep = np.array([r["episode"] for r in rows])
    y = np.array([r["win_rate"] for r in rows])
    result = np.full(len(X), np.nan)
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        ix = (ep >= lo) & (ep < hi)
        if hi <= ep[-1] and ix.any():
            result[i] = y[ix].mean()
    return result


def _line(y, label, color, style="-", width=2.5, alpha=None):
    return dict(x=X.tolist(), y=[float(v) if np.isfinite(v) else None for v in y],
                label=label, color=color, linestyle=style, linewidth=width, alpha=alpha)


def read_manifest(path):
    """Read only explicitly selected byte prefixes; return exact plotted data."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest["schema"] != 1 or manifest["protocol"] != "decay-20261007-v1":
        raise ValueError("Unsupported historical protocol")
    groups, seen, banks = {}, set(), set()
    for entry in manifest["runs"]:
        identity = (entry["arm"], entry["group"], entry["seed"])
        if identity in seen:
            raise ValueError("Duplicate arm/group/seed")
        seen.add(identity)
        if entry["format"] not in ("eval", "progress"):
            raise ValueError("Unknown log format")
        if entry["arm"] not in (*ARMS, "Uniform", "PER") or entry["group"] not in (*GROUPS, "baseline"):
            raise ValueError("Unknown arm/group")
        if entry["seed"] not in range(5) or ((entry["arm"] in ("Uniform", "PER")) != (entry["group"] == "baseline")):
            raise ValueError("Invalid historical identity")
        rows = _records(entry, path.parent)
        if rows:
            groups.setdefault((entry["arm"], entry["group"]), []).append((entry["seed"], rows))
            if entry["format"] == "progress":
                banks.add(entry["eval_bank_hash"])
    if banks != set(manifest["banks"]) or len(banks) > 1:
        raise ValueError("Evaluation bank mismatch")
    panels, summary = [], []
    for arm, title in zip(ARMS, TITLES):
        lines = []
        for baseline, color, style in (("Uniform", "#14976c", "--"), ("PER", "#c63445", "-.")):
            runs = groups.get((baseline, "baseline"), [])
            if not runs:
                continue
            if len(runs) != 5 or {s for s, _ in runs} != set(range(5)):
                raise ValueError("Baseline requires seeds 0..4")
            values = np.array([_bins(r) for _, r in runs])
            if not np.all(np.isfinite(values)):
                raise ValueError("Incomplete baseline")
            lines.append(_line(values.mean(axis=0), f"{baseline}: historical old, mean of 5/5 seeds", color, style, 2.3))
        for name, color in zip(GROUPS, COLORS):
            runs = groups.get((arm, name), [])
            if not runs:
                lines.append(dict(x=[], y=[], label=f"{name}: waiting (0/5)", color=color,
                                  linestyle="-", linewidth=1.5, alpha=None))
                continue
            values = np.array([_bins(r) for _, r in runs])
            lines.extend(_line(y, None, color, width=.85, alpha=.22) for y in values)
            valid = np.all(np.isfinite(values), axis=0)
            mean = np.full(len(X), np.nan)
            mean[valid] = values[:, valid].mean(axis=0)
            lines.append(_line(mean, f"{name}: mean of {len(runs)}/5 seeds", color,
                               "--" if name.startswith("Historical") else "-"))
            summary.append(dict(arm=arm, group=name, seeds={str(s): r[-1]["episode"] for s, r in runs},
                                mean_through=int(EDGES[1:][valid][-1]) if valid.any() else 0))
        panels.append(dict(title=title, lines=lines))
    selection = [{k: v for k, v in entry.items() if k not in ("path", "compression")}
                 for entry in manifest["runs"]]
    return dict(panels=panels, summary=summary, selection=selection, absent=manifest["absent"])


def equality_gate(actual, frozen):
    """Exact recursive comparison to old-code output; no fitted tolerance."""
    blob = Path(frozen).read_bytes()
    expected = json.loads(blob)

    def difference(a, b, path="$"):
        if type(a) is not type(b):
            return path + ": type differs"
        if isinstance(a, dict):
            if a.keys() != b.keys():
                return path + ": keys differ"
            for key in a:
                found = difference(a[key], b[key], path + "." + key)
                if found:
                    return found
        elif isinstance(a, list):
            if len(a) != len(b):
                return path + ": length differs"
            for i, (av, bv) in enumerate(zip(a, b)):
                found = difference(av, bv, f"{path}[{i}]")
                if found:
                    return found
        elif a != b:
            return f"{path}: {a!r} != {b!r}"
        return None

    first = difference(actual, expected)
    panels = []
    if "panels" in actual and "panels" in expected:
        for a, b in zip(actual["panels"], expected["panels"]):
            diff = difference(a, b)
            panels.append(dict(title=b["title"], status="fail" if diff else "pass", first_difference=diff))
    return dict(status="fail" if first else "pass", first_difference=first, panels=panels,
                tolerance=0, reference_sha256=hashlib.sha256(blob).hexdigest())


def render(data, output, snapshot):
    """Preserve original legend labels, axes and scientific footnotes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True, sharey=True)
    for ax, panel in zip(axes, data["panels"]):
        ax.axvspan(100, 150, color="#ffe6aa", alpha=.5)
        for x in (100, 150):
            ax.axvline(x, color="#bb943d", lw=1, ls=":")
        for line in panel["lines"]:
            ax.plot(line["x"], [np.nan if v is None else v for v in line["y"]],
                    **{k: v for k, v in line.items() if k not in ("x", "y")})
        ax.set_title(panel["title"], loc="left", weight="bold")
        ax.set_ylim(0, .85)
        ax.set_xlim(0, 200)
        ax.set_ylabel("Goal reach rate")
        ax.grid(alpha=.16)
        ax.legend(loc="upper left", fontsize=9, ncol=2)
    axes[-1].set_xlabel("Training episodes (thousands)")
    fig.suptitle("Live JaxNav decay runs — sparse reward, 8×8 maps", fontsize=19, weight="bold", y=.99)
    fig.text(.09, .951, "Shaded 100k–150k: shift to ranked TD sampling. Thin = each seed; thick = arithmetic mean.", fontsize=11)
    fig.text(.09, .026, f"Log snapshot: {snapshot} | Arrays 280874 / 280876 / 280899 | 5k-episode bins\n"
             "Mean stops at the shared end of the shown seeds; thin tails show faster seeds. No filled tails.\n"
             "Eval: 1,000 fixed maps, seed 20260923. Uniform/PER/gray are historical old-collector runs; hardware differs.\n"
             "PER baseline uses raw TD, not the ranked TD decay endpoint. Baselines show means only. No confidence bands.",
             fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .08, 1, .94))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    reference = manifest["reference"]
    checked_bytes(reference, args.manifest.parent)
    checked_bytes(manifest["old_source"], args.manifest.parent)
    data = read_manifest(args.manifest)
    report = equality_gate(data, args.manifest.parent / reference["path"])
    report.update(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  old_source_sha256=manifest["old_source"]["sha256"],
                  selected_inputs=len(manifest["runs"]), absent=manifest["absent"],
                  runtime=dict(python=sys.version, numpy=np.__version__),
                  scope="Historical plotted data only; not training parity")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (args.output / "plotted-data.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    if report["status"] != "pass":
        raise SystemExit(1)
    render(data, args.output / "reproduced.png", manifest["snapshot"])
    print(json.dumps({k: v for k, v in report.items() if k != "absent"}, indent=2))


if __name__ == "__main__":
    main()
