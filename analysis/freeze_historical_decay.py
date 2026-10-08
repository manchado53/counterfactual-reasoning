"""Freeze the recovered 2026-10-07 22:24 figure using the OLD plotting code.

No training imports or jobs. Original files are read only. The old script's last
three output calls are removed; its data selection and plotting run unchanged.
Snapshot byte lengths/hashes recover the old prefix even if a log later grew.
This is a one-time trusted-source capture, NOT a generic Python sandbox.
"""
import argparse
import ast
import hashlib
import gzip
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

OLD_SHA256 = "942ae0c4bd30e6981011d075319980401c9c7b175efc2cdacaecb8810d38738b"


def freeze(visualizations, output):
    visualizations, output = Path(visualizations), Path(output)
    source = visualizations / "decay-training-curves.py"
    code = source.read_bytes()
    if hashlib.sha256(code).hexdigest() != OLD_SHA256:
        raise ValueError("Unreviewed old plot source; refusing execution")
    snapshot_path = visualizations / "decay-training-curves-with-baselines-sources.json"
    snapshot_blob = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_blob)
    blobs = {}
    for entry in snapshot["sources"]:
        with Path(entry["path"]).open("rb") as stream:
            blob = stream.read(entry["bytes"])
        if len(blob) != entry["bytes"] or hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            raise ValueError(f"Unavailable snapshot prefix: {entry['path']}")
        blobs[entry["path"]] = blob

    tree = ast.parse(code)
    # Guard the exact reviewed output statements before suppressing source writes.
    tails = [ast.unparse(node) for node in tree.body[-3:]]
    if not (tails[0].startswith("fig.savefig(") and ".write_text(" in tails[1]
            and tails[2].startswith("print(")):
        raise ValueError("Old output statements changed")
    tree.body = tree.body[:-3]
    # Only the display timestamp is restored; selection/binning remain old code.
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "stamp" for t in node.targets):
            node.value = ast.Constant(snapshot["snapshot"])
    ast.fix_missing_locations(tree)

    original_read_text, original_exists = Path.read_text, Path.exists
    text_inputs = {}

    def read_text(path, *args, **kwargs):
        value = original_read_text(path, *args, **kwargs)
        text_inputs[str(path)] = value
        return value

    def exists(path):
        if path.parent.name == "jaxnav_decay_logs":
            return str(path) in blobs
        return original_exists(path)

    namespace = dict(__file__=str(source), __name__="historical_capture")
    with patch.object(Path, "read_bytes", lambda path: blobs[str(path)]), \
            patch.object(Path, "read_text", read_text), patch.object(Path, "exists", exists):
        exec(compile(tree, str(source), "exec"), namespace)
    if namespace["sources"] != snapshot["sources"] or namespace["summary"] != snapshot["summary"]:
        raise ValueError("Old source selection/summary differs from saved historical figure")
    if set(namespace["banks"]) != set(snapshot["banks"]):
        raise ValueError("Old bank identity differs")

    panels = []
    for ax in namespace["axes"]:
        lines = []
        for line in ax.lines[2:]:  # Two original vertical decay guides, not data.
            label = line.get_label()
            lines.append(dict(x=[float(x) for x in line.get_xdata()],
                              y=[float(y) if np.isfinite(y) else None for y in line.get_ydata()],
                              label=None if label.startswith("_") else label,
                              color=line.get_color(), linestyle=line.get_linestyle(),
                              linewidth=line.get_linewidth(), alpha=line.get_alpha()))
        panels.append(dict(title=ax.get_title(loc="left"), lines=lines))
    old_data = dict(panels=panels, summary=namespace["summary"])

    output.mkdir(parents=True, exist_ok=False)

    def save(name, blob, origin=None):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        entry = dict(path=name, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        if origin:
            entry["origin"] = origin
        return entry

    def save_json(name, data):
        return save(name, json.dumps(data, indent=2, allow_nan=False).encode() + b"\n")

    old_source = save("old/decay-training-curves.py", code, str(source))
    save("old/snapshot.json", snapshot_blob, str(snapshot_path))
    input_index = [save(f"old/inputs/{i:03d}-{Path(p).name}", value.encode(), p)
                   for i, (p, value) in enumerate(text_inputs.items())]
    save_json("old/input-index.json", input_index)
    namespace["fig"].savefig(output / "old/reproduced.png", dpi=180)
    namespace["plt"].close(namespace["fig"])

    # Explicit identities are resolved from the old manifests, never guessed from
    # curve shape or score. The old execution above independently validates them.
    identities, absent, seen = {}, [], set()
    for row in namespace["rows"]:
        if row["mode"] != "train":
            continue
        arm, seed = row["arm"], row["seed"]
        if (arm, seed) not in seen:
            seen.add((arm, seed))
            identities[str(Path(row["parent_run"]) / "eval.jsonl")] = dict(
                arm=arm, seed=seed, group="Historical: old / no decay", format="eval",
                config_status="historical metrics header retained only if read by old plot")
        array = {"h100": 280874, "v100": 280899, "t4": 280876}[row["gpu"]]
        path = f'/home/ad.msoe.edu/manchadoa/jaxnav_decay_logs/{array}_{row["task"]}.out'
        group = "Old / decay" if row["collector"] == "old" else "Fixed / decay" if row["decay"] else "Fixed / no decay"
        identity = dict(arm=arm, seed=seed, group=group, format="progress", task=row["task"],
                        attempt=f'{array}_{row["task"]}', gpu=row["gpu"], config=row["config"],
                        eval_bank_hash=snapshot["banks"][0])
        if path in blobs:
            identities[path] = identity
        else:
            absent.append(dict(**identity, origin=path, reason="absent at saved snapshot; not filled or replaced"))
    for row in namespace["audit"]["runs"]:
        if row["arm"] in ("Uniform", "PER"):
            identities[str(Path(row["run"]) / "eval.jsonl")] = dict(
                arm=row["arm"], seed=row["seed"], group="baseline", format="eval",
                config_status="metrics header checked by old source; retained in old/inputs")
    runs = []
    for i, entry in enumerate(snapshot["sources"]):
        origin = entry["path"]
        stored = save(f"logs/{i:03d}.log.gz", gzip.compress(blobs[origin], mtime=0), origin)
        stored.update(bytes=entry["bytes"], sha256=entry["sha256"], compression="gzip")
        runs.append(dict(**stored, **identities[origin], count=entry["count"]))
    old_data["selection"] = [{k: v for k, v in entry.items() if k not in ("path", "compression")} for entry in runs]
    old_data["absent"] = absent
    ref = save_json("old/plotted-data.json", old_data)
    manifest = dict(schema=1, protocol="decay-20261007-v1", snapshot=snapshot["snapshot"],
                    old_source=old_source, reference=ref, banks=snapshot["banks"], runs=runs, absent=absent,
                    rules=dict(bin_width=5000, range=[0, 200000], x="bin centers / 1000",
                               complete_bins_only=True, smoothing="none", statistic="arithmetic mean",
                               median="not plotted", tails="thin observed seeds; mean uses all shown seeds",
                               planned_seeds=list(range(5)), stopping="saved log prefix; no extension",
                               selection="exact saved snapshot sources, including zero-record files",
                               config_differences="old/fixed collector, decay, hardware and historical controls remain distinct"))
    save_json("manifest.json", manifest)
    return dict(sources=len(runs), absent=len(absent), reference=ref)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualizations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.visualizations, args.output), indent=2))


if __name__ == "__main__":
    main()
