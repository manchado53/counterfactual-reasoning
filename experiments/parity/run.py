"""Pin a sealed JaxNav study and compare replay in separate CPU processes.

Usage: python -m experiments.parity.run --reference SEALED_SUITE --output NEW_DIR
Never launches SLURM jobs, trains an agent, or edits the reference.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


MAX_SOURCE_BYTES = 100 * 1024 * 1024
WORKER_TIMEOUT = 60


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def safe_file(root, name):
    path = root / name
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"Unsafe manifest path: {name}")
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Source escapes root: {name}")
    if not path.is_file():
        raise ValueError(f"Missing source: {name}")
    return path


def snapshot(reference, destination):
    hashes = json.loads((reference / "hashes.json").read_text())
    required = {"manifest.json", "versions.json", "scripts/decay_buffer.py"}
    for collector in ("old", "fixed"):
        required.add(f"{collector}/src/counterfactual_rl/agents/shared/consequence_buffers.py")
    if not required.issubset(hashes):
        raise ValueError("Sealed reference lacks required hashed inputs")
    total = 0
    for name, expected in hashes.items():
        path = safe_file(reference, name)
        total += path.stat().st_size
        if total > MAX_SOURCE_BYTES:
            raise ValueError("Reference exceeds 100 MiB source cap")
        if digest(path) != expected:
            raise ValueError(f"Reference hash mismatch: {name}")
    destination.mkdir()
    for name, expected in hashes.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_file(reference, name), target)
        if digest(target) != expected:
            raise ValueError(f"Source changed during capture: {name}")
    dump(destination / "hashes.json", hashes)
    return dict(files=len(hashes), bytes=total,
                manifest_sha256=hashes["manifest.json"], hashes_sha256=digest(destination / "hashes.json"))


def first_difference(left, right, path="$"):
    if type(left) is not type(right):
        return path + ": type"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path + ": keys"
        for key in left:
            mismatch = first_difference(left[key], right[key], path + "." + key)
            if mismatch:
                return mismatch
    elif isinstance(left, list):
        if len(left) != len(right):
            return path + ": length"
        for index, (a, b) in enumerate(zip(left, right)):
            mismatch = first_difference(a, b, f"{path}[{index}]")
            if mismatch:
                return mismatch
    elif left != right:
        return path
    return None


def run(reference, candidate, output):
    if output.exists():
        raise ValueError("Refusing to reuse an output directory")
    if output.is_relative_to(reference):
        raise ValueError("Output must not be inside the protected reference")
    # Keep this explicit CPU check small; not production training parity.
    output.mkdir(parents=True)
    report = dict(schema_version=1, status="running", scope="replay-and-decay-only",
        full_training_parity="NOT_TESTED", reference=str(reference), candidate=str(candidate),
        limits=dict(source_bytes=MAX_SOURCE_BYTES, worker_seconds=WORKER_TIMEOUT,
                    worker_address_bytes=8 * 1024**3, worker_file_bytes=64 * 1024**2,
                    worker_cpu_seconds=55, workers_at_once=1,
                    buffer_capacity=17, insertions_per_recipe=36, device="cpu"),
        skipped=["collector", "CCE rollout scores", "Q updates/gradients", "eval bank",
                 "historical figures", "GPU parity", "resume"],
        checks=[])
    try:
        captured = output / "reference"
        report["reference_snapshot"] = snapshot(reference, captured)
        versions = json.loads((captured / "versions.json").read_text())
        actual = {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "jax", "jaxlib")}
        report["runtime"] = dict(python=sys.version, packages=actual)
        if sys.version != versions["python"] or any(actual[k] != versions["packages"][k] for k in actual):
            raise ValueError("Use the pinned reference Python/numpy/scipy/JAX runtime")
        files = ["src/counterfactual_rl/replay/consequence_buffers.py",
                 "src/counterfactual_rl/replay/decay.py",
                 "src/counterfactual_rl/replay/__init__.py",
                 "src/counterfactual_rl/__init__.py", "src/counterfactual_rl/agents/__init__.py"]
        candidate_copy = output / "candidate"
        # Capture full Python source so the imported package can be recovered.
        source_files = sorted((candidate / "src").rglob("*.py"))
        if sum(p.stat().st_size for p in source_files) > MAX_SOURCE_BYTES:
            raise ValueError("Candidate exceeds source cap")
        for rel in files:
            safe_file(candidate, rel)
        candidate_hashes = {}
        for src in source_files:
            rel = str(src.relative_to(candidate))
            safe_file(candidate, rel)
            dst = candidate_copy / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            candidate_hashes[rel] = digest(dst)
        report["candidate_sources"] = candidate_hashes
        rows = json.loads((captured / "manifest.json").read_text())["rows"]
        training = [r for r in rows if r["mode"] == "train"]
        if not training or len(training) > 100:
            raise ValueError("Expected 1..100 bounded training recipes")
        if len({r["task"] for r in rows}) != len(rows):
            raise ValueError("Duplicate task IDs")
        if any(r["collector"] not in ("old", "fixed") for r in training):
            raise ValueError("Unsupported collector")
        worker = Path(__file__).with_name("replay_worker.py")
        shutil.copyfile(worker, output / "replay_worker.py")
        shutil.copyfile(Path(__file__), output / "run.py")
        report["harness_sha256"] = {n: digest(output/n) for n in ("run.py", "replay_worker.py")}
        env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONNOUSERSITE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHONPATH="")
        for collector in ("old", "fixed"):
            traces = {}
            for name, source, kind in (("old_a", captured, "reference"),
                                       ("old_b", captured, "reference"),
                                       ("new", candidate_copy, "candidate")):
                target = output / f"{collector}_{name}.json"
                proc = subprocess.run([sys.executable, str(output / "replay_worker.py"),
                    "--source", str(source), "--manifest", str(captured / "manifest.json"),
                    "--collector", collector, "--kind", kind, "--output", str(target)],
                    cwd=output, env=env, capture_output=True, text=True, timeout=WORKER_TIMEOUT)
                (output/f"{collector}_{name}.stderr").write_text(proc.stderr)
                if proc.returncode:
                    raise RuntimeError(f"{collector}/{name} failed: {proc.stderr[-2000:]}")
                traces[name] = json.loads(target.read_text())
            for row in rows:
                if row["mode"] != "train" or row["collector"] != collector:
                    continue
                key = str(row["task"])
                repeat = first_difference(traces["old_a"][key], traces["old_b"][key])
                parity = first_difference(traces["old_a"][key], traces["new"][key])
                report["checks"].append(dict(task=row["task"], arm=row["arm"], seed=row["seed"],
                    collector=collector, decay=row["decay"], repeat_first_difference=repeat,
                    parity_first_difference=parity, passed=repeat is None and parity is None,
                    comparisons=len(traces["old_a"][key])))
        report["status"] = "passed" if report["checks"] and all(c["passed"] for c in report["checks"]) else "failed"
    except Exception as exc:
        report["status"] = "error"
        report["error"] = str(exc)
    dump(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    result = run(args.reference.resolve(), args.candidate.resolve(), args.output.resolve())
    print(json.dumps({k: result[k] for k in ("status", "scope", "full_training_parity")}))
    print(f"Report: {args.output.resolve() / 'report.json'}")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
