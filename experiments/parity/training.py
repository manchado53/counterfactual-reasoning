"""Actual learn-loop parity on bounded smoke settings, never a full-budget claim."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import importlib.metadata
from experiments.parity.run import snapshot, digest, dump, first_difference, MAX_SOURCE_BYTES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", default="1,7,13,18,21,24,27,30,33")
    args = p.parse_args()
    tasks = [int(t) for t in args.tasks.split(",")]
    if not tasks or len(tasks) > 9 or len(set(tasks)) != len(tasks):
        p.error("Choose 1..9 distinct tasks")
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    if out.is_relative_to(args.reference.resolve()):
        p.error("Output must not be inside reference")
    out.mkdir(parents=True, exist_ok=False)
    report = dict(scope="bounded-real-jaxnav-learn-loop", status="running", checks=[],
                  production_budget_parity="NOT_TESTED", direct_gradient_capture="NOT_TESTED",
                  limits=dict(worker_wall_seconds=300, cpu_affinity=4, workers=1,
                              worker_address_bytes=16*1024**3, max_tasks=9))
    try:
        report["reference_snapshot"] = snapshot(args.reference.resolve(), out/"reference")
        pinned = json.loads((out/"reference/versions.json").read_text())
        actual = {k: importlib.metadata.version(k) for k in pinned["packages"]}
        report["runtime"] = dict(python=sys.version,packages=actual)
        if sys.version != pinned["python"] or actual != pinned["packages"]:
            raise ValueError("Training parity requires the frozen runtime versions")
        # Recoverable candidate copy; exclude existing outputs and pycache.
        paths = [f for directory in ("src", "experiments") for f in (root/directory).rglob("*.py")]
        if sum(f.stat().st_size for f in paths) > MAX_SOURCE_BYTES:
            raise ValueError("Candidate too large")
        hashes = {}
        for f in paths:
            rel = f.relative_to(root)
            dest = out/"candidate"/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)
            hashes[str(rel)] = digest(dest)
        report["candidate_hashes"] = hashes
        worker = out/"candidate/experiments/parity/training_worker.py"
        env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONNOUSERSITE="1", PYTHONPATH="",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   JAX_COMPILATION_CACHE_DIR=str(out/"compile-cache"),
                   JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0")
        for task in tasks:
            traces = {}
            for name, source, kind in (("old_a",out/"reference","reference"),
                                       ("old_b",out/"reference","reference"),
                                       ("new",out/"candidate","candidate")):
                target = out/f"task-{task}-{name}"
                print(f"START task={task} copy={name}", flush=True)
                proc = subprocess.run([sys.executable,str(worker), "--source",str(source),
                    "--reference",str(out/"reference"),"--kind",kind,"--task",str(task),
                    "--output",str(target)], cwd=out, env=env, capture_output=True,
                    text=True, timeout=300)
                (out/f"task-{task}-{name}.log").write_text(proc.stdout + proc.stderr)
                if proc.returncode:
                    raise RuntimeError(f"task={task}/{name}: {proc.stderr[-2500:]}")
                traces[name] = json.loads((target/"trace.json").read_text())
            repeat = first_difference(traces["old_a"],traces["old_b"])
            mismatch = first_difference(traces["old_a"],traces["new"])
            check = dict(task=task,repeat_difference=repeat,parity_difference=mismatch,
                         passed=repeat is None and mismatch is None,
                         events=len(traces["old_a"]["events"]),
                         smoke_overrides=traces["old_a"]["overrides"])
            report["checks"].append(check)
            dump(out/"report.json",report)
            print(json.dumps(check),flush=True)
            if not check["passed"]:
                raise ValueError("Parity failed; stopped at first mismatch")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed",error=str(exc))
    dump(out/"report.json",report)
    print(f"{report['status']}: {out/'report.json'}",flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
