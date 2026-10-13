"""Validate or run a named recipe. No scheduler submission or implicit overrides."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import zipfile
from training.recipes import load_recipe
from training.registry import get_environment
from training.common.provenance import project_sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["validate", "train"])
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--vendor", type=Path, help="Explicit JaxMARL source root, if not installed")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--smoke", action="store_true", help="Explicit reduced test settings, not a research result")
    p.add_argument("--max-wall-seconds", type=int, default=600)
    args = p.parse_args()
    row = load_recipe(args.recipe, args.task)
    environment = json.loads(args.recipe.read_text())["environment"]
    adapter = get_environment(environment)
    cfg = row["config"].copy()
    if args.smoke:
        cfg.update(adapter.smoke_overrides(cfg))
    adapter.validate_config(cfg)
    if args.action == "validate":
        print(json.dumps(dict(row=row, resolved_config=cfg, smoke=args.smoke), indent=2))
        return
    if args.output is None or args.max_wall_seconds < 1:
        p.error("train requires a fresh --output and a positive time limit")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["CCE_RUN_DIR"] = str(out)
    os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "gpu" else "cpu"
    if args.device == "cpu":
        if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = "1"
    if args.vendor:
        sys.path.insert(0, str(args.vendor.resolve()))
    record = dict(status="starting", recipe=row, smoke=args.smoke,
                  environment=environment,
                  eval_axis=adapter.evaluation_axis(cfg),
                  resolved_config=cfg, recipe_sha256=hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
                  python=sys.version, wall_limit=args.max_wall_seconds)
    def save():
        (out/"attempt.json").write_text(json.dumps(record, indent=2))
    def deadline(signum, frame):
        raise TimeoutError("Training wall-time limit reached")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_wall_seconds)
    save()
    try:
        import jax
        if jax.default_backend() != args.device:
            raise RuntimeError("Requested backend is unavailable")
        record["packages"] = {k: importlib.metadata.version(k) for k in
            ("numpy", "scipy", "jax", "jaxlib", "flax", "optax", "chex")}
        for package in ("jaxmarl", "pgx", "rliable"):
            try:
                record["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                record["packages"][package] = None
        project_files = project_sources()
        record["source_hashes"] = {name: hashlib.sha256(f.read_bytes()).hexdigest()
            for name, f in project_files.items()}
        simulator_files = adapter.simulator_sources(cfg)
        simulator_hashes = {name: hashlib.sha256(f.read_bytes()).hexdigest()
                            for name, f in simulator_files.items()}
        record["simulator_source_hashes"] = simulator_hashes
        if environment == "jaxnav":
            record["jaxmarl_source_hashes"] = simulator_hashes
        # Hashes identify source; the bundle makes it recoverable after worktree cleanup.
        sources = [(project_files[name], "project/" + name, expected)
                   for name, expected in record["source_hashes"].items()]
        sources += [(simulator_files[name], "simulator/" + name, expected)
                    for name, expected in simulator_hashes.items()]
        if sum(f.stat().st_size for f, _, _ in sources) > 100 * 1024**2:
            raise ValueError("Source bundle exceeds 100 MiB cap")
        with zipfile.ZipFile(out/"source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for file, name, expected in sources:
                content = file.read_bytes()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError("Source changed while making attempt bundle")
                archive.writestr(name, content)
        record["source_bundle_sha256"] = hashlib.sha256((out/"source.zip").read_bytes()).hexdigest()
        record["device"] = [str(d) for d in jax.devices()]
        (out/"config.json").write_text(json.dumps(cfg, indent=2))
        save()
        agent = adapter.build_agent(cfg)
        if agent.config != cfg:
            raise ValueError("Resolved config drift")
        def record_bank():
            if hasattr(agent, "evaluation_identity"):
                record.update(agent.evaluation_identity())
                save()
                return
            if not hasattr(agent, "_fixed_eval_init_states"):
                return
            import numpy as np
            h = hashlib.sha256()
            for leaf in jax.tree.leaves((agent._fixed_eval_init_states, agent._fixed_eval_step_keys)):
                a = np.asarray(leaf)
                h.update(str((a.shape,a.dtype)).encode())
                h.update(a.tobytes())
            record["eval_bank_hash"] = h.hexdigest()
            save()
        original_evaluate = agent.evaluate
        def evaluate_with_identity(*values, **kwargs):
            result = original_evaluate(*values, **kwargs)
            record_bank()
            return result
        agent.evaluate = evaluate_with_identity
        record["status"] = "running"
        save()
        agent.learn(verbose=False)
        record_bank()
        record["status"] = "completed"
    except BaseException as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        signal.alarm(0)
        save()


if __name__ == "__main__":
    main()
