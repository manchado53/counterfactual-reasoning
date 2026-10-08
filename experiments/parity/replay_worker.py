"""Isolated replay exercise. Synthetic inputs; NOT a training/evaluation run."""
import argparse
import json
from pathlib import Path
import sys
import resource

import numpy as np


def pack(value):
    """Preserve array bits, shape and dtype, including signed zero."""
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        return {"dtype": a.dtype.str, "shape": list(a.shape),
                "bytes": a.tobytes().hex()}
    if isinstance(value, np.generic):
        return pack(np.asarray(value))
    if isinstance(value, float):
        return {"float64": np.float64(value).tobytes().hex()}
    if isinstance(value, dict):
        return {str(k): pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack(v) for v in value]
    return value


def buffer_kwargs(cfg):
    per = cfg["PER_parameters"]
    return dict(capacity=17, eps=per["eps"], beta=per["beta"],
                max_priority=per["maximum_priority"], mu=cfg["mu"],
                priority_mixing=cfg["priority_mixing"], mu_c=cfg["mu_c"],
                mu_delta=cfg["mu_delta"], normalise_priorities=cfg["normalise_priorities"],
                cce_balance=cfg.get("cce_balance"), target_ess_frac=cfg.get("target_ess_frac"),
                ess_recalib_every=cfg.get("ess_recalib_every", 50))


def exercise(cls, clock, row):
    cfg = row["config"]
    np.random.seed(row["seed"])
    b = cls(**buffer_kwargs(cfg))
    start, end = cfg["decay_start"], cfg["decay_end"]
    clocks = [start - 1, start, (start + end)//2, end, end + 1, start]
    trace = []
    for turn in range(12):
        x = np.arange(turn * 3, turn * 3 + 3)
        transitions = dict(r=(x % 3 == 0).astype(float), a=x % 15,
                           obs=np.column_stack((x, x + 1)).astype(np.float32))
        written = b.add_batch(transitions, jax_states={"position": x[:, None]})
        # Recorded synthetic scores include ties and zeroes. No CCE rollout is claimed.
        b.update_consequence_scores(written, (x % 4).astype(float), q_update_count=turn)
        ids = np.arange(len(b))
        for phase in ("scores", "td"):
            if phase == "td":
                b.update_priorities(ids, ((ids + turn) % 5 - 2).astype(float))
            for episode in clocks:
                b.blend = clock(episode, row["decay"], start, end)
                probabilities = b._compute_priorities().copy()
                b._compute_priorities()  # cache-hit path must not change state
                batch, sampled, weights = b.sample(min(4, len(b)))
                trace.append(pack(dict(turn=turn, phase=phase, episode=episode,
                    blend=b.blend, written=written, size=len(b), write_pos=b._write_pos,
                    scores=b.consequence_scores[:len(b)].copy(),
                    td=b.td_magnitudes[:len(b)].copy(), scored_at=b.scored_at[:len(b)].copy(),
                    probabilities=probabilities, sampled=sampled, weights=weights,
                    batch=batch, states=[b.get_jax_state(int(i)) for i in sampled],
                    rng=np.random.get_state())))
    return trace


def main():
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (55, 55))
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collector", choices=["old", "fixed"], required=True)
    parser.add_argument("--kind", choices=["reference", "candidate"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if args.kind == "reference":
        sys.path[:0] = [str(source / args.collector / "src"), str(source / "scripts")]
        from decay_buffer import DecayBuffer, blend_weight
    else:
        sys.path.insert(0, str(source / "src"))
        from counterfactual_rl.replay.decay import DecayBuffer, blend_weight
    import inspect
    actual = Path(inspect.getfile(DecayBuffer)).resolve()
    if not actual.is_relative_to(source):
        raise RuntimeError(f"Wrong source imported: {actual}")
    rows = [r for r in json.loads(args.manifest.read_text())["rows"]
            if r["mode"] == "train" and r["collector"] == args.collector]
    if not rows:
        raise ValueError("No training recipes selected")
    output = {str(r["task"]): exercise(DecayBuffer, blend_weight, r) for r in rows}
    args.output.write_text(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
