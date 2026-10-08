"""Build the reproducibility cache for the graded-stochasticity experiment.

Parses the 200-run graded-slip sweep (from the raw run dirs, while they still
exist) into rliable-ready arrays, grouped by slip level, and saves them as one
small .npz so the graded-slip figures can be rebuilt WITHOUT the raw runs.

Mirrors build_cache.py, but the graded sweep has one raw-array group PER slip
level, and it applies the same dead-run filter the analysis uses.

Run from the repo root (with the runs still on disk):
    PYTHONPATH=src python paper/repro/build_graded_slip_cache.py

Output -> paper/repro/cache/claim2_graded_slip.npz
Rebuild figures from it later with:
    PYTHONPATH=src python -m counterfactual_rl.analysis.claim2.graded_slip \
        --from-cache paper/repro/cache/claim2_graded_slip.npz --out <dir>
(the --from-cache reader is documented in the provenance note)
"""
import json
import os
import tempfile

import numpy as np

from counterfactual_rl.analysis.claim2.parse_logs import (
    load_manifest, filter_complete_runs,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifests", "claim2_graded_slip_2026-08-03.json")
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)


def build():
    manifest = json.load(open(MANIFEST))
    n_submitted = len(manifest)
    manifest, dropped = filter_complete_runs(manifest)
    print(f"kept {len(manifest)}/{n_submitted}, dropped {len(dropped)} dead runs")

    groups = {}
    for jid, cfg in manifest.items():
        p = float(cfg.get("slip_prob", 2.0 / 3.0))
        groups.setdefault(p, {})[jid] = cfg

    out = {}
    meta = {"manifest": os.path.basename(MANIFEST), "n_submitted": n_submitted,
            "n_used": len(manifest), "dropped_runs": dropped, "slips": [], "groups": {}}

    with tempfile.TemporaryDirectory() as tmp:
        for p in sorted(groups):
            sub = os.path.join(tmp, f"slip_{p:.3f}.json")
            json.dump(groups[p], open(sub, "w"))
            data = load_manifest(sub)
            raw, steps = data["raw"], data["eval_steps"]
            labels = list(raw.keys())
            key = f"{p:.3f}"
            for i, a in enumerate(labels):
                out[f"slip{key}_raw_{i}"] = raw[a]
                out[f"slip{key}_steps_{i}"] = steps[a]
            meta["slips"].append(p)
            meta["groups"][key] = {
                "labels": labels,
                "n_seeds": {a: int(raw[a].shape[0]) for a in labels},
                "n_checkpoints": {a: int(raw[a].shape[2]) for a in labels},
            }
            print(f"  slip {key}: " + ", ".join(f"{a}={raw[a].shape}" for a in labels))

    out["meta"] = json.dumps(meta)
    dst = os.path.join(CACHE, "claim2_graded_slip.npz")
    np.savez_compressed(dst, **out)
    print(f"\nDone -> {dst}  ({os.path.getsize(dst)//1024} KB)")


if __name__ == "__main__":
    build()
