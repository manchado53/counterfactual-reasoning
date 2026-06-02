"""Build small reproducibility caches for the paper figures.

For each Claim-2 manifest it parses the per-run metrics (from the raw run dirs,
wherever they live) into the rliable-ready arrays and saves them as a small
.npz, so the figures can be regenerated WITHOUT the ~260 GB of run directories.
It also caches the exact Claim-1 oracle (cheap value iteration).

Run from the repo root:
    PYTHONPATH=src python paper/repro/build_cache.py

Outputs -> paper/repro/cache/*.npz
"""
import json
import os

import numpy as np

from counterfactual_rl.analysis.claim2.parse_logs import load_manifest
from counterfactual_rl.analysis.claim1.frozen_lake.oracle import compute_oracle

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, "manifests")
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

# name -> (manifest filename, analysis env, threshold used)
CLAIM2 = {
    "claim2_frozen_lake_no_slip": ("claim2_no_slip_2026-05-09.json", "frozen_lake_no_slip", 0.75),
    "claim2_frozen_lake":         ("claim2_main_merged.json",        "frozen_lake",         0.75),
    "claim2_smax_3m":             ("claim2_main_3m_2026-05-07.json",  "smax_3m",             0.60),
}


def build_claim2(name, manifest_file, env, threshold):
    path = os.path.join(MAN, manifest_file)
    data = load_manifest(path)
    raw = data["raw"]
    eval_steps = data["eval_steps"]
    raw_length = data.get("raw_length") or {}
    raw_allies = data.get("raw_allies") or {}

    labels = list(raw.keys())
    out = {}
    for i, a in enumerate(labels):
        out[f"raw_{i}"] = raw[a]
        out[f"steps_{i}"] = eval_steps[a]
        if a in raw_length and raw_length[a] is not None:
            out[f"len_{i}"] = raw_length[a]
        if a in raw_allies and raw_allies[a] is not None:
            out[f"allies_{i}"] = raw_allies[a]

    meta = {
        "manifest": manifest_file,
        "env": env,
        "threshold": threshold,
        "env_type": data.get("env_type"),
        "labels": labels,
        "n_seeds": {a: int(raw[a].shape[0]) for a in labels},
        "n_checkpoints": {a: int(raw[a].shape[2]) for a in labels},
    }
    out["meta"] = json.dumps(meta)
    np.savez_compressed(os.path.join(CACHE, name + ".npz"), **out)
    print(f"  {name}: " + ", ".join(f"{a}={raw[a].shape}" for a in labels))


def build_claim1_oracle():
    Q, oracle, non_terminal = compute_oracle(map_name="8x8", is_slippery=True, gamma=0.99)
    meta = {
        "map": "8x8", "is_slippery": True, "gamma": 0.99,
        "oracle_def": "mean_{a!=a*} |Q*(s,a*) - Q*(s,a)|",
        "n_non_terminal": len(non_terminal),
    }
    np.savez_compressed(
        os.path.join(CACHE, "claim1_frozen_lake_oracle.npz"),
        Q=Q,
        oracle_vals=np.array([oracle[s] for s in non_terminal]),
        non_terminal=np.array(non_terminal),
        meta=json.dumps(meta),
    )
    print(f"  claim1 oracle: {len(non_terminal)} non-terminal states (slippery)")


if __name__ == "__main__":
    print("Building Claim-2 caches:")
    for name, (mf, env, th) in CLAIM2.items():
        build_claim2(name, mf, env, th)
    print("Building Claim-1 oracle cache:")
    build_claim1_oracle()
    print(f"\nDone -> {CACHE}")
