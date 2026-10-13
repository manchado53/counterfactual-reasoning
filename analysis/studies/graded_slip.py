import json
import numpy as np
NOISE_PEAK = 2.0 / 3.0

def slip_entropy(p):
    """Shannon entropy (nats) of the outcome distribution [p/2, 1-p, p/2].

    This is the actual 'how noisy is the environment' axis. It rises from 0 at
    p=0 to its maximum ln(3)=1.099 at p=2/3, then falls back to ln(2)=0.693 at
    p=1 where the intended action becomes impossible.
    """
    d = np.array([p / 2.0, 1.0 - p, p / 2.0], dtype=float)
    d = d[d > 0]
    return float(-(d * np.log(d)).sum())


def _load_cache(npz_path):
    """Reverse of build_graded_slip_cache.py: rebuild the per-slip parsed arrays
    from the .npz so figures regenerate WITHOUT the raw run dirs.

    Returns (raw_by_slip, evalsteps_by_slip, n_submitted, dropped).
    """
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    raw_by_slip, evalsteps_by_slip = {}, {}
    for p in meta["slips"]:
        k = f"{p:.3f}"
        labels = meta["groups"][k]["labels"]
        raw_by_slip[float(p)] = {a: z[f"slip{k}_raw_{i}"] for i, a in enumerate(labels)}
        evalsteps_by_slip[float(p)] = {a: z[f"slip{k}_steps_{i}"] for i, a in enumerate(labels)}
    return (raw_by_slip, evalsteps_by_slip,
            meta.get("n_submitted"), meta.get("dropped_runs", []))


def _split_main_probe(slips):
    """Split slip levels into the noise curve (<= 2/3) and the probe (> 2/3)."""
    main = [p for p in slips if p <= NOISE_PEAK + 1e-9]
    probe = [p for p in slips if p > NOISE_PEAK + 1e-9]
    return main, probe
