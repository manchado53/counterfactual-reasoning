import numpy as np
from scipy.stats import spearmanr

def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def spearman_boot(a, b, reps=2000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float('nan'), (float('nan'), float('nan'))
    rho = spearmanr(a, b).correlation
    rng = np.random.default_rng(seed)
    boots = [spearmanr(a[i], b[i]).correlation for i in (rng.integers(0, a.size, a.size) for _ in range(reps))]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(rho), (float(lo), float(hi))
