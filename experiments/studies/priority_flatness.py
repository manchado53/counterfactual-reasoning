EPS, BETA = 0.01, 0.25
import numpy as np

def shaped(scores):
    """Sampling distribution the buffer would use for these scores."""
    p = (np.asarray(scores) + EPS) ** BETA
    return p / p.sum()


def gini(q):
    x = np.sort(np.asarray(q, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def summarise(scores):
    c = np.asarray(scores)
    q = shaped(c)
    p = (c + EPS) ** BETA
    ess = 1.0 / np.sum(q ** 2)
    return dict(n=len(c), c_mean=float(c.mean()), c_std=float(c.std()),
                c_unique=int(len(np.unique(c))),
                c_frac_at_1=float(np.mean(c == 1.0)),
                p_ratio=float(p.max() / p.min()),
                ess=float(ess), ess_pct=float(100 * ess / len(c)),
                gini=gini(q))
