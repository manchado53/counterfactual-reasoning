import numpy as np
from .priority_flatness import EPS, BETA

def q_star(env, gamma, tol=1e-12, max_iter=100_000):
    desc = [ch for row in env.desc for ch in row]
    terminal = {s for s, ch in enumerate(desc) if ch in ("H", "G")}
    V = np.zeros(env.n_states)
    for _ in range(max_iter):
        delta = 0.0
        for s in range(env.n_states):
            if s in terminal:
                continue
            v = max(sum(p * (r + gamma * V[ns]) for p, ns, r, _ in env.P[s][a])
                    for a in range(env.n_actions))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    Q = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        if s in terminal:
            continue
        for a in range(env.n_actions):
            Q[s, a] = sum(p * (r + gamma * V[ns]) for p, ns, r, _ in env.P[s][a])
    return Q, sorted(set(range(env.n_states)) - terminal)


def enumerate_transitions(env, non_terminal, drop_zero_prob=True):
    """Every (s, a, outcome) the buffer could hold. Returns a dict of arrays."""
    s_l, a_l, k_l, p_l, ns_l, r_l, d_l = [], [], [], [], [], [], []
    for s in non_terminal:
        for a in range(env.n_actions):
            for k, (p, ns, r, done) in enumerate(env.P[s][a]):
                if drop_zero_prob and p <= 0.0:
                    continue
                s_l.append(s); a_l.append(a); k_l.append(k); p_l.append(p)
                ns_l.append(ns); r_l.append(r); d_l.append(bool(done))
    return dict(s=np.array(s_l), a=np.array(a_l), k=np.array(k_l),
                prob=np.array(p_l, dtype=float), ns=np.array(ns_l),
                r=np.array(r_l, dtype=float), done=np.array(d_l))


def _cov(x, y):
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def predicate(c, d, u):
    """Theorem 3's two sides, plus the direct expectations they are equivalent to."""
    Ec, Ed = float(c.mean()), float(d.mean())
    out = dict(E_c=Ec, E_d=Ed, E_u=float(u.mean()),
               cov_cu=_cov(c, u), cov_du=_cov(d, u))
    out["lhs"] = out["cov_cu"] / Ec if Ec > 0 else float("nan")   # CCE side
    out["rhs"] = out["cov_du"] / Ed if Ed > 0 else float("nan")   # TD side
    out["cce_wins"] = bool(out["lhs"] >= out["rhs"])
    # direct form: E_{p^c}[u] vs E_{p^d}[u], the thing the ratios are equivalent to
    out["E_pc_u"] = float((c * u).sum() / c.sum()) if c.sum() > 0 else float("nan")
    out["E_pd_u"] = float((d * u).sum() / d.sum()) if d.sum() > 0 else float("nan")
    return out


def deployed_predicate(c, d, u):
    """Same predicate for the sampler actually shipped: p ∝ (score + eps)^beta.

    The theorems are stated for p ∝ score. The code ships p ∝ (score+eps)^beta,
    which is a different measure -- so the predicate is reported both ways.
    """
    return predicate((c + EPS) ** BETA, (d + EPS) ** BETA, u)


def check_identities(c, d, u, tol=1e-9):
    """Theorems 3 and 4 are algebra in c, d, u. If these fail, weighting is wrong."""
    res = {}
    p = predicate(c, d, u)
    # Thm 3: E_{p^c}[u] - E_{p^d}[u]  ==  lhs - rhs
    res["thm3_residual"] = abs((p["E_pc_u"] - p["E_pd_u"]) - (p["lhs"] - p["rhs"]))
    # Thm 4: E_{p^cd}[u] - E_{p^d}[u] == Cov_{p^d}(c,u) / E_{p^d}[c]
    w = d / d.sum()
    E_pd_c = float((w * c).sum())
    E_pd_u = float((w * u).sum())
    cov_pd_cu = float((w * c * u).sum() - E_pd_c * E_pd_u)
    lhs4 = float((c * d * u).sum() / (c * d).sum()) - E_pd_u
    rhs4 = cov_pd_cu / E_pd_c
    res["thm4_residual"] = abs(lhs4 - rhs4)
    res["thm4_lhs"] = lhs4
    res["thm4_rhs"] = rhs4
    res["ok"] = bool(res["thm3_residual"] < tol and res["thm4_residual"] < tol)
    return res
