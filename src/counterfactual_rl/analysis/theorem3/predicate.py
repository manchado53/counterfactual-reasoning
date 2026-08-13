"""
Plan step 2 — measure the predicate: c, d and u over the full transition space.

Theorem 3 says CCE priority beats TD priority exactly when

    Cov(c, u) / E[c]  >=  Cov(d, u) / E[d]

Both sides are numbers. This module computes them directly on FrozenLake 8x8,
whose transition space is small enough to enumerate as the buffer:

    53 non-terminal states x 4 actions x 3 outcomes = 636 transitions

636, not 212. Under slip one (s, a) has three outcomes with different next
states, different realised TD, and different utility; the buffer stores them
separately, and the reducible/irreducible noise split is impossible without
that granularity. At slip 0 two slots carry probability 0 and are dropped.

Quantities per transition i = (s, a, outcome k)
-----------------------------------------------
c_i   CCE score of (s, a), from policy rollouts. Computed under BOTH `max` and
      `mean` aggregation (see issue #3 — FrozenLake silently uses `max`).
d_i   |TD error|, bootstrapped off the TARGET net because that is what training
      uses.
u_i   replay utility -- how much replaying this one transition moves Q toward
      exact Q*, where Q* comes from value iteration on env.P.

Landmines this module is written against
----------------------------------------
1. **Adam destroys u.** From a fresh optimiser state Adam's first step is
   -lr * sign(g): gradient magnitude cancels, so u degenerates into a sign
   pattern carrying no information. We use the exact directional derivative
   <grad E, grad loss_i> instead, which is the SGD utility to first order and
   gets all 636 transitions in one JVP rather than 636 network clones.
2. **u factors as delta x G.** One-step utility is signed TD error times how
   the network propagates that update into global error. PER wins the delta
   half by construction, so Cov(c, delta) and Cov(c, G) are reported
   separately -- G is the real question.
3. **Wrong-MDP scoring.** Every checkpoint is loaded via
   `FrozenLakeDQN.from_checkpoint`, which asserts the env matches.
4. **Per-(s,a) collapse.** The full (S, A) score matrix is kept; nothing is
   collapsed to a per-state scalar.

Caveats
-------
- The predicate is evaluated with the buffer taken to be the enumerated
  transition space weighted uniformly. A live buffer is visitation-weighted,
  which is a different measure.
- u is a first-order, one-step quantity. It is myopic by construction; step 3
  (falsification) is what tests whether it deserves the name "utility".
- Theorems 3 and 4 are algebraic identities in c, d, u. Computing both sides
  here validates the code, not the theory -- see `--check-identities`.

Usage
-----
    python -m counterfactual_rl.analysis.theorem3.predicate
    python -m counterfactual_rl.analysis.theorem3.predicate --stages trained
"""
import argparse
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

from counterfactual_rl.agents.frozen_lake.dqn import FrozenLakeDQN
from counterfactual_rl.analysis.metrics import compute_total_variation
from counterfactual_rl.analysis.theorem3.priority_flatness import (
    EPS, BETA, build_rollout_fn, OUT_DIR,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
REPRO_CKPTS = os.path.join(_REPO, "paper", "repro", "cache", "checkpoints")
STAGES = ("untrained", "mid", "trained")
SEEDS = (0, 1, 2)


# --------------------------------------------------------------------------
# exact Q* — value iteration on env.P (works at any slip, unlike claim1.oracle
# which only takes the binary is_slippery flag)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# c, d, u
# --------------------------------------------------------------------------
def cce_matrix(agent, non_terminal, n_rollouts=20, seed=0):
    """(S, A) CCE score per (state, taken action), under both aggregations."""
    env, cfg = agent.env, agent.config
    horizon = int(cfg.get("cf_horizon", 200))
    gamma = float(cfg.get("cf_gamma", 0.99))
    rollout = build_rollout_fn(env, agent.network, horizon, gamma)
    B, A = len(non_terminal), env.n_actions
    keys = jax.random.split(jax.random.PRNGKey(seed),
                            B * A * n_rollouts).reshape(B, A, n_rollouts, 2)
    R = np.array(rollout(agent.params, jnp.array(non_terminal, jnp.int32),
                         jnp.arange(A, dtype=jnp.int32), keys))     # (B, A, N)

    out = {"max": np.zeros((B, A)), "mean": np.zeros((B, A))}
    for i in range(B):
        for a in range(A):
            divs = [compute_total_variation(R[i, a], R[i, b])
                    for b in range(A) if b != a]
            out["max"][i, a] = max(divs)
            out["mean"][i, a] = float(np.mean(divs))
    return out


def td_and_utility(agent, Qs, tr, non_terminal, gamma):
    """Signed TD, its noise split, and utility u = delta * G.

    G_i = -2 <grad E, grad Q(s_i, a_i)>, so u_i = delta_i * G_i exactly.
    E = mean |Q_theta - Q*| over all non-terminal (s, a).
    """
    params, tparams = agent.params, agent.target_params
    net = agent.network
    states = jnp.array(non_terminal, dtype=jnp.int32)
    Qs_nt = jnp.array(Qs[np.array(non_terminal)])           # (B, A) exact Q*

    # --- signed TD error off the TARGET net ------------------------------
    q_all = np.array(jax.vmap(lambda s: net.apply(params, s))(states))   # (B, A)
    idx = {s: i for i, s in enumerate(non_terminal)}
    q_next_t = np.array(jax.vmap(lambda s: net.apply(tparams, s))(
        jnp.array(tr["ns"], dtype=jnp.int32)))                            # (T, A)

    q_sa = np.array([q_all[idx[s], a] for s, a in zip(tr["s"], tr["a"])])
    boot = np.where(tr["done"], 0.0, q_next_t.max(axis=1))
    y = tr["r"] + gamma * boot
    delta = y - q_sa                       # signed
    d = np.abs(delta)

    # --- noise split, exact from Q* --------------------------------------
    # At Q = Q* the EXPECTED TD is zero but the PER-OUTCOME TD is not; that
    # residual spread is the irreducible noise.
    qs_next = Qs[tr["ns"]]                                              # (T, A)
    boot_star = np.where(tr["done"], 0.0, qs_next.max(axis=1))
    q_star_sa = Qs[tr["s"], tr["a"]]
    delta_star = tr["r"] + gamma * boot_star - q_star_sa
    eps_n = np.abs(delta_star)             # irreducible
    eps_r = d - eps_n                      # reducible (can go negative -- reported)

    # --- utility via one JVP ---------------------------------------------
    def global_err(p):
        q = jax.vmap(lambda s: net.apply(p, s))(states)
        return jnp.mean(jnp.abs(q - Qs_nt))

    grad_E = jax.grad(global_err)(params)

    tr_s = jnp.array(tr["s"], dtype=jnp.int32)
    tr_a = jnp.array(tr["a"], dtype=jnp.int32)

    def q_taken(p):
        q = jax.vmap(lambda s: net.apply(p, s))(tr_s)      # (T, A)
        return q[jnp.arange(q.shape[0]), tr_a]             # (T,)

    _, dq = jax.jvp(q_taken, (params,), (grad_E,))         # <grad Q_i, grad E>
    G = -2.0 * np.array(dq)
    u = delta * G
    return dict(delta=delta, d=d, eps_n=eps_n, eps_r=eps_r, G=G, u=u,
                E=float(global_err(params)))


def verify_linearisation(agent, Qs, tr, non_terminal, gamma, u, n=6, lr=1e-3, seed=0):
    """Sanity-check the directional derivative against real SGD steps.

    Landmine 1 says do not use the trainer's Adam step. This does plain SGD by
    hand on a few transitions and compares the measured error drop against the
    first-order prediction.
    """
    net, params = agent.network, agent.params
    states = jnp.array(non_terminal, dtype=jnp.int32)
    Qs_nt = jnp.array(Qs[np.array(non_terminal)])

    def err(p):
        q = jax.vmap(lambda s: net.apply(p, s))(states)
        return float(jnp.mean(jnp.abs(q - Qs_nt)))

    base = err(params)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(u), size=min(n, len(u)), replace=False)
    rows = []
    for i in pick:
        s_i = int(tr["s"][i]); a_i = int(tr["a"][i])
        q_next_t = net.apply(agent.target_params, jnp.int32(tr["ns"][i]))
        y_i = tr["r"][i] + (0.0 if tr["done"][i] else gamma * float(jnp.max(q_next_t)))

        def loss(p):
            q = net.apply(p, jnp.int32(s_i))
            return (q[a_i] - y_i) ** 2

        g = jax.grad(loss)(params)
        stepped = jax.tree.map(lambda w, gg: w - lr * gg, params, g)
        rows.append((float(u[i]), (base - err(stepped)) / lr))
    return rows


# --------------------------------------------------------------------------
# the predicate
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run_checkpoint(ckpt, label, n_rollouts=20, verify=True):
    agent = FrozenLakeDQN.from_checkpoint(ckpt)
    env = agent.env
    gamma = float(agent.config.get("gamma", 0.99))

    Qs, non_terminal = q_star(env, gamma)
    tr = enumerate_transitions(env, non_terminal)
    cmat = cce_matrix(agent, non_terminal, n_rollouts=n_rollouts)
    tu = td_and_utility(agent, Qs, tr, non_terminal, gamma)

    idx = {s: i for i, s in enumerate(non_terminal)}
    rec = dict(label=label, ckpt=ckpt, n_transitions=int(len(tr["s"])),
               slip=float(env.slip_prob), gamma=gamma,
               global_err=tu["E"],
               frac_eps_r_negative=float(np.mean(tu["eps_r"] < 0)),
               mean_eps_n=float(tu["eps_n"].mean()),
               mean_eps_r=float(tu["eps_r"].mean()),
               cov_epsn_u=_cov(tu["eps_n"], tu["u"]),      # Cor 1's premise
               corr_epsn_u=float(np.corrcoef(tu["eps_n"], tu["u"])[0, 1])
               if tu["eps_n"].std() > 0 else float("nan"))

    for agg in ("max", "mean"):
        c = np.array([cmat[agg][idx[s], a] for s, a in zip(tr["s"], tr["a"])])
        rec[f"{agg}_frac_c_zero"] = float(np.mean(c == 0))
        rec[f"{agg}_c_mean"] = float(c.mean())
        if c.sum() <= 0:
            rec[f"{agg}_predicate"] = None      # no CCE signal: 0/0, not a bug
            continue
        rec[f"{agg}_predicate"] = predicate(c, tu["d"], tu["u"])
        rec[f"{agg}_deployed"] = deployed_predicate(c, tu["d"], tu["u"])
        rec[f"{agg}_identities"] = check_identities(c, tu["d"], tu["u"])
        # landmine 2: u = delta * G, so split the covariance
        rec[f"{agg}_cov_c_delta"] = _cov(c, tu["d"])
        rec[f"{agg}_cov_c_G"] = _cov(c, tu["G"])

    if verify:
        rec["linearisation_check"] = verify_linearisation(
            agent, Qs, tr, non_terminal, gamma, tu["u"])
    return rec


def main(stages=STAGES, seeds=SEEDS, n_rollouts=20):
    records = []
    for seed in seeds:
        for stage in stages:
            ckpt = os.path.join(REPRO_CKPTS, f"seed_{seed}", f"{stage}.pkl")
            if not os.path.exists(ckpt):
                print(f"missing {ckpt}")
                continue
            label = f"seed{seed}/{stage}"
            rec = run_checkpoint(ckpt, label, n_rollouts=n_rollouts)
            records.append(rec)

            print(f"\n=== {label} ===")
            print(f"  transitions {rec['n_transitions']}  slip {rec['slip']:.3f}  "
                  f"mean|Q-Q*| {rec['global_err']:.4f}")
            print(f"  eps_n mean {rec['mean_eps_n']:.4f}  eps_r mean "
                  f"{rec['mean_eps_r']:.4f}  frac eps_r<0 "
                  f"{rec['frac_eps_r_negative']:.3f}")
            print(f"  Cor1 premise  Cov(eps_n, u) = {rec['cov_epsn_u']:+.3e}  "
                  f"(corr {rec['corr_epsn_u']:+.3f})")
            for agg in ("max", "mean"):
                p = rec.get(f"{agg}_predicate")
                if p is None:
                    print(f"  {agg:>4}: no CCE signal "
                          f"(frac c==0 = {rec[f'{agg}_frac_c_zero']:.3f}) -- predicate skipped")
                    continue
                ident = rec[f"{agg}_identities"]
                dep = rec[f"{agg}_deployed"]
                print(f"  {agg:>4}: LHS Cov(c,u)/E[c] = {p['lhs']:+.3e}   "
                      f"RHS Cov(d,u)/E[d] = {p['rhs']:+.3e}   "
                      f"-> {'CCE' if p['cce_wins'] else 'TD'} wins")
                print(f"        deployed (score+eps)^beta: LHS {dep['lhs']:+.3e}  "
                      f"RHS {dep['rhs']:+.3e}  -> "
                      f"{'CCE' if dep['cce_wins'] else 'TD'} wins")
                print(f"        Cov(c,delta) {rec[f'{agg}_cov_c_delta']:+.3e}   "
                      f"Cov(c,G) {rec[f'{agg}_cov_c_G']:+.3e}   "
                      f"(landmine 2: G is the real question)")
                print(f"        identities thm3 {ident['thm3_residual']:.2e}  "
                      f"thm4 {ident['thm4_residual']:.2e}  "
                      f"{'OK' if ident['ok'] else 'FAIL'}")
            lc = rec.get("linearisation_check")
            if lc:
                pred = np.array([a for a, _ in lc])
                meas = np.array([b for _, b in lc])
                ok = np.allclose(pred, meas, rtol=0.05, atol=1e-6)
                print(f"  linearisation check ({len(lc)} SGD steps): "
                      f"{'MATCHES' if ok else 'MISMATCH'}  "
                      f"max rel err {np.max(np.abs(pred - meas) / (np.abs(meas) + 1e-12)):.2e}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "step2_predicate.json")
    with open(path, "w") as f:
        json.dump(records, f, indent=1, default=float)
    print(f"\nwrote {path} ({len(records)} records)")
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", nargs="*", default=list(STAGES))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--n-rollouts", type=int, default=20)
    args = ap.parse_args()
    main(stages=tuple(args.stages), seeds=tuple(args.seeds),
         n_rollouts=args.n_rollouts)
