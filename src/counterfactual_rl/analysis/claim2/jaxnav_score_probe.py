"""What does the CCE score actually contain on JaxNav? (issue #7)

Loads a trained checkpoint, collects states the way the replay buffer would, runs
the agent's OWN compiled counterfactual rollout, and reports what the (B, A, N)
return array holds. No training, no sweep -- minutes on one T4.

The question it answers is not "does CCE beat PER" but the one upstream of it:
is there a signal to prioritise by at all. A score that is exactly 0 for most of
the buffer cannot concentrate replay no matter what `beta` is, because
``(0 + eps) ** beta`` is the same constant for every transition.

Reads the agent's rollout via ``JaxNavConsequenceDQN._build_rollout_fn`` rather
than reimplementing it, so this measures the code that actually runs.

    PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src \
      python -m counterfactual_rl.analysis.claim2.jaxnav_score_probe \
        --cells 8x8_f01 8x8_f05 11x11_f03 [--bootstrap] [--n-states 64]

`--bootstrap` overrides `cf_bootstrap` for the scoring rollout only; the loaded
policy is untouched, so the same checkpoint can be scored both ways.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp

from counterfactual_rl.agents.jax_nav.consequence_dqn import JaxNavConsequenceDQN

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
MANIFEST = os.path.join(ROOT, "docs/figures/real/claim2/jaxnav/data/manifest_factorial.json")
RUNS = os.path.join(ROOT, "src/counterfactual_rl/agents/jax_nav/runs")

# The arm whose checkpoints these cells are probed through. cce_wmean is the
# shipped aggregation, so it is the one whose score health actually matters.
ARM = "cce_wmean"


def collect_states(agent, n, key, eps):
    """n on-policy states, one per lane, sampled from a random step of a rollout.

    Mirrors what the buffer holds: states from every stage of an episode under
    the checkpoint's own policy AND its own epsilon, not just resets and not a
    near-greedy walk. Auto-resets so a lane keeps producing after a terminal.
    """
    env, net, T = agent.env, agent.network, 200

    def lane(params, k):
        k, rk = jax.random.split(k)
        obs, st = env.reset(rk)

        def step(carry, _):
            obs, st, k = carry
            q = net.apply(params, obs)
            k, ak, bk, sk = jax.random.split(k, 4)
            a = jnp.where(jax.random.uniform(bk) < eps,
                          jax.random.randint(ak, (), 0, agent.n_actions), jnp.argmax(q))
            nobs, nst, r, done, _ = env.step(sk, st, a)
            k, rk2 = jax.random.split(k)
            robs, rst = env.reset(rk2)
            nobs = jnp.where(done, robs, nobs)
            nst = jax.tree.map(lambda x, y: jnp.where(done, x, y), rst, nst)
            return (nobs, nst, k), st

        _, states = jax.lax.scan(step, (obs, st, k), xs=None, length=T)
        return states

    allst = jax.jit(jax.vmap(lane, in_axes=(None, 0)))(agent.params, jax.random.split(key, n))
    pick = np.random.RandomState(0).randint(0, T, size=n)
    return jax.tree.map(lambda x: x[jnp.arange(n), pick], allst)


def _tv(p, q, edges):
    hp = np.histogram(p, bins=edges)[0] / len(p)
    hq = np.histogram(q, bins=edges)[0] / len(q)
    return 0.5 * np.abs(hp - hq).sum()


def per_state_scores(R):
    """Mean TV of every alternative against the best-looking action, per state."""
    B, A, _ = R.shape
    out = []
    for b in range(B):
        lo, hi = R[b].min(), R[b].max()
        if hi <= lo:                       # every action identical -> no signal
            out.append(0.0)
            continue
        e = np.linspace(lo, hi, 41)
        a0 = int(np.argmax([R[b, a].mean() for a in range(A)]))
        out.append(np.mean([_tv(R[b, a0], R[b, a2], e) for a2 in range(A) if a2 != a0]))
    return np.array(out)


def ess_frac(scores, exponent, eps=0.01):
    """Effective sample size the buffer would get if these were the priorities.

    ess_frac ~ 1.0 means the sampler is drawing uniformly -- the score is not
    differentiating the buffer, whether because it is flat at zero OR because it
    is uniformly high. Magnitude is not the same as spread.
    """
    p = (scores + eps) ** exponent
    return float(p.sum() ** 2 / (p ** 2).sum() / len(p))


def probe(cell, seed, bootstrap, n_states, manifest):
    rec = manifest[f"{cell}/{ARM}/{seed}"]
    blob = pickle.load(open(os.path.join(RUNS, str(rec["run_dir"]), "last.pkl"), "rb"))
    cfg = dict(blob["config"])
    cfg["cf_bootstrap"] = bootstrap          # scoring only; policy is unchanged

    agent = JaxNavConsequenceDQN(cfg)
    agent.params = blob["params"]
    agent.target_params = blob.get("target_params", blob["params"])
    agent._build_rollout_fn()

    t0 = time.time()
    sb = collect_states(agent, n_states, jax.random.PRNGKey(1),
                        float(blob.get("epsilon", 0.05)))
    A, N, H = agent.n_actions, agent.cf_n_rollouts, agent.cf_horizon
    keys = jax.random.split(jax.random.PRNGKey(2), n_states * A * N).reshape(n_states, A, N, 2)
    R = np.array(agent._compiled_rollout_fn(
        agent.params, agent.target_params, sb, agent._all_actions, keys))

    ps = per_state_scores(R)
    beta = cfg.get("PER_parameters", {}).get("beta", 0.25)
    print(f"\n{'='*78}\n{cell}  run_dir={rec['run_dir']}  map={cfg.get('map_size')} "
          f"fill={cfg.get('fill')}  H={H} N={N} A={A}  "
          f"cf_bootstrap={bootstrap}   ({time.time()-t0:.0f}s)\n{'='*78}")
    print(f"  rollouts returning EXACTLY 0 : {np.mean(R == 0)*100:5.1f}%")
    print(f"  states scoring EXACTLY 0     : {np.mean(ps == 0)*100:5.1f}%  "
          f"({int((ps == 0).sum())}/{len(ps)})")
    print(f"  per-state score  median {np.median(ps):.4f}  mean {ps.mean():.4f}  "
          f"max {ps.max():.4f}")
    print(f"  ess_frac @ beta={beta:<5}: {ess_frac(ps, beta):.4f}    "
          f"@ 1.43: {ess_frac(ps, 1.43):.4f}")
    return dict(cell=cell, bootstrap=bootstrap, run_dir=rec["run_dir"],
                zero_rollouts=float(np.mean(R == 0)),
                zero_states=float(np.mean(ps == 0)),
                median=float(np.median(ps)), mean=float(ps.mean()),
                ess_beta=ess_frac(ps, beta), ess_143=ess_frac(ps, 1.43))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["8x8_f01", "8x8_f05", "11x11_f03"])
    ap.add_argument("--seed", type=int, default=1, help="which seed's checkpoint")
    ap.add_argument("--n-states", type=int, default=64)
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--both", action="store_true",
                    help="score each cell with cf_bootstrap off AND on, side by side")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    a = ap.parse_args()

    manifest = json.load(open(MANIFEST))
    modes = [False, True] if a.both else [a.bootstrap]
    rows = [probe(c, a.seed, m, a.n_states, manifest) for c in a.cells for m in modes]

    if a.both:
        print(f"\n{'='*78}\nSUMMARY  (off -> on)\n{'='*78}")
        print(f"{'cell':12s} {'zero rollouts':>22s} {'zero states':>20s} {'ess@1.43':>18s}")
        for c in a.cells:
            o = next(r for r in rows if r["cell"] == c and not r["bootstrap"])
            n = next(r for r in rows if r["cell"] == c and r["bootstrap"])
            print(f"{c:12s} {o['zero_rollouts']*100:9.1f}% ->{n['zero_rollouts']*100:8.1f}% "
                  f"{o['zero_states']*100:9.1f}% ->{n['zero_states']*100:7.1f}% "
                  f"{o['ess_143']:8.4f} ->{n['ess_143']:8.4f}")

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
