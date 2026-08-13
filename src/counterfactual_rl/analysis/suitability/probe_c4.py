"""C4 suitability PROBE — gate before the full grid.

ONE question: when the best mcts-trained player plays a foe, do the boards it actually faces have
real STAKES (forks where the move decides the game), or are they degenerate (C(board) ~= 0 because
the strong player wins no matter what)?

We answer with data, cheaply (random foe), BEFORE building the full pipeline. See
`plans/cce-suitability-c4-probe.md`.

Design (standalone — does NOT touch metrics.py / rollout_sweep.py / envs.py):
  1. Load the best checkpoint of a run; build a Connect4ConsequenceDQN with the chosen foe.
  2. Collect agent-to-move boards by self-play vs that foe (SEAT-NORMALIZED so current_player==0,
     foe logic == the rollout atom's foe). Dedup; record game phase.
  3. Score each board with the agent's own counterfactual rollout atom -> returns(B,7,N).
  4. Stakes per board, TWO ways (review fix): C_mean = nanmax-nanmin of per-column mean returns,
     and C_tv = total-variation consequence (what CCE actually uses). Illegal columns -> NaN/excluded.
  5. Report the C distributions + by game-phase tertile + example boards; save a histogram.

Run (full probe on a GPU):
  srun ... ~/.conda/envs/counterfactual/bin/python -m \
     counterfactual_rl.analysis.suitability.probe_c4 \
     --run-dir src/counterfactual_rl/agents/shared/runs/259285 \
     --foe random --n-boards 600 --n-rollouts 20 \
     --out docs/figures/suitability/c4/probe_259285_random.json
Tiny correctness test on CPU: add --n-boards 8 --n-rollouts 2 --max-games 4.
"""

import argparse
import json
import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import pgx

from counterfactual_rl.agents.connect_four.consequence_dqn import Connect4ConsequenceDQN, C4_ACTIONS
from counterfactual_rl.analysis.metrics import compute_consequence_metric

_ENV = pgx.make("connect_four")


# --------------------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------------------
def load_agent(run_dir, foe, ckpt_name="best.pkl"):
    """Construct an agent with the chosen FOE, then assign frozen params.

    load() does not overwrite config, but we construct-with-config then assign params directly so
    the opponent is set BEFORE the rollout fn is baked (the foe is closed over at build time)."""
    path = os.path.join(run_dir, ckpt_name)
    with open(path, "rb") as f:
        ck = pickle.load(f)
    cfg = dict(ck["config"])
    cfg["opponent"] = foe                      # the probe foe (random/rule_based/mcts)
    agent = Connect4ConsequenceDQN(ck["env_info"], config=cfg)
    agent.params = jax.tree.map(jnp.array, ck["params"])
    agent.target_params = jax.tree.map(jnp.array, ck["target_params"])
    agent._build_batched_rollout_fn()          # bakes the foe into _compiled_batched_fn
    return agent, cfg


# --------------------------------------------------------------------------------------
# foe + greedy (mirror consequence_dqn.py exactly so collection == scoring)
# --------------------------------------------------------------------------------------
def _greedy_action(agent, state):
    q = agent.network.apply(agent.params, state.observation.reshape(-1))   # (1,7)
    return jnp.argmax(jnp.where(state.legal_action_mask, q[0], -jnp.inf))


def _foe_random_action(state, key):
    # IDENTICAL to the atom's random foe (consequence_dqn.py:120-123): argmax of masked normal.
    logits = jax.random.normal(key, (C4_ACTIONS,))
    return jnp.argmax(jnp.where(state.legal_action_mask, logits, -jnp.inf))


def _foe_action(state, key, foe):
    if foe == "rule_based":
        from counterfactual_rl.agents.connect_four.opponent import rule_based_action
        return rule_based_action(state, key)
    if foe == "mcts":
        from counterfactual_rl.agents.connect_four.opponent_mcts import mcts_action
        return mcts_action(state, key)
    return _foe_random_action(state, key)


# jit the env step so the python self-play loop isn't dominated by dispatch
# (greedy/foe call network.apply eagerly — fine for a few hundred boards)
_step_jit = jax.jit(_ENV.step)


# --------------------------------------------------------------------------------------
# collect boards by self-play (seat-normalized: every recorded state has current_player==0)
# --------------------------------------------------------------------------------------
def collect_boards(agent, foe, n_boards, max_games, seed=0):
    key = jax.random.PRNGKey(seed)
    states, phases = [], []
    seen = set()
    games = 0
    while len(states) < n_boards and games < max_games:
        key, gk = jax.random.split(key)
        state = _ENV.init(gk)
        # seat-normalize: if the foe moves first, take one foe pre-move so it's our turn
        if int(state.current_player) != 0:
            key, fk = jax.random.split(key)
            state = _step_jit(state, _foe_action(state, fk, foe))
        ply = 0
        while not bool(state.terminated | state.truncated):
            if int(state.current_player) != 0:        # safety: should be our turn
                key, fk = jax.random.split(key)
                state = _step_jit(state, _foe_action(state, fk, foe))
                continue
            # OUR turn: record (deduped)
            board_key = np.array(state.observation).tobytes()
            if board_key not in seen:
                seen.add(board_key)
                states.append(state)
                phases.append(ply)
            a = _greedy_action(agent, state)
            state = _step_jit(state, a)
            if bool(state.terminated | state.truncated):
                break
            key, fk = jax.random.split(key)            # foe reply
            state = _step_jit(state, _foe_action(state, fk, foe))
            ply += 1
            if len(states) >= n_boards:
                break
        games += 1
    # assert seat-normalization held
    for s in states:
        assert int(s.current_player) == 0, "seat-normalization violated: current_player != 0"
    return states[:n_boards], np.array(phases[:n_boards]), games


# --------------------------------------------------------------------------------------
# score boards with the rollout atom -> returns(B,7,N); stakes two ways
# --------------------------------------------------------------------------------------
def score_boards(agent, states, n_rollouts, chunk=64, seed=1):
    N = n_rollouts
    actions_row = jnp.arange(C4_ACTIONS, dtype=jnp.int32)        # all 7 columns
    key = jax.random.PRNGKey(seed)
    B = len(states)
    all_returns = np.full((B, C4_ACTIONS, N), np.nan, dtype=np.float64)
    for lo in range(0, B, chunk):
        sub = states[lo:lo + chunk]
        b = len(sub)
        # pad to a fixed chunk size so the jit shape is stable (avoid recompiles); discard pad
        pad = chunk - b
        padded = sub + [sub[-1]] * pad if pad else sub
        batched = jax.tree.map(lambda *xs: jnp.stack(xs, 0), *padded)
        actions = jnp.broadcast_to(actions_row, (chunk, C4_ACTIONS))
        key, sk = jax.random.split(key)
        keys = jax.random.split(sk, chunk * C4_ACTIONS * N).reshape(chunk, C4_ACTIONS, N, 2)
        ret = np.array(agent._compiled_batched_fn(agent.params, batched, actions, keys))  # (chunk,7,N)
        all_returns[lo:lo + b] = ret[:b]
    return all_returns


def stakes(states, returns, metric, aggregation):
    """C_mean = nanmax-nanmin of per-column mean returns; C_tv = total-variation consequence."""
    B = len(states)
    C_mean = np.full(B, np.nan)
    C_tv = np.full(B, np.nan)
    examples = []
    for i, s in enumerate(states):
        legal = np.array(s.legal_action_mask).astype(bool)
        m = returns[i].mean(axis=1)                 # (7,) mean over N
        m_legal = np.where(legal, m, np.nan)
        C_mean[i] = np.nanmax(m_legal) - np.nanmin(m_legal)
        # C_tv: distributional consequence over LEGAL columns, ref = greedy column
        legal_cols = [c for c in range(C4_ACTIONS) if legal[c]]
        if len(legal_cols) >= 2:
            rd = {(c,): returns[i, c] for c in legal_cols}      # each (N,)
            greedy_c = int(legal_cols[int(np.nanargmax([m[c] for c in legal_cols]))])
            probs = {(c,): 1.0 for c in legal_cols}             # uniform alt-weights
            C_tv[i] = compute_consequence_metric((greedy_c,), rd, metric=metric,
                                                 action_probs=probs, aggregation=aggregation)
        if i < 3:
            examples.append({"phase": None, "legal": legal.astype(int).tolist(),
                             "col_mean_return": np.round(m, 3).tolist(),
                             "C_mean": round(float(C_mean[i]), 3),
                             "C_tv": None if np.isnan(C_tv[i]) else round(float(C_tv[i]), 3)})
    return C_mean, C_tv, examples


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def _eff_n(x):
    """Crude effective sample size via lag-1 autocorrelation (boards within a game are correlated)."""
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float(len(x))
    x0 = x - x.mean()
    r1 = float(np.sum(x0[:-1] * x0[1:]) / np.sum(x0 * x0)) if np.sum(x0 * x0) > 0 else 0.0
    r1 = max(-0.99, min(0.99, r1))
    return float(len(x) * (1 - r1) / (1 + r1))


def summarize(C, label):
    c = C[np.isfinite(C)]
    if len(c) == 0:
        return {"label": label, "n": 0}
    return {"label": label, "n": int(len(c)), "n_eff": round(_eff_n(C), 1),
            "mean": round(float(c.mean()), 4), "median": round(float(np.median(c)), 4),
            "p90": round(float(np.percentile(c, 90)), 4),
            "frac_gt_025": round(float((c > 0.25).mean()), 3),
            "frac_lt_005": round(float((c < 0.05).mean()), 3)}


def by_phase(C, phases):
    out = {}
    fin = np.isfinite(C)
    if fin.sum() == 0:
        return out
    p = phases[fin]; c = C[fin]
    if len(p) == 0:
        return out
    t1, t2 = np.percentile(p, [33, 66])
    for name, mask in [("early", p <= t1), ("mid", (p > t1) & (p <= t2)), ("late", p > t2)]:
        if mask.sum():
            out[name] = {"n": int(mask.sum()), "median_C_mean": round(float(np.median(c[mask])), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="best.pkl")
    ap.add_argument("--foe", default="random", choices=["random", "rule_based", "mcts"])
    ap.add_argument("--n-boards", type=int, default=600)
    ap.add_argument("--n-rollouts", type=int, default=20)
    ap.add_argument("--max-games", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()

    print(f"[probe] loading {args.run_dir}/{args.ckpt} with foe={args.foe}", flush=True)
    agent, cfg = load_agent(args.run_dir, args.foe, args.ckpt)
    metric = cfg.get("consequence_metric", "total_variation")
    aggregation = cfg.get("consequence_aggregation", "weighted_mean")

    print(f"[probe] collecting up to {args.n_boards} boards by self-play vs {args.foe} ...", flush=True)
    states, phases, games = collect_boards(agent, args.foe, args.n_boards, args.max_games, args.seed)
    print(f"[probe] collected {len(states)} unique boards from {games} games", flush=True)

    print(f"[probe] scoring (N={args.n_rollouts} rollouts/col, chunk={args.chunk}) ...", flush=True)
    returns = score_boards(agent, states, args.n_rollouts, args.chunk, args.seed + 1)
    C_mean, C_tv, examples = stakes(states, returns, metric, aggregation)
    for e, ph in zip(examples, phases[:3]):
        e["phase"] = int(ph)

    report = {
        "run_dir": os.path.abspath(args.run_dir), "ckpt": args.ckpt, "foe": args.foe,
        "metric": metric, "n_boards": len(states), "n_rollouts": args.n_rollouts, "games": int(games),
        "C_mean": summarize(C_mean, "C_mean (action gap of means)"),
        "C_tv": summarize(C_tv, f"C_tv ({metric}, what CCE uses)"),
        "C_mean_by_phase": by_phase(C_mean, phases),
        "examples": examples,
    }
    # the gate
    cm, ct = report["C_mean"], report["C_tv"]
    healthy = (cm.get("median", 0) >= 0.2 and cm.get("frac_gt_025", 0) >= 0.3
               and ct.get("median", 0) >= 0.3)
    report["gate"] = {"healthy": bool(healthy),
                      "rule": "median C_mean>=0.2 AND >=30% boards C_mean>0.25 AND median C_tv>=0.3"}

    print(json.dumps(report, indent=2), flush=True)
    print(f"\n[GATE] {'HEALTHY -> proceed to full grid' if healthy else 'DEGENERATE -> switch design (filter/matched)'}", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[probe] wrote {args.out}", flush=True)

    fig = args.fig or (args.out and args.out.replace(".json", "_Cdist.png"))
    if fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a, C, lab in [(ax[0], C_mean, "C_mean (action gap)"), (ax[1], C_tv, f"C_tv ({metric})")]:
            c = C[np.isfinite(C)]
            a.hist(c, bins=30, color="#7c5cff", edgecolor="#222")
            a.axvline(np.median(c), color="#e3b341", ls="--", label=f"median={np.median(c):.2f}")
            a.set_title(f"{lab}  vs {args.foe}"); a.set_xlabel("stakes C(board)"); a.legend()
        f.suptitle(f"C4 probe — best player vs {args.foe} ({len(states)} boards)")
        f.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(fig)), exist_ok=True)
        f.savefig(fig, dpi=110)
        print(f"[probe] wrote {fig}", flush=True)


if __name__ == "__main__":
    main()
