# CCE on JaxNav — Robotics Transfer

## Goal
Show CCE (our counterfactual replay-priority signal) carries from the FrozenLake grid
world to a **robotics-flavored navigation** task: a wheeled robot with LiDAR choosing a
path to a goal. Primary target is **Claim 2** (CCE speeds learning). **Claim 1** (CCE
matches ground-truth importance) is a stretch — feasible only via a *surrogate* oracle,
not the FrozenLake tabular oracle.

## Environment: JaxNav (from JaxMARL)
Differential-drive robot, 200-beam LiDAR, continuous 2D world, goal-reaching. JAX-native,
the **same JaxMARL stack we already run for SMAX**. Verified by a live CPU probe:

- **Discrete action mode** — `JaxNav(act_type="Discrete")` → `Discrete(15)`, i.e. 15 fixed
  `(v, ω)` motion primitives (v∈{0,.5,1} × ω∈{±.5,±.25,0}). ✓ CCE needs discrete actions.
- **Functional / settable state** — `State` is a `flax.struct.dataclass` pytree.
  `env.step_env(key, state, actions)` steps from ANY stored state with **no auto-reset** —
  exactly the counterfactual primitive CCE needs (branch each action from a saved state).
  ✓ Use `step_env`; **never** the base-class `step()` (it auto-resets on done and would
  corrupt a counterfactual branch).
- **Sparse reward via constructor kwargs, no code edits** — `weight_g=0, dt_rew=0,
  lidar_rew=0, weight_w=0`, keep `goal_rew=4.0` (optionally keep `coll_rew=-4.0` for
  goal+collision-only).
- **Observation** = 205-dim vector: 200 LiDAR beams + v + ω + goal distance + goal bearing
  + rew_lambda (constant, single-agent-dead — can ignore).
- **No `env.P`** — pose is continuous, so there is no tabular transition model. This is the
  Claim-1 blocker. A Dijkstra shortest-path helper (`map_obj.dikstra_path`) exists and is a
  candidate *approximate* oracle.

## Feasibility verdict

| | Claim 2 (speeds learning) | Claim 1 (finds the moments that matter) |
|---|---|---|
| Needs from env | training logs + an eval success metric | a ground-truth "stakes" oracle over states |
| FrozenLake used | env-agnostic learning curves | exact Q* via tabular `env.P` value iteration |
| JaxNav | **Feasible** — wire logs + threshold + goal-reach metric | **Hard** — no tabular P; needs a *surrogate* oracle (chess-style value net, Monte-Carlo return-to-go, or Dijkstra distance) over *sampled* states |
| Verdict | **Commit** | **Stretch, gated on Claim 2 landing** |

Precedent that Claim 1 is not impossible: the repo already runs a Claim-1 pipeline for
**Gardner chess** — a non-enumerable env with no `env.P` — using a surrogate value-head
oracle over sampled states. That is the template if we attempt Claim 1 on JaxNav.

## The one non-obvious risk: where does CCE's signal come from?
CCE scores a transition by the **spread between per-action return distributions**. In
FrozenLake that spread is produced by **slip** (stochastic ice). But **single-agent JaxNav
dynamics are deterministic** given `(state, action)`. With a deterministic (greedy) rollout,
the N rollouts per action collapse to a single value → TV degrades to a coarse binary
"different / not," losing the graded priority that makes CCE useful.

**Decision:** make the CCE **rollout policy stochastic** (softmax temperature / ε) so the N
rollouts per action spread out and TV is graded again. FrozenLake had a stochastic *env* +
deterministic *policy*; JaxNav flips it to a deterministic *env* + stochastic *policy* —
symmetric, minimal, principled. Deterministic-coarse stays available as an ablation (our
deterministic FrozenLake still won Claim 2 on the coarse signal, so this is a quality knob,
not a hard blocker). Stage 2's suitability probe measures whether the signal is real before
we spend compute.

## Mirror the FrozenLake layout (explicit user requirement)
New code parallels `agents/frozen_lake/` and `envs/frozen_lake.py` one-to-one:

- `envs/jax_nav.py` — thin **single-agent adapter** exposing the FrozenLake-style contract
  CCE depends on: `reset(key)`, `step(key, state, action) → (obs, next_state, reward, done,
  info)` (wraps `step_env`, unwraps the `agent_0` dict to scalars), `n_actions = 15`.
  Sparse-reward kwargs baked in. jit/vmap-safe.
- `agents/jax_nav/config.py` — mirror FL config + CCE knobs (`cf_horizon`, `cf_n_rollouts`,
  `cf_gamma`, `mu`, `priority_mixing`, …) + new `cf_rollout_temperature`, map/scenario,
  sparse-reward flags.
- `agents/jax_nav/{dqn,dqn_vectorized,consequence_dqn,consequence_dqn_vectorized}.py` —
  mirror FL trainers. Two real changes: (1) Q-net is an **MLP over the 205-dim obs** (not an
  int→one-hot embedding); (2) the rollout uses `actions = arange(15)`, a **stochastic**
  rollout policy, and stores/stacks **pytree** states.
- `agents/shared/consequence_buffers.py` — generalize the one FrozenLake-specific spot
  (`add_batch` casts `int(jax_state)`) to store pytree states via `jax.tree.map`. The single
  `add` already stores states as-is.
- `agents/jax_nav/{train,run_experiments}.py` — mirror.

## Staged plan (stop-and-check gates)
0. **Env adapter + smoke.** Build `envs/jax_nav.py`; verify reset/step/counterfactual-branch,
   jit + vmap safe. (small)
1. **Vanilla DQN + PER baseline.** MLP Q-net; confirm plain DQN learns (goal-reach rate
   climbs) on a simple scenario. *Gate: does DQN learn at all here?*
2. **Suitability probe (oracle-free go/no-go).** Add `make_jaxnav_adapter` (`qstar_spread=
   None`) mirroring the planned Connect-Four adapter; run the scorecard. *Gate: real forks +
   signal distinct from |TD|? If no-go, retune env before spending compute.*
3. **CCE + Claim 2 (headline).** Mirror `consequence_dqn_vectorized`; run CCE(additive/
   multiplicative) vs PER vs uniform sweep; wire Claim-2 analysis (parse_logs branch,
   threshold, goal-reach metric). Deliverable: learning-curve win (or an honest null).
4. **Claim 1 (stretch, gated on 3).** Surrogate oracle over sampled states (Dijkstra distance
   or a trained critic); correlate with CCE (Spearman + precision@k), mirroring the chess
   Claim-1 pipeline.

## How to run (worktree gotchas)
- Conda env `counterfactual` (has JAX/JaxMARL), **not** `bucks`.
- From this worktree, prepend the worktree src so jobs load THIS branch's code, and disable
  user-site to dodge a broken `~/.local` cffi:
  `PYTHONNOUSERSITE=1 PYTHONPATH=<worktree>/src python -m counterfactual_rl.agents.jax_nav.train …`
- Always `--dry-run` sweeps first.

## Open questions for approval
- **Map:** start on a fixed singleton (`SingleNav1`, reproducible) or random `Grid-Rand-Poly`
  (harder, better generalization story)? Recommend fixed/simple first, scale later.
- **Scope:** commit Stages 0–3 (Claim 2); treat Stage 4 (Claim 1) as stretch. Confirm.
- **Stochastic-rollout knob:** OK to make the CCE rollout policy stochastic by default?
