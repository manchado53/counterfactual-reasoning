# Counterfactual Reasoning Research

## What this is
Research on **CCE (Counterfactual Consequence Estimation)** — a replay-priority signal that
scores a transition by how much the *action choice* changed the outcome (total variation among
per-action return distributions from policy rollouts), mixed with TD-error priority (additive
Eq4 / multiplicative Eq5). Aim: publish (ICLR) that CCE **(C1)** finds the moments that matter
and **(C2)** speeds learning, across environments — FrozenLake (single-agent), SMAX
(multi-agent), Connect Four. Goal/status → `lab-notebook.md`.

## Lab Notebook (READ FIRST)
Start every session by reading `lab-notebook.md` (project root): current STATUS, NEXT, dead
ends, per-env pre-flight checklist. Before the user clears context, append a dated LOG entry
(especially DEAD ENDS). LOG is append-only; only rewrite STATUS and NEXT.

## Code Layout (`src/counterfactual_rl/`)
- `envs/` — JAX envs (frozen_lake, smax, chess). Connect Four uses `pgx` directly.
- `agents/<env>/` — trainers: `config.py`, `dqn.py`/`consequence_dqn.py` (+ `*_vectorized.py`),
  `train.py`, `run_experiments.py` (SLURM sweeps → manifests under `experiments/<month>/`).
- `agents/shared/` — buffers, consequence scoring, metrics logger, slurm_throttle, timing.
  Run outputs: `agents/<env>/runs/<job_id>/` (legacy `agents/shared/runs/`).
- `analysis/claim1/` & `analysis/claim2/` — figure pipelines (oracle, scoring, rliable, plots).
- `paper/` — `paper.tex` + `repro/` (rebuilds figures from cache). `docs/_archive/` = don't cite.

## Environments
- **FrozenLake** (`envs/frozen_lake.py`): OUR custom JAX reimpl of Gymnasium FrozenLake-v1
  (not an import). Exposes `env.P` (Gymnasium-style transition dict) — the Claim-1 oracle uses
  it. Maps `4x4`/`8x8` built in (others via `desc=`). `is_slippery` defaults True (slippery =
  3 equiprob outcomes `[(a-1)%4,a,(a+1)%4]`); reward 1.0 only on first landing G. jit/vmap-safe.
- **SMAX / JaxMARL** (multi-agent): `make('HeuristicEnemySMAX', ...)`; `won_battle_bonus=10` is
  hard-coded; per-scenario arch presets in `smax/config.py`. Fetch docs before relying on API:
  https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/smax
- **pgx** (Connect Four — active 2nd env): JAX board games; opponent = random/rule_based/mcts.
  Fetch https://www.sotets.uk/pgx/api/ , /api_usage/ . Gardner chess (also pgx) tried & **dropped**.

## pgx Gotchas (CRITICAL — Connect Four)
- pgx **randomizes the first player** (`env.init(key)` → current_player 0/1, ~50/50).
- Always `rewards[state.current_player]`, NEVER `rewards[0]` — else reward sign flips in ~50% of
  envs and nothing learns. (Cost all C4 runs 05-07→05-13. Fix: `agent_player =
  state.current_player` at episode start, `rewards[agent_player]` throughout.)
- Observations already from current player's perspective — no manual flip.
- Exception: the chess wrapper seat-normalizes so the DQN always sees white (`current_player==0`)
  and reads `rewards[0]` — opposite convention from Connect Four.

## Training gotchas (these have bitten us)
- Vectorized trainers fire eval / save / target-sync via CROSSING logic `total//f > prev//f`,
  NEVER `% f == 0` (the counter jumps past boundaries). Target freq is set in env steps but
  applied in gradient steps: `target_freq_q = max(1, C // n_steps_for_Q_update)`.
- Sparse boolean obs (Connect Four): use **LeakyReLU** and `use_layer_norm=False` — plain ReLU /
  LayerNorm zero the conv gradient and Q diverges.
- Config keys differ: FL uses `alpha/n_episodes/map_name`, trains by episodes; board games use
  `C/M/n_chunks/exploration_fraction`, train by chunks. Don't mix them.

## CCE config knobs (same names across envs)
`algorithm` (dqn-uniform / dqn=PER / consequence-dqn) · `consequence_metric` (default
total_variation; KL is unbounded, can be inf) · `priority_mixing` additive(Eq4)/multiplicative(Eq5)
· `mu` (0=TD … 1=consequence; `mu_c`/`mu_delta` for multiplicative) · `consequence_aggregation`
(weighted_mean) · `score_interval` (per Q-update — tied to buffer turnover, easy to misset) ·
`n_score_sample` · `cf_horizon`/`cf_n_rollouts`/`cf_top_k`/`cf_gamma` (separate from training
`gamma`) · `diagnostics_enabled` (expensive; default False).

## How to run
- Config override (universal): base64-JSON in env var `CONFIG_OVERRIDES_B64`; merge order
  DEFAULT < SCENARIO_PRESET < CONFIG_OVERRIDES_B64.
- One job: `python -m counterfactual_rl.agents.<env>.train [--algorithm/--mu/--seed/--override K=V]`
- Sweep: `python -m counterfactual_rl.agents.<env>.run_experiments <name> [--dry-run] [--max-concurrent N]`
  (names in each `run_experiments.py`; SMAX in `smax/experiments.py`). **Always --dry-run first.**
- Analysis: `…analysis.claim1.frozen_lake.run_analysis` · `…analysis.claim2.run_analysis --manifest <p> --env <n>`

## Cluster (Rosie / SLURM)
- Check jobs: `squeue -u $USER`. `agents/shared/slurm_throttle.py` caps concurrency (`--max-concurrent`).
- After every `sbatch`, immediately start a persistent Monitor on the job's `.out`
  (grep: training progress, eval metrics, errors, completion) — don't wait to be asked.
