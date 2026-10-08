# Legacy learner migration

Work date: 2026-10-08. Branch: `feature/jaxnav-parity-migration`.

## What moved

| ID | Code | Clock | Source |
|---|---|---|---|
| smax | Historical centralized DQN, PER and CCE | episode | `b5eb298`, clean `src` snapshot |
| chess | Historical vectorized Gardner DQN, array replay and CCE | chunk | same snapshot |
| cvrp | Fixed-instance tabular CVRP/TSP DQN, PER and CCE | episode | same snapshot |
| rpdqn | Neural fresh-instance routing, attention DQN, PER and CCE | episode | `51477d0` **plus dirty files** |
| cvrp_budget | Budget routing; separate old-budget and Option A window/stranding variants | episode | `14cb90f` and `c2bdb00`, pinned separately |

Each has lazy `validate_config`, `build_agent`, `smoke_overrides`,
`simulator_sources` and `evaluation_axis` hooks. All five use JAX.
`cvrp_budget` is registered separately from plain CVRP.
The RP-DQN snapshot includes dirty `dqn.py`, `consequence_dqn.py` and
`consequence_dqn_vectorized.py`; the commit alone does not recover this variant.

Old buffers, score metrics, timers and rollout helpers are copied under each
learner's `_historical` folder. They do not use the modern ranked replay core.
Source files stay in place. Port changes are imports and output paths only.
SMAX's build hook binds its old free `evaluate` function as a method.
`CCE_RUN_DIR` routes new logs, plots and checkpoints to the shared run folder.
RP-DQN configs must state capacity and max steps: its old constructor resolves
these values and writes them into `agent.config`.

## Recipes and study status

`*-historical.json` files are **generic recovered learner examples**, with three
arms and seed 0. They are not full historical-study replays.

`cvrp-logged-settings.json` has 61 complete logged configs.
`rpdqn-logged-settings.json` has 32 complete logged configs.
`cvrp_budget-logged-settings.json` has 2,624 complete logged configs:
2,128 old-budget rows and 496 Option A rows. No generic budget template replaces
these saved settings. Each row names its pinned port source and keeps all flags.
Every row links its source run and exact saved header hash. No missing scientific
setting was filled and no unknown flag was dropped. These configs run on the
frozen ports; the source revision used by each old job is still unbound.

`logged-recipe-coverage.json` records every rejected header and its exact gap.
Old SMAX logs lack settings such as update frequency and checkpoint count.
Old chess logs use episode budgets and lack the later chunk settings.
Budget, time-window and stranding configs go to `cvrp_budget`, not plain CVRP.
Unsupported or incomplete headers stay in the coverage report. The exact source
revision used by each historical job remains unverified, including budget jobs.

## Recovery files

- `source-manifest.json`: clean source hashes and import-move map for the first three ports.
- `oracle-source.json`: tabular CVRP oracle hash.
- `rpdqn-source-manifest.json`: dirty source status, hashes and copy map.
- `budget-budget-source-manifest.json`, `budget-option_a-source-manifest.json`:
  separate pinned source hashes and recoverable original bytes under `budget-sources/`.
- `sources/`, `rpdqn-sources/`: retained original source, usable without sibling worktrees.
- `history-manifest.json`: scan of 46 known local/remote refs; 54 selected historical text files.
- `study-catalog.json`: 99 manifest locations, 56 distinct manifests, 23 declared study plans.
- `manifests/*.bytes.json`: base64 envelopes recover the exact original bytes, including missing final newlines.
- `run-headers.json`: 2,968 exact log headers (3,524,725 source bytes). Counts: 232 SMAX, 6 chess, 2,693 tabular/budget CVRP, 37 neural CVRP.

The catalog retains all discovered manifest override rows, seeds and arms.
Declared studies are not proof of submission. A manifest is not proof of success.
Headers are not full logs. Checkpoints and full logs remain at the recorded paths.
No remote refs were fetched. Deleted/remote-only files and run roots outside the
recorded scan are not covered. History text views with a normalized final newline
link an `exact_bytes` envelope when needed.

## Bounded checks

Tests: `tests/test_legacy_environment_ports.py`.
Saved test reports: `checks/*.xml`.
Exact comparisons use array equality, not a loose statistical threshold.
Final core run: 19 passed. Nine simulator/learner cases passed in separate bounded
processes (six SMAX/chess cases and three RP-DQN cases). Together: 28 checks.
The shared recipe loader validated all 105 rows across the four ports.
That count predates budget registration; no new shared-loader run is claimed here.
The separate budget run passed 16 checks in 32.09 seconds; results are already
saved in `checks/budget-all.xml`. Both variants and all three replay arms matched
old/new weights and evaluation. All eight window/stranding/reward combinations
matched transition and oracle outputs on the small instance.
The old-budget saved-task CPU smoke completed through `smoke_budget.py`, before
registry integration. Its existing record is retained as `checks/budget-smoke-attempt.json`.
Outputs remain at `/tmp/cvrp-budget-budget-642wu3n2`.
**Option A saved-task smoke is unrun.** Its reduced old/new checks did pass.
The shared runner completed the CVRP logged-settings task-1 CPU smoke; its run
record is saved as `checks/cvrp-runner-attempt.json`. Logs and generated timing
plots are at `/tmp/legacy-cvrp-smoke.pvGavr/run`.
The shared runner also completed the RP-DQN logged-settings task-1 CPU smoke;
`checks/rpdqn-runner-attempt.json` saves its record. Outputs and plots are at
`/tmp/legacy-rpdqn-smoke.8wJ80M/run`.

| Check | Scope | Result |
|---|---|---|
| CVRP transition tables and step outputs | every small-instance state/action; TSP/CVRP; noise 0 and .2; fixed JAX keys | exact |
| CVRP oracle arrays | same small instances | exact old/new; not a new oracle validity claim |
| CVRP learning | 2 episodes, small instance; uniform/PER/CCE | exact weights and evaluation |
| SMAX learning | 1 episode, 3m; uniform/PER/CCE | exact weights and evaluation |
| Chess learning | 1 chunk of 2 transitions; uniform/PER/CCE; random opponent | exact weights and evaluation |
| RP-DQN learning | 2 episodes, 3 customers; uniform/PER/CCE; frozen dirty source vs new port | exact weights, episode returns, step count and evaluation |

These checks do not reproduce the full studies, all seeds, or upstream RP-DQN.
They do exercise scoring and gradient updates with small batches.
Smoke hooks keep task settings; the old/new tests explicitly state their separate
small task changes. The historical chess clock is `chunk`, not env steps.

Run each simulator case in its own bounded process, e.g.:

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=.:src JAX_PLATFORMS=cpu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 180 taskset -c 0-3 \
  /tmp/cce-clean-install-uudpga/venv/bin/python -m pytest \
  'tests/test_legacy_environment_ports.py::test_simulator_reduced_learning_exact[consequence-dqn-smax]' -q
```

For chess use `/home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python`.
The clean CPU venv has no `pgx`; the conda env has `pgx==2.6.0`.
Both checked runtimes had JAX/JAXlib 0.9.1, NumPy 2.4.0, SciPy 1.16.3,
Flax 0.10.4, Optax 0.2.5, JaxMARL 0.1.0 and pytest 9.1.1.
No dependencies were installed or upgraded.

## Kept defects and gaps

- Historical `weighted_mean` without action probabilities falls back to max.
  The older RP-DQN metric has no later warning. Both versions stay distinct.
- CVRP multiplies cost by `max(0, 1 + noise*z)`. Clipping changes its expectation;
  the old claim that this leaves expected rewards exactly unchanged is false.
  The oracle still uses the old noiseless table. No behavior was fixed here.
- Chess wrapper comments assert player 0, but PGX may randomize the player.
  The vectorized collector reads the actual player. No wrapper rewrite is hidden.
- RP-DQN's logged `win_rate` is a reference-length score, not an exact per-instance oracle.
- Full saved-study parity needs the source revision used by the old jobs, full logs,
  and result/figure regeneration. Reduced learner parity is narrower evidence.
- Older SMAC is **archive-only**. Conda has `smac` and `pysc2`, but no importable
  `naive_ppo`. The old good-policy entrypoint also uses bare `utils` and historical
  analyzer imports. StarCraft2 engine availability was not tested. The old base
  module is recoverable from `origin/refactorization`. No simulator test was invented.
- Routing budget/time-window/stranding variants are ported under `cvrp_budget`.
  Reduced parity and the old-budget saved-task smoke passed. Option A saved-task
  smoke remains unrun; historical job-to-source linkage remains unverified.

No branch, commit, push, job, source deletion or shared-runner edit was made by
this workstream. Other workers' changes were left in place.
