# Old ConnectFour and larger DoorKey — scope and source rules

This extends the first migration slice. The generic runner, registry and other
workers' files remain outside this worker's edits. No job was submitted.

## Old ConnectFour

`protocol=legacy-connect-four` selects the original convolutional DQN and CCE,
the original chess-named array replay buffer, and the original opponents. The
fixed JaxNav transfer remains `protocol=fixed-jaxnav-transfer`. These protocols
do not share learner or replay equations.

The old learner, network, foe, scoring, and buffer files are exact source copies
apart from import paths. A test compares their complete text after only those
path substitutions. `legacy/shared/metrics.py` also redirects output through
`CCE_RUN_DIR`. `legacy/runner.py` adds eval JSON and initial RNG records around
the unchanged source `learn`/`evaluate` methods.

Five launch-definition recipes were resolved from the captured source:
`smoke`, `claim2_main`, `claim2_rulebased`, `claim2_mcts`, `claim2_mcts_v2`.
These hold 205 rows. They use the captured defaults plus the exact source
definition overrides. They are **not** claims that the full configs used by
older submitted jobs were recovered. In particular, the source's MCTS-v1
comment says horizon 30 / 16 rollouts, but its current defaults are 42 / 30.
Both current definitions therefore resolve to 42 / 30. We do not turn an old
comment or an override-only manifest into a verified past full config.

Old rules retained:

* Budget and epsilon decay use chunks. Original metrics still label the chunk
  column `episode`. The new eval JSON records both `source_chunk` and exact
  `env_steps`; generic-runner evaluation axis is `env_steps`.
* Train, rollout and evaluation all use `opponent`: `random`, `rule_based`, or
  `mcts`. The `eval_opponent='random'` config field is inert in the original.
  Validation and evaluation identity say so; no fixed-transfer foe is substituted.
* MCTS keeps 32 simulations in the CPU checks. PGX role handling, reset rules,
  legal masks, discounted returns and terminal rules stay in the source code.
* Evaluation uses the zero-based chunk index as its seed during training.
  A direct `evaluate()` call defaults to seed 42. This is not a fixed shared bank.
* Source code seeds JAX but does not seed global NumPy replay draws. The port
  keeps that behavior and records the inherited state in
  `numpy_initial_state.json`. The config's seed alone does not fully reproduce
  this historical protocol. Tests supply the same NumPy state to old and new.
* CCE refreshes sampled old replay entries under its source cadence; it does
  not become fixed-transfer rank CCE or score-on-insert CCE.

The extra array-buffer and action-selection dependencies were captured before
porting. `extra-source-manifest.json` holds their hashes, source head and dirty
status. Their exact bytes live under `sources/root/`. Runtime imports never
refer to sibling or private worktrees.

## Larger DoorKey

All six source layouts now validate: 6x6, 6x6_lava, 12x12, 16x16,
12x12_lava, and 16x16_lava. The old source supports sequential and vectorized
learners; both remain explicit through `collector_variant` and `vectorized`.

`doorkey-headroom-legacy.json` contains the exact 12 source study rows:
12x12/16x16 × uniform/PER × seeds 0,1,2. These keep exploring training starts,
fixed evaluation starts, the 120-step collection cap, and the 30,000-episode
budget. CPU smoke shrinks compute but keeps those start and terminal rules.
Collection length must still cover the old collector's timeout because its
episode-length counter resets each chunk.

`doorkey-large-lava-cli-legacy.json` has ten explicit source-supported CLI
instances: two large lava layouts × five arms, seed 0. It uses the source
sequential default and source CLI layout/algorithm/mixing overrides. It is
labeled as supported task instances, **not** a recovered submitted study.
The 50-step cap and fixed starts are source defaults in these rows; they are
not copied from the separate headroom study. No past result is claimed.

## Seed scope

Fixed-transfer validation intentionally accepts only seeds 0 and 1, the frozen
source transfer cohort. This is a protocol pin, not a limit on the algorithms
or this broader migration. Old ConnectFour source studies keep seeds 0–9;
DoorKey headroom keeps seeds 0–2; legacy build hooks accept valid uint32 seeds.
Additional fixed-transfer research needs a separately named, explicit recipe
protocol; it should not silently alter the pinned source comparison.

## Runtime and checks

No checkpoint download is required for these old ConnectFour foes. The
historical conda runtime has PGX 2.6.0 and mctx 0.0.6. The clean CPU venv still
lacks PGX. `simulator_sources` includes the installed PGX and, for MCTS, mctx
Python sources in the generic runner's bundle. No dependency was upgraded.

Each source comparison is a separate CPU subprocess, limited to four cores and
180 seconds. ConnectFour tests compare actual reduced `learn` calls: parameters,
target parameters, optimizer state, random key, replay TD magnitudes, CCE scores,
completed episodes and seeded evaluation. Each has two 16-transition chunks and
eight real updates; all three foes retain their source logic. Large DoorKey
tests likewise compare actual old/new learning, state tables, parameters,
targets, RNG, returns, lengths and evaluation.

The small-run outputs and exact test results are in `extension-evidence.json`.
These checks establish source-port behavior on seed 0 and reduced compute,
not learning success, full-buffer cost, cross-hardware performance, or parity
with an old run whose complete configuration/source snapshot is unavailable.
