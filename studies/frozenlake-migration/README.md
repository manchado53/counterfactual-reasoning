# FrozenLake move

Code moved. Small checks pass. Full past-run parity is not claimed.

```text
sealed Oct 1 source -> fixed learner -> step-budget run
old repo source    -> old learner   -> episode-budget run
```

See `source-map.json` for full source and new-file SHA-256 hashes.
See `parity-results.json` for exact old/old and old/new checks.
The old source for those checks is the sealed **fixed transfer** source.
It is not the older, faulty chunk-reset collector.

| Path | Meaning |
|---|---|
| `configs/experiment/frozenlake-stochastic-transfer.json` | All 25 sealed rows; five arms, five seeds; fixed step budget |
| `configs/experiment/frozenlake-deterministic-old.json` | Five old arms at seed 0; source-defined subset of `CLAIM2_NO_SLIP` |
| `configs/experiment/frozenlake-smoke.json` | Three tiny fixed runs; no research claim |

The fixed Q rule stays one-hot state -> ReLU MLP -> four Q values.
The target uses max Q from the target net. Loss is weighted squared TD error.
Adam, gradient clipping, target sync, epsilon clock, replay weights, score draw
order, and per-block score reuse stay as in the sealed source.
Fixed replay files match the core files byte for byte. Old replay stays separate.

## Runner hooks

`experiments.learners.frozenlake.build` exports `validate_config(config)`,
`build_agent(config)`, `smoke_config(algorithm)`, and `smoke_overrides(config)`.
Validation needs no JAX import. Configs must be fully resolved.
The shared runner owns attempt IDs, source bundles, time limits, and output paths.
Set `CCE_RUN_DIR` to a fresh output folder before `learn(verbose=False)`.
The agent keeps `config == config` and `evaluate(n)`.
Fixed agents expose `evaluation_identity(n=None)` with a hash of the actual keys.
Old eval consumes the train key. It has no fixed bank hash.

Fixed runs log `env_steps` in `eval.jsonl`.
Old runs keep `metrics.log` and their episode axis.
Fixed training has no early stop. Old early-stop rules stay in the copied code.
The fixed learner does not run the held-out final test from the old launch script.
Weights can be loaded with the old API; **exact resume is not supported**.

## Env contract

`FrozenLakeEnv` has four legal actions. A wall clamps a move.
Obs and state are the tile index. The env object holds the map and slip rule.
To restore a branch, keep that same task, tile, and explicit JAX key.
Reuse a key for coupled draws. Split keys for independent draws.
There is no hidden live RNG to consume and no auto-reset in `step`.

`step` reports raw terminal flags. Training has no time limit.
`step_with_limit` adds a caller-owned cutoff; it keeps the final obs.
A cutoff bootstraps. A goal or hole does not. A terminal event wins at the limit.
This helper is not inserted into the old training rule.
Eval cuts off at 200 steps; this does not turn its last step into a training terminal.

## Checks and gaps

The CPU probe uses four cores and a 180-second external timeout per process.
Each run has 32 steps, eight updates, four blocks, a 16-entry replay, and h4/N4.
The map is 4x4. Eval uses four episodes. These are explicit reduced settings.
Uniform, PER, and additive CCE match old/old and old/new exactly on the saved fields.
Replay wraps. Target sync runs. Eval leaves the fixed train key intact.
The tiny training traces have no goal rewards. Separate env tests check goal/hole.
Contracts also check deterministic, graded, and full slip draws with 12,000 keys.
Frequency tolerance is 0.02 per outcome, fixed before the run.

No full 8x8 trajectory, GPU parity, full 1,000-episode bank, exact resume,
paper-figure parity, or held-out test is claimed. The probe checks final state,
not a full per-step loss/gradient trace. The first report covers three arms.
The added `parity-extra-arms.json` covers CCE-only and multiplicative behavior.
Both match old/old and old/new exactly on all saved fields. The first report
and its hash stay intact. `extra-arms-smoke.json` changes only the mix fields
of the first CCE smoke. The full test file now has 17 passing cases (35.46s).

Run from the repo root with the pinned research deps:

```bash
timeout 180s env PYTHONNOUSERSITE=1 PYTHONPATH=.:src JAX_PLATFORMS=cpu \
  python -c 'import os; os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4]); import pytest; raise SystemExit(pytest.main(["-q", "tests/test_frozenlake_migration.py"]))'
```

`compare_reference.py` runs in two fresh processes. Use the sealed runtime `src`
as `PYTHONPATH` for `reference`, and `.:src` for `migrated`. Pass
`--recipe configs/experiment/frozenlake-smoke.json --task 0` (also 1 and 2).
Use a new `CCE_RUN_DIR` for each migrated run. Repeat the reference first.
Compare the final JSON line exactly. Keep the 180-second process timeout.
