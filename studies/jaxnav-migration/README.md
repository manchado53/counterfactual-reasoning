# JaxNav migration: first parity gate

Status: JaxNav training path wired; bounded CPU parity and reporting checks added.
Production-budget/GPU parity and whole-project migration are NOT done.

The reference is the sealed October 7 ranked-TD decay study, not the latest
mutable research branch. The original suite records arrays 280874/280875/280876;
280899 replaces the V100 allocation. Scheduler completion is not inferred here.

## What this build tests

Three CCE arms (multiplicative, additive half, CCE-only), five seed configurations,
old-collector decay and fixed-collector constant/decay: 45 training recipes.
Collector labels select the matching frozen replay source. They do NOT mean this
test executes either collector. We feed small synthetic transitions and scores,
not production experience. Buffer capacity is deliberately reduced to 17.

Each source runs in an isolated CPU process. Old-versus-old is checked before
old-versus-new. Arrays retain dtype, shape and raw bytes. No float tolerance is
used. We check score/TD updates, caching, wrapping, ranks, sampled transitions,
importance weights, schedule start/mid/end, and NumPy RNG state.

The new core replay module uses the study's exact rules. The new research trainer
imports that module and the extracted scoring code. Existing trainers and running
jobs are untouched. The new path is opt-in through the run command below.

## Run

Use the Python environment recorded by the sealed study.
From this checkout:

```bash
PYTHONNOUSERSITE=1 python -m experiments.parity.run \
  --reference /path/to/sealed/decay_20261007_v1 \
  --output /path/to/new/parity-output

PYTHONPATH=src:. JAX_PLATFORMS=cpu python -m pytest -q tests/test_parity_harness.py
```

The runner verifies all sealed input hashes, copies source/config inputs (not
training logs), captures candidate Python sources, and saves a report plus traces.
It refuses output reuse and source hash drift. It forces CPU and one worker at a
time. Each of six workers gets 60 seconds wall time, 55 CPU seconds, 8 GiB address
space, and a 64 MiB per-file limit. Source bundles are capped at 100 MiB each.
The runtime checks Python, NumPy, SciPy, JAX and jaxlib; this is NOT a full lock of
every training dependency.

The report exits nonzero on mismatch or worker failure. Trace files show the first
differing field. Reference source is copied intact; no current implementation is
used to generate expected outputs. Independent endpoint tests use hand-derived
average ranks rather than the sampler's own helper.

## New runnable layout

```text
src/counterfactual_rl/replay/       replay and decay
src/counterfactual_rl/scoring/      CCE distance metrics
experiments/envs/jaxnav.py         simulator adapter
experiments/learners/jaxnav/       frozen math and old/fixed collectors
experiments/common/               stopping and timing
experiments/run.py                recipe -> fresh attempt
configs/experiment/               45 resolved treatment recipes
analysis/plot_migrated_runs.py     logs -> mean/median + source data
```

The CCE learner accepts its buffer class explicitly. It does not patch a global
module to change replay. Legacy collector files are explicit variants, not defaults
silently changed under old run IDs. The observation/action details stay in the
JaxNav adapter. Do not force other environments through this trainer.

From the checkout, using the reference runtime:

```bash
PYTHONPATH=src:. python -m experiments.run validate \
  --recipe configs/experiment/jaxnav_decay_20261007.json --task 13

PYTHONPATH=src:. python -m experiments.run train \
  --recipe configs/experiment/jaxnav_decay_20261007.json --task 13 \
  --smoke --device cpu --vendor /path/to/pinned/vendor \
  --output /new/smoke-attempt --max-wall-seconds 300

python analysis/plot_migrated_runs.py --runs /new/smoke-attempt \
  --output /new/smoke-curve.png --bin-size 1

PYTHONPATH=src:. python -m experiments.parity.training \
  --reference /path/to/sealed/decay_20261007_v1 \
  --output /new/training-parity
```

The runner saves settings before training, captures a recoverable source zip, records
package versions and hashes the simulator sources. Eval-bank identity is recorded
after evaluation, including on a partial run. Output reuse is rejected. Resume is
not supported. No command submits a scheduler job.

The plotter rejects different eval banks/simulator versions, incompatible non-method
settings, duplicate method/seed attempts and non-monotonic eval logs. It plots equal-
weight seed means/medians on shared observed bins only. These are not confidence
intervals; no significance claim is made. It does not yet read historical logs.

## Training parity scope

The bounded matrix uses seed 0 in nine cells: Mul/Add/CCE-only, crossed with old
collector + decay, fixed collector + constant, and fixed collector + decay.
Each uses two old runs and one new run, in separate processes. It calls the real
`learn()`, score-on-insert rollouts, replay, optimizer, target sync, and fixed eval.
It compares collector inputs/outputs, rollout outputs, score/TD arrays, sampled
indices/weights, update inputs/outputs, parameter/optimizer state, RNG, eval banks,
and scientific eval/Q-diagnostic logs exactly.

Overrides are saved: two environments, four-step chunks, batch four, buffer 17,
eight-step episode limit, six training episodes, two eval maps, earlier decay/target
clocks, and smaller score chunks. Network size and h60 scoring horizon are retained.
This does NOT prove full-size GPU equality, the full 1,000-map bank, or every
bootstrap/long-episode branch. Direct gradient/TD-target tensors are not separately
captured; update inputs, TD errors, optimizer state and resulting parameters are.
AST fixtures additionally guard unchanged original learning functions/control flow.

## Still blocks full migration

- Full production-scale bank and long-horizon branch checks; direct gradient capture.
- Historical figure reproduction and the 15 historical constant-old controls.
- Production launch-time progress/coverage assertions and scheduler tooling.
- GPU repeatability, full-budget training, checkpoint resume.
- Clean-install training dependencies and all other environments.

Passing bounded parity is not proof that all long learning curves will match.
The JaxMARL vendor is still an explicit input; no clean-room install is claimed.
CPU tests may emit CUDA-plugin warnings from the existing GPU-enabled environment,
and persistent compilation cache warnings; they do not certify runtime portability.

## Other research

Use `python -m experiments.inventory --repo /path/to/repo --output /new/index.json`
to list branch tips, dirty files and tracked entrypoints without changing them.
This is a discovery index, NOT a backup or complete historical-run ledger.
FrozenLake, DoorKey and Connect Four have uncommitted work to preserve before
their source can move. Paper reproduction inputs stay untouched.
