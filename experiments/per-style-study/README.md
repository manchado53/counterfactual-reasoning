# Shared replay study setup

This is the shared presentation and comparison contract for issue #43. It takes
the useful shape of Schaul et al., [*Prioritized Experience Replay*](https://arxiv.org/pdf/1511.05952):
matched replay methods within each task, raw per-task learning curves, a fixed
early checkpoint, a learning-speed comparison against Uniform, and a compact
result table. The paper's Figure 8 uses median
curves and interquartile bands across eight runs. Our seed count must be stated
for each task; existing studies with fewer seeds do not become eight-run studies.

## Freeze each comparison cell before training

An environment cell has one simulator version, learner, reward rule, fixed
evaluation bank, training axis, seed list, budget, early checkpoint, and score
definition. The only allowed config differences are replay-method fields named
in `method_config_keys`. Train all arms with the same seed list. Record exact
sampling rules: Uniform, canonical rank PER, raw-TD proportional PER, and any
CCE rank or raw-score blend are distinct arms. The PER audit (#41) and blend
study (#42) supply their checked definitions. Avoid calling a rank-normalized
CCE mixture "rank PER".

Use environment steps where available. If an older run records episodes or
chunks, its plot says so, and it stays in a separate cell. Predeclare the early
checkpoint; keep full training curves. Do not choose a zoom after seeing wins.
Score plots use the task's own units. This setup does not pool raw scores across
different tasks or claim an Atari reproduction. The PER paper's human-normalized
Atari score has no automatic analogue for our mixed tasks.

The candidate panel is FrozenLake, JaxNav, DoorKey, Four Rooms, and Connect Four.
Add a task only after its run files pass the shared checks. A within-task core
comparison should have Uniform, a validated PER rule, and CCE; a task with
both rank and proportional PER or both CCE blend rules can show those extra
arms in the same panel. Each cell records its own exact seed list and budget.
Until #41 and #42 settle their controls, this is a reporting setup and a
protocol template; it does not schedule new training.

## Study file

Save a JSON file outside this folder with one or more cells. Paths to run
directories may be absolute or relative to the JSON file. Each directory must
contain `config.json`, `attempt.json`, and an evaluation JSONL file. By default
the file is `eval.jsonl` and its checkpoint field is the training axis name.
Set `log_file` and `step_field` in a cell for another saved run format, such as
CartPole's `evaluations.jsonl` and `step`. The identity checks still use the
saved attempt record.

```json
{
  "schema_version": 1,
  "study": "fixed-bank-replay-comparison",
  "cells": [
    {
      "id": "example-fixed-task",
      "title": "Example fixed task",
      "environment": "frozenlake",
      "axis": "env_steps",
      "metric": {
        "field": "win_rate",
        "label": "Goal reach rate",
        "bounds": [0, 1],
        "higher_is_better": true
      },
      "budget": 1000000,
      "early_step": 250000,
      "seeds": [0, 1, 2, 3, 4],
      "arms": ["uniform", "proportional_per", "cce_rank"],
      "method_config_keys": ["algorithm", "priority_mixing", "mu", "normalise_priorities", "PER_parameters"],
      "runs": [
        {"directory": "/path/to/uniform-seed-0", "arm": "uniform", "seed": 0}
      ]
    }
  ]
}
```

The numbers and path above are format examples, not an approved FrozenLake
protocol. A real cell must list exactly one physical run for every arm and
seed, use actual observed early and budget checkpoints, and freeze its values
before results are interpreted. If an `experiment` tag differs across arms,
declare it in `method_config_keys`; the saved method config will expose it.

Run the shared report with:

```bash
python -m analysis.plotting.per_style_study \
  --manifest /path/to/study.json --output /path/to/new-output-directory
```

The command reads data only. It writes `study-results.json`, `summary.csv`,
`cross-environment.csv`, full learning curves, and an early-window curve in PNG
and PDF to a new directory. The early window ends at the fixed `early_step`.
`summary.csv` records the median score at the fixed early step, area under the
observed early curve, the full-budget score when valid, measured run time when
present, and the first observed step that reaches Uniform's full-budget score.
`cross-environment.csv` counts final-budget cells where each arm beats Uniform
at the early checkpoint, over the observed early-curve area, and at the final
checkpoint. It does not pool incompatible raw scores.
The command refuses
smoke runs, unmatched learner settings, duplicate or missing arm/seed pairs,
different evaluation banks, different simulator sources, bad scores, and
misordered checkpoints. Every plotted point was observed in **every** listed
run of that cell; no tail is filled in. A final-budget score appears only when
all attempts completed and all reached the declared budget. Otherwise the
comparison is marked `interim`. Missing measured run time stays blank.

The run files and their hashes remain in the JSON output. Keep the original
logs and failed attempts. Re-run this report after new data arrives; do not
replace an earlier dated output directory.
