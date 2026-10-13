# Experiment bundles

Each folder names one research question. `training.json` creates new data.
`analysis.json` reads existing data. `README.md` states the purpose and evidence
limit. Generated logs and figures belong under `runs/`, not here.

```text
experiments/<environment>/<question>/
  README.md
  training.json   optional
  analysis.json   optional
```

Past work lives in `experiments/history/`.

```text
experiments/history/experiments.json   what we asked
experiments/history/runs.jsonl         one row per recovered physical run
experiments/history/sources.json       where each row came from
experiments/history/scheduler-*.json   what Slurm reported on one date
```

The history files are pointers. They do not copy large logs or checkpoints.
They also do not turn a submitted job into a good result.
