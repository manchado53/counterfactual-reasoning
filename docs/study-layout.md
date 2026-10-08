# Study layout

One place for each kind of work.

```text
src/counterfactual_rl/    shared CCE and replay math
experiments/envs/        task rules and state
experiments/learners/    training loops, split by task and old protocol
configs/experiment/     full run settings
configs/studies/        analysis settings; not training jobs
experiments/studies/    old paper and diagnostic analysis
studies/                source proof, checks, and known gaps
runs/                   new outputs; not source code
```

Old `src` paths stay for now. Old jobs still use them. This move does not switch
running jobs or erase the old code. Old faulty rules stay named old rules.
Different replay math stays in each port's pinned history folder. It is not
silently replaced with the latest ranked CCE.

## Find a study

```bash
python -m experiments.catalog --root /path/to/checkout
```

This lists training recipes and analysis configs. Valid settings do not mean
that training finished or that a result was reproduced. Read the linked study
reports for those checks. `studies/all-study-migration/discovery.json` states
where the search looked and what it skipped.

## Run a small check

Use the pinned runtime in `docs/migration-install.md`. Keep outputs fresh.

```bash
cce-run validate --recipe configs/experiment/jaxnav-temporal-20260930.json --task 0
cce-run train --recipe configs/experiment/jaxnav-temporal-20260930.json --task 0 \
  --smoke --device cpu --max-wall-seconds 180 --output /tmp/new-hold-check
```

`--smoke` changes compute settings on purpose. The run record saves them. It is
not a study result. Without that flag, the saved settings remain unchanged.
No command here submits a cluster job.

Training records contain settings, code hashes, a source ZIP, package versions,
device, clock, and final status. A fixed evaluation bank is hashed only when
that learner really has one. Old fresh-eval learners do not get a fake bank.

## Rebuild an analysis

```bash
cce-study --config configs/studies/claim2.json --out /tmp/new-paper-check
```

Read the config first. A saved-cache plot is not new training. A small synthetic
check is not evidence that CCE helps. Old fill rules and stopping rules remain
documented with the saved study; they are not silently imposed on new plots.

## Add a task

Keep its reset, step, legal-action and terminal rules in `experiments/envs`.
Keep its learner behind the small `build.py` interface. Validation must not load
JAX or a simulator. Register its name only once the adapter exists.

Each adapter supplies `validate_config`, `build_agent`, `smoke_overrides`,
`evaluation_axis`, and `simulator_sources`. Its learner keeps the exact config
and implements `learn` and `evaluate`. An optional `evaluation_identity` reports
real fixed-bank details. Different clocks must keep different names.

Before calling a port done, keep the old source, compare old/new behavior with
the same small inputs, run the shared entrypoint, and rebuild the report where
the needed data exists. If source or data is missing, say exactly what is missing.
