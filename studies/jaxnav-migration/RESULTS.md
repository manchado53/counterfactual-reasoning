# Migration checks

See [STATUS.md](STATUS.md) for the October 8 install, historical reporting,
FrozenLake and GPU-gate updates. Counts below describe earlier checks.

## JaxNav training path

- Nine bounded seed-0 cells passed exact old/old and old/new parity.
- Three CCE arms, crossed with old+decay / fixed+constant / fixed+decay.
- 27 isolated process runs; 42 trace events per cell, 378 matched events total.
- Real collection, score-on-insert CCE rollouts, sampled updates, target sync,
  fixed evaluation and eval/Q-diagnostic logs were compared.
- Report: `runs/parity/training-v1/report.json`.
- All 45 saved production treatment configs validate without filling new defaults.
- 42 focused tests pass, including original function AST guards and plot safeguards.
- The opt-in recipe runner completed a CPU smoke and produced a raw-log figure.
- Latest CLI artifact: `runs/parity/cli-smoke-final/`, including source zip,
  resolved config, eval/Q logs, attempt status, bank hash, figure and plotted data.
  This final CLI check ran with persistent compilation caching disabled.

These are reduced smoke runs, not 27 full research runs. Five-seed production
budgets, H100 parity, the full bank and direct gradient tensors remain unchecked.
See README for every override. The full trainer source is preserved in the report
bundle. Production collectors/running jobs were not replaced.

The nine-cell report was produced before adding a stricter runtime-version check
to the driver; no learner function changed afterward. The report is not relabeled
as a later driver run. The final CLI smoke checks the latest orchestration path.

## Earlier replay-only checks

| Check | Result |
|---|---|
| Original sealed suite | 45/45 recipe checks pass |
| T4-replacement sealed suite | 45/45 recipe checks pass |
| Trace points per suite | 6,480 |
| Old/old and old/new replay traces | exact match, including array bytes and RNG state |
| New harness tests | 14 passed |
| Existing buffer script | 25 checks passed |
| Full training parity | NOT TESTED |

The two suite manifests overlap; these are NOT 90 independent experiments.
Each suite has five seeds and three CCE arms across three collector/schedule cells.
The worker selects the old or fixed replay source but does not execute a collector.
Synthetic inputs test the replay rule. These are not learned scores or new runs.
An intentionally broken importance-weight implementation is caught by the tests.

Artifacts in this worktree (gitignored):

- `runs/parity/decay-original/report.json`
- `runs/parity/t4-replacement/report.json`

Each artifact directory includes copied hashed reference inputs, candidate sources,
the harness, and all six traces. The original reference snapshot has 400 sealed
files totaling 3,486,741 bytes. Its manifest SHA256 is
`95a7b57bf86515b3caf82c71cac644fd12ce54fd419e912eb3621845ff025a13`.
The core replay file is byte-identical to the sealed fixed-collector source.
Only the decay module's import path changes.

The old buffer script is not a pytest test module: pytest collects zero tests.
It was run directly and reported 25 passes. No whole-repo test pass is claimed.

At the earlier replay-only stage, training trace and actual rollout checks were
still open; the bounded checks above now cover them. Full-size GPU checks, direct
gradients, historical figure parity and clean-install training remain open.
Production imports, run data, and default-branch files remain unchanged.
