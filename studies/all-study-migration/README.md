# All-study migration checkpoint

October 8, 2026. Same branch: `feature/jaxnav-parity-migration`.
PR #30 stays draft. This is **not** a whole-migration completion claim.

Final focused regression: **127 passed in 39.89 seconds**, exit zero.
The graded FrozenLake source/recovery and both collector paths add nine passing
checks. See `checks.json` for scope and timings; these counts overlap other reports.
Frozen source whitespace is retained so its saved hashes stay valid.

```text
old source + saved settings -> new run path -> bounded old/new checks
missing proof              -> named gap, not a guessed default
```

## Scope saved here

`discovery.json` records 46 known Git refs, 23 distinct trees, and 22 worktrees.
It includes ignored research source under `work/`. No scan cap was hit. It does
not cover all remote-only files or every external run folder. No fetch was done.

`catalog.json` lists 30 training recipe files with 4,032 rows and 15 analysis
configs at this checkpoint. Rows are **not unique completed experiments**.
Some are smoke templates or alternate manifests for overlapping old jobs.
Validation is not execution or proof of a past result.

| Family | New path | Proof and gaps |
|---|---|---|
| JaxNav decay | Existing port retained | CPU gates retained. Full-size T4 gate failed; see below. |
| JaxNav hold/carry | 40 hold rows; 20 step-budget pilot rows | 8 hold and 10 carry reduced CPU pairs passed; frozen source checks; shared carry runner passed. |
| FrozenLake | Fixed transfer, old collectors, 792 graded logged rows | Existing fixed checks; four graded-arm old/new checks at slip .333. Earlier headers kept blocked when this source era does not fit. |
| DoorKey/lava, ConnectFour | Fixed transfers plus older protocols/layouts | Source/control and reduced checks; see small-transfer reports. |
| SMAX, chess, CVRP, RP-DQN | Separate old learners, not new CCE substitutions | Reduced old/new checks; logged settings where recoverable. Old job/source links remain unproved. |
| Budget routing | Separate budget and Option-A variants | Source and reduced checks; do not treat as plain CVRP. |
| Paper and diagnostic analysis | Oracle, cache, raw-log, gradient, suitability routes | Real-input small checks where available; synthetic checks labeled; full sweeps not rerun. |

Detailed evidence lives in:

- `../jaxnav-historical/verification.json`
- `../frozenlake-migration/README.md` and `recovered-history/`
- `../small-transfers-migration/REPORT.md` and `EXTENSION.md`
- `../legacy-environments-migration/README.md`
- `../historical-analysis-migration/STATUS.md` and `catalog.json`

## GPU parity is still blocked

T4 job 281087 failed after 20m16s. Tasks 1 and 7 passed old/old and old/new.
Task 13 stopped before the candidate: two old-code runs differed in three
rollout-return arrays and three CCE-score arrays. The other recorded fields in
their 3,122-event traces matched. Only score arrays have an approved tolerance;
rollout returns must still match exactly. The tolerance was not widened.
Six later cells were not run. No full 200k-episode replay was requested or done.

## Still open

- Diagnose the old-code rollout mismatch before claiming the GPU gate passes.
- Prove launch-time source/settings for older studies that lack a sealed source.
  A complete log header alone does not prove the hidden learning rules.
- Full no-slip raw-log rebuild lacks job `257574/metrics.log`.
- Old SMAC needs missing runtime pieces; no fake simulator check was added.
- Full historical sweeps, remaining chess/C4/SMAX analysis orchestration, and
  all historical figures are not certified by these small checks.
- The expanded wheel still needs a fresh installed-runtime check. Earlier
  JaxNav/FrozenLake wheel evidence does not certify every new adapter.

Old source, runs, and checkpoints remain untouched. No new cluster training,
dependency upgrade, main merge, or history deletion was done for this checkpoint.

See `../../docs/study-layout.md` for the run and analysis commands.
