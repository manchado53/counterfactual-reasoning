# Sparse Four Rooms — 2× room width

This doubles each room's interior width **and height**, keeping walls and all
four doors one cell wide. 23×23 outer grid; 404 open cells (400 in rooms plus
four doors), versus 13×13 and 104 in the original study. Left rooms are each
10×10; right rooms are 12×10 and 8×10, preserving the offset wall layout.

Start (1,1), goal (16,16). Doors (5,11), (11,3), (13,16), (18,11).
Coordinates are zero-based, from top left. The goal is the first cell in the
scaled counterpart of the old G2 cell. This is a **scaled Sparse Four Rooms
variant**, not the original Sutton, Precup & Singh benchmark.

Source and inherited protocol: [original study](../cce-transfer/README.md).
The original study was stopped by user request. Raw logs remain in its own
worktree. No cancelled run is counted as a completed final-budget result.

Only map size changes in the learning task. Keep intended movement probability
2/3, other directions 1/9 each, wall collisions stay put, +1 on goal only,
200-step finite-horizon cap, fixed start/goal, all three replay arms, seeds 0–4,
524288 training steps, 32704 updates, and the full JaxNav CCE recipe.
Hidden layers remain 128×3; one-hot position plus time grows from 105 to 405
inputs, equally for all arms. Shared environment/learner/scoring/replay code
is reused. No score, optimizer, replay priority, epsilon or budget tuning.

Tasks are interleaved by seed so all arms start early:

| Arm | Tasks |
|---|---|
| Uniform | 0,3,6,9,12 |
| Rank PER | 1,4,7,10,13 |
| Full CCE | 2,5,8,11,14 |

The fixed evaluation key bank is unchanged (seed 20261015, 1000 episodes), with
all three arms sharing identical episodes on the new map. Reports, raw logs,
and monitoring are separate under `runs/four-rooms-2x`. Mean/median/seed curves,
descriptive SEM, matched bootstrap and no-forward-fill rules are unchanged.
Before launch, a known-map shortest-path policy checks the unchanged time cap;
its result is a feasibility check, never a learned agent score.

## Reproduce

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src:$PWD" JAX_PLATFORMS=cpu
python -m pytest tests/test_four_rooms.py tests/test_four_rooms_2x.py tests/test_four_rooms_report.py tests/test_shared_learning_core.py tests/test_per_max_priority.py tests/test_study_catalog.py -q
python -m training.run validate --recipe experiments/four_rooms/cce-transfer-2x/training.json --task 2
python -m training.run train --recipe experiments/four_rooms/cce-transfer-2x/training.json --task 2 --smoke --device cpu --max-wall-seconds 300 --output runs/four-rooms-2x/cpu-smoke
python -m analysis.studies.four_rooms --runs runs/four-rooms-2x/full --out runs/four-rooms-2x/report --room-scale 2
```

Use the existing JAX 0.9.1 runtime or the pinned migration dependency files,
with the historical dependency caveats in `docs/migration-install.md`.
The recipe freezes every setting. Attempts capture source hashes/bundle,
package versions, config, device, evaluation bank bytes and status.

SLURM: smoke tasks 0,1,2 after a dry run, then full array 0–14%7 after all
smokes pass. Keep the worktree at its pinned commit until queued jobs start.

```bash
export FR_ROOT="$PWD" FR_MODE=full FR_COMMIT="$(git rev-parse HEAD)"
export FR_PYTHON=/home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python
mkdir -p logs
sbatch --test-only --array=0-14%7 experiments/four_rooms/cce-transfer-2x/job.sbatch
sbatch --parsable --array=0-14%7 experiments/four_rooms/cce-transfer-2x/job.sbatch
python -m training.monitor_four_rooms --root "$PWD" --job ARRAY_ID --study-dir runs/four-rooms-2x
```

## Time-cap feasibility check

`feasibility.json` records exact propagation of state probabilities under two
fixed policies (not training):

| Policy | Original goal chance | 2× goal chance | 2× successful length |
|---|---:|---:|---:|
| Uniform random actions | 10.4648% | 0.1329% | 165.46 |
| Known-map shortest-path policy | ~100% | 99.999998% | 61.06 |

The shortest possible route grows from 16 to 30 moves. Keep the 200-step cap.
The known-map policy uses the true map and is only a feasibility check; it is
not a learned control arm or an optimal-policy claim under stochastic moves.
Rebuild with:

```bash
python -m analysis.studies.four_rooms_feasibility --out experiments/four_rooms/cce-transfer-2x/feasibility.json
```
