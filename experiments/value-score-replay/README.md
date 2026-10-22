# Value scores versus rollout CCE

Same saved setup, different score:

```text
state -> simulated returns -> CCE score -> original replay rule
state -> target-network Q gaps          -> original replay rule
```

Issue #47. Original design: three seeds (0–2), two arms, two tasks.

**Current launch:** at the user's request, array 286966 was cancelled and
replaced by six **value-only T4** jobs (array 286976, tasks 1,3,5,7,9,11).
At most six GPUs run at once. Two value-only T4 smoke jobs (286974) gate
production. CCE controls are stopped; H100 partial logs remain for audit.
Fresh CCE reproduction is not complete. Value runs restart from their seed,
not from partial checkpoints. Future wall-clock comparisons need matched hardware.
The first comparison is CCE versus its value-score replacement. Old Uniform/PER
curves are context only; the old JaxNav chart mixed collector versions.

## What is held fixed

- JaxNav: all 65 saved config fields from tasks 33–35 of array 280874.
  Fixed collector; 8x8 sparse maps; 1,000-map bank seed 20260923;
  CCE-only -> ranked-TD fade at 100k–150k; 200k episode limit.
- FrozenLake: all 36 saved fields from runs 257599–257601. Deterministic 8x8;
  multiplicative mixing; 15k episode limit; **0.95 early stop retained**;
  256 environments x 128 collection steps. No new collector or stopping rule.
- All collection, training, replay, evaluation, and target-sync code is inherited
  from historical sources. The modern learner migration is not used.
- Score refresh/insert timing, valid-state filtering, sampled slots, and one
  parent JAX-key split per scoring call are retained. Child rollout keys are
  discarded in the value arm; they never changed the parent key.
- JaxNav drift checks remain enabled. They use value scores in the value arm,
  with the same call sites and random-stream steps.

## Score definition

`abs(Q_target(s,stored_action) - Q_target(s,alternative_action))`.
FrozenLake takes the largest gap, matching its historical effective max
aggregation. JaxNav takes the weighted mean over other actions, using its
unchanged **online** Q softmax weights at temperature 0.5. The target network
supplies gaps only. This substitutes learned mean differences for return
distribution differences; it is not an equivalent CCE estimator.

The value arm raises if a scoring rollout builder is reached. Its final check
requires that no rollout function was built and that value scoring ran.

## Source evidence and limit

`manifest.json` holds configs recovered from original logs, log hashes, and
336 source-file hashes. `prepare.py` reconstructs the local runtime:

- JaxNav files must match the original sealed source hashes. The original Python
  package versions are enforced. Eval-bank identity is checked on every full run.
- FrozenLake comes from pre-run Git revision `807367d`, with original logged
  configs (also checked against saved checkpoint 257600). There is no source or
  software-version hash stored with the May runs. This is a reconstruction of
  that source, not a claim of bit-for-bit historical replay. Both new arms use
  the same reconstructed source and software. Fresh CCE controls test whether
  the gain persists. The three seeds do not reproduce the 25-seed headline.

No success or speedup is assumed. Small smoke timings are not production speed
claims. Jobs that hit the wall limit are incomplete; keep their observed tails.

## Run and inspect

Use the historical conda environment in the sbatch files. Run `prepare.py` only
before sealing/submission; it needs the archived worktree and original log paths
listed in that script. Its output lives in ignored `work/value-score-replay/`.

```bash
python experiments/value-score-replay/prepare.py
python -m pytest -q experiments/value-score-replay/test_value_scores.py
python experiments/value-score-replay/report.py --plot
```

Current T4 gate: two value-only jobs, each capped at 20 minutes.
Current full batch: six one-GPU jobs capped at 12 hours each, at most six concurrent.
Production ceiling: **72 GPU-hours**, plus 40 GPU-minutes for T4 gates and
about 20 GPU-minutes already spent on H100 gates/partial runs. Total stays
below the approved 96 GPU-hour cap. No automatic retries.
The CCE reference JaxNav seed-0 run took 6h47 on H100, within the full-run cap;
new gate timings verify startup but cannot predict final learning time.

Status JSON and `study_eval.jsonl` contain run locations and actual observations.
Use Slurm state as the authority for killed/timed-out jobs; a process killed by
Slurm cannot update its status file. `report.py` prints each process's last
recorded state, not a fresh scheduler assertion.
