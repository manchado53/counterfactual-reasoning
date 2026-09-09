# HANDOFF — JaxNav rollout-policy experiment (written 2026-09-09)

Continue the JaxNav work from here. `lab-notebook.md` 2026-09-09 LOG is the source of
truth for WHY; this is the actionable subset.

## Where to work
```
worktree   .claude/worktrees/research+cce-robotics-transfer
branch     research/buffer-instrumentation   (pushed to origin, 56 commits ahead of master)
python     ~/.conda/envs/counterfactual/bin/python     NOT the machine default
env vars   PYTHONNOUSERSITE=1   PYTHONPATH=$WT/src     (both required)
deadline   ICLR abstract Sep 18
```

## State
```
DONE   FL coverage dose-response      208 runs, FLAT over 48x, p=0.854   -> coverage is DEAD
DONE   JaxNav capacity outcome        32 runs, 16/16 seeds collapse, p=0.001
RUN    array 275325 cce_8192_i4       8 runs, ~30h. Largely MOOT: capacity 8,192 collapses
                                      regardless of coverage. Let it finish, do not extend.
```

## THE NEXT EXPERIMENT — rollout policy (issue #20)

Best remaining lead for the JaxNav null. **Untested.** JaxNav is the ONLY one of five envs
with `cf_rollout_temperature` (0.5); frozen_lake / smax / chess / connect_four all use a
hard `jnp.argmax`. JaxNav transitions are VERIFIED deterministic, so its 20 rollouts differ
only because of injected action noise, and at T=0.5 the rollout follows the greedy action
~35% of the time — it scores a policy the agent does not have.

### Controls ALREADY EXIST — do not rerun them
Uniform and PER never call the scoring path, so `consequence_metric`,
`cf_rollout_temperature` and `cf_n_rollouts` are irrelevant to them. Their results are
metric-independent.
```
   JaxNav uniform                     n=8    manifest_capacity_outcome.json
   JaxNav cce_base (TV, T=.5, R=20)   n=8    = arm A, 250k, done
   FL     dqn-uniform                 n=29   coverage_dose experiments
   FL     shipped config (TV, greedy) n=26   = n_score_sample 128 arm
   JaxNav per                         n=3    <- only gap; 8 runs at 1.7h would match it
```

### Arms to run — CONFIG ONLY, no code changes
```
   env      arm                          cost     isolates
   JaxNav   B  TV,  T=0,   R=20          ~8.6h    the injected noise
   JaxNav   C  W,   T=0.5, R=20          ~8.6h    the metric
   JaxNav   D  W,   T=0,   R=1           ~1.6h    both — ~20x cheaper
   FL          W,   greedy, R=20         ~0.5h    does the win survive a graded score
   FL          W,   greedy, R=1          ~0.5h    ...at 1/20 the cost
   8 seeds JaxNav / 26 seeds FL   ~200 GPU-h
```
FrozenLake is ALREADY greedy, so `consequence_metric` + `cf_n_rollouts` are pure config
there. Drivers to copy: `slurm/sweep_capacity_outcome.py` (JaxNav) and the
`coverage_dose` preset in `agents/frozen_lake/run_experiments.py` (FL).

### LANDMINE — read before setting cf_n_rollouts=1
`compute_total_variation` returns **0.0** when either sample has `len < 2`, and returns
exactly 0.0/1.0 when both are constants. So **TV + R=1 silently zeroes every CCE score**
and degrades CCE to uniform with no error. Wasserstein has no such guard:
```
   N=1  total_variation 0.00000  DEAD      N=20  total_variation 1.00000  saturated
   N=1  wasserstein     0.25200            N=20  wasserstein     0.25200  IDENTICAL
```
`cf_n_rollouts=1` REQUIRES `consequence_metric='wasserstein'`.

### Optional 6th arm — needs an UNAPPLIED, UNTESTED patch
The symmetric "inject noise into the winner" arm (FL at T=0.5) needs a rollout-temperature
knob in FrozenLake. Adrian declined the edit; it is saved at
`/tmp/cce_saved/fl_rollout_temperature.patch` (31 lines, `git apply`). **Never run.** If
applied, parity-test T=0 against the committed argmax path first.

## Gotchas that have already cost runs
```
  dh-node12 is a BAD NODE. Now excluded in train_frozen_lake_dqn.sh, and by every JaxNav
     array driver. 12 of 36 FL runs died on it before the fix.
  Sweeps submitted from a WORKTREE ran MAIN-REPO code until 2026-09-08. run_experiments.py
     now passes REPO_SRC and the sbatch PREPENDS it. Verify `REPO_SRC=` in the .out.
  ALWAYS filter analysis on sacct state. Keying off "has a metrics.log" counts CRASHED runs
     as "never solved" and biases results unevenly across arms.
  squeue rows != task counts. A PENDING array is ONE row. Use `squeue -r`.
  SLURM array tasks write to JobIDRaw, not <array>_<task>.
  n_score_sample is no longer GPU-memory capped (rollout is chunked, cf_score_max_envs).
  NEVER read a sweep before it finishes — it flattered a claim three times this session.
```

## DO NOT
```
  - pursue COVERAGE. Killed 2026-09-09 from both directions, 240 runs.
  - shrink the JaxNav buffer. 16/16 seeds collapse.
  - sweep beta.
  - present docs/jaxnav_story.md as written — beats 2,3,4,7 argue the refuted hypothesis.
```

## Also owed
```
  1  rewrite docs/jaxnav_story.md around what survived (no compute, highest priority)
  2  issues #17 (retracted, updated) #20 #22 are the live ones
```
