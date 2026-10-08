# Hold and carry ports

Source: sealed `temporal_replay_20260930_v2` and
`episode_carry_pilot_20260930` suites. Exact relevant originals are retained under
`sources/`; hash manifests check those bytes. No private worktree import is needed
by the new learners.

Hold has 40 rows: PER, CCE Mul, CCE Add, and CCE-only; hold 1/20; five seeds.
Its collector resets episode accounting at chunk boundaries, just as before.
Hold 1 still uses the historical held-key rule; it is not the plain baseline RNG.

Carry has 20 rows: five arms, old/fixed collectors, two seeds. It keeps a step
budget and episode-based epsilon. The old and fixed collector classes are copied
separately. The learning loop keeps update order, target sync and evaluation.
Budget fields once stored in the launch manifest are explicit config fields now.
`save_step_checkpoints` is true in all research recipes; smoke disables these
extra output files. This does not change learning.

Eight hold and ten carry CPU comparisons passed. Tests check frozen class and
parent code, then compare final parameters, targets, optimizer, random keys and
evaluation logs. Hold also compares final CCE score arrays. Reference collectors
are compiled from the saved source; shared parent equivalence is checked separately.
These are small training checks, not full historical trajectories or a claim of
GPU bit identity. Each test process had four CPU cores and a 180-second cap.

The shared runner also completed carry task 13 with explicit smoke settings:
32 steps, eight update attempts, two evaluation maps, and unchanged h60/N1 CCE.
The attempt records an `env_steps` axis and the actual fixed bank hash.
Evidence is in `verification.json`; the complete local output is at
`/tmp/cce-carry-shared-run-20261008`.
