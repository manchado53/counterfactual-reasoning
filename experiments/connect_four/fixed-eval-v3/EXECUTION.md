# Launch evidence

User approved the fixed 500/500 test-bank design and launch on October 14, 2026.
Base: freshly fetched remote master 5f2d313, plus the v2 implementation 391399e.
A separate branch/worktree preserves the active v2 source and all DoorKey jobs.
Branch: experiment/connect-four-fixed-eval-v3.

## Validation

31 CPU checks passed initially: 12 new evaluation tests, 16 v2 regression tests,
and 3 source/registry/catalog tests. After the search-cache fix, all 13 evaluation
tests passed again, including a new two-fresh-process regression. That is 32
unique passing checks. CPU logs are retained under runs/.

First GPU smoke array 283917 failed in Uniform task 0 with an unresolved LLVM
symbol from the recursive Numba disk cache. Tasks 5 and 10 completed, but that
array did not pass the launch gate. Disable disk caching for search and compile
within each process. No shared environment, DoorKey job, or v2 job was changed.

All replacement GPU smokes passed at code commit
3da72e491ab97f6d4e3251eee8045e83c80fc7f0:

| Job | Arm | Decisions | Updates | Warm block | Warm evaluation |
|---|---|---:|---:|---:|---:|
| 283921_0 | Uniform | 21,504 | 5,376 | 9.55 s | 2.65 s |
| 283921_5 | PER | 21,504 | 5,376 | 9.36 s | 2.10 s |
| 283921_10 | CCE decay | 21,504 | 5,376 | 21.80 s | 2.05 s |

All used a real CUDA backend, the full CNN, full buffer capacity, and all 1,000
cases at identical steps. Finite updates, changed parameters, checkpoint reload,
RNG isolation, and bank identity passed. CCE scored coverage was 1.0; zero,
midpoint, and ranked-TD endpoint checks passed. Every saved path/outcome artifact
was checked against its evaluation hash. Initial outcome hashes match all arms.
Every bundled Python source hash matches the worktree used for full submission.
These are smoke checks, not replay-comparison results.

Bank SHA-256:
15921cffa6df3e13751a1303bf0d02ce17213d60022651ecd7ca0f3240976cb2

## Full submission

Submit tasks 0–14 with cap 4 after sbatch --test-only succeeds.
V2 array 283882 has cap 3 and one DoorKey task 283824_14 remains, so the combined
maximum is eight. Do not cancel or edit either existing array.
Store the returned job ID, source commit, and monitor PID in
runs/train-submission.json; report the ID in the PR and handoff.
Start the read-only training/monitor_connect_four_v3.py immediately on submission.

Direct smoke extrapolation is about 1.9 h per control and 4.3 h per CCE run.
The smoke buffer holds only 21,504 entries; full runs reach 100,000. Allow 2–4 h
per control and 5–10 h per CCE run, plus unknown queue time. With cap four, allow
roughly 15–30 h for the full set, depending on packing and actual replay cost.
Each task has a 48-hour Slurm limit. Smoke timing is not a completion guarantee.

## Choices retained explicitly

Training still uses the deterministic win/block/center foe from v2. Minimax
(depths 1–5) is for evaluation only. The agreed change is the evaluation bank,
not a new training distribution. Other training and replay settings are identical
to v2. The 50 openings, 500/500 split, heuristic weights, and per-case seed rule
are our declared choices; they are not an exact reproduction of a paper.
See README.md for metric definitions and the limits of repeated game paths.
