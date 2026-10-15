# Execution evidence

The user approved the plan on October 14, 2026.

- Base: latest remote master at 5f2d313edf9eb4dc934a8a1d6461f47d5fe3f940.
- GPU-tested learning commit: b5422c9833630df725721c2831a6db8707368e91.
- CPU checks: 16 new tests, 25 shared/source tests, 2 catalog/registry tests passed.
- Smoke array: 283871. Uniform task 0, PER task 5, CCE task 10.
- All three attempts completed on H100 with CUDA, 21,504 decisions and 5,376
  updates each. These are smoke checks, not replay-comparison results.
- All passed finite updates, changed weights, fixed-bank identity, reload parity
  and evaluation RNG isolation. CCE scored coverage was 1.0. Separate fixtures
  checked zero, midpoint and full ranked-TD blend.
- Each initial evaluation had 1,000 distinct starts and 1,000 distinct paths.
  All three arms had identical initial outcome hashes.
- Bank SHA-256: 0bbd989f9ff6ddc513d889bf9cb52a52dea67831a1b446d4a7a39538d2e5e841.
- After smoke, only documentation and the monitor location changed: the catalog
  requires executable Python outside experiments/. All bundled learning-source
  hashes were checked against the worktree and still match all three smokes.

Warm block times: Uniform 8.99 s, PER 8.79 s, CCE 21.84 s.
A direct 700-block extrapolation is 1.75 h, 1.71 h and 4.25 h respectively.
The smoke buffer has only 21,504 rows; production reaches 100,000, so ranking
and sampling will cost more. Allow roughly 2–4 h per control run and 5–10 h per
CCE run, excluding queue time. These estimates are not completion promises.
The full Slurm wall limit is 48 h per run.

Submit all 15 tasks after test-only validation. Use a cap of three while five
DoorKey tasks remain active or queued, so even if all five run the combined
GPU count cannot exceed eight. This is a conservative fixed cap; no DoorKey job
is edited. Start a persistent watch immediately after submission. Submission
IDs and live state are saved under ignored runs/*-submission.json and monitor
logs. The PR and final handoff include the full array ID.

New choices retained from the approved plan: multiplicative CCE as the sole CCE
arm; the local Connect Four model and 7,526,400-decision budget; one fixed
win/block/center training foe; five evaluation policies with varied starts.
The score chunk is 256 rather than the source maximum of 1280 to bound CNN
memory. This changes batching only, not the one-rollout score or replay formula.
