# Separate numeric gate, October 8

The original bit-exact T4 test remains **failed**. Jobs 281082 and 281085
compared the old code against itself and stopped before the new-code comparison.
This is not evidence of a migration regression.

In the first pair, 3 of 3,122 trace events differed. All three were raw score
arrays after insertion. Collection, rollout outputs, sampled indices/weights,
updates, final parameters, optimizer, evaluation bank and evaluation logs matched.

Job 281085 saved the actual score arrays. Largest absolute old/old differences:

| Collected chunks | Stored scores | Maximum absolute difference | Ranks/ties |
|---|---:|---:|---|
| 1 | 8,192 | 1.7257749895804864e-08 | exact |
| 2 | 16,384 | 2.9249400235187295e-08 | exact |
| 3 | 24,576 | 2.9249400235187295e-08 | exact |

This locates the variation in the score path. It does not identify a particular
GPU kernel as the cause. No learner change was made to suppress it.

The user approved a **separate** numeric check before any candidate comparison
under that rule. Contract `score-numeric-v1` is fixed in `experiments/parity/numeric.py`:

- Absolute tolerance 1e-7 for raw CCE score arrays only; relative tolerance zero.
- Score ranks and tie handling must match exactly.
- Every other trace field must match exactly, including sampled batches,
  weights, updates, parameters, optimizer, evaluation inputs and results.
- Saved score arrays must match their original trace hashes before comparison.
- Old/old must pass this same rule before old/new can run. Stop on first failure.
- The original failed exact reports are retained. No numeric result replaces them.

New job: **281087**, one T4. Output: `runs/parity/full-size-t4-numeric-v1-20261008`.
It retains the full production batch sizes and 1,000-map bank, but only a
three-chunk training prefix and controlled decay-clock checks. The nine seed-0
cells are not a full five-seed study or a 200k-episode replay.

## Final job result

281087 **failed** after 20m16s. Tasks 1 and 7 passed old/old and old/new,
3,122 events per comparison. Task 13 stopped on old/old:
`$.events[8].outputs.leaves[0].sha256`. The raw-score maximum difference
was 5.090493640567788e-08, within the declared score limit, but the output
hash difference is a separate field that must remain exact. The candidate
was not run for that cell. Six later cells were not tested. The tolerance
has not been widened. This is not a passing full-size GPU gate.

Evidence: `runs/parity/full-size-t4-numeric-v1-20261008/results/report.json`.

H100 job 281077 was cancelled while still pending: its sealed snapshot had the
old broken GPU backend setting. Source and output records were retained. No
training job was cancelled. No H100 pass is claimed.
