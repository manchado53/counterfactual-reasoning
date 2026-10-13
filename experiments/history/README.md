# Historical experiment index

This is the map of old work.

It keeps three facts apart:

```text
QUESTION                RUN                     RESULT
What did we test?  ->   What job/config ran? -> Was the evidence usable?
experiments.json        runs.jsonl              evidence_status/result_state
```

## Files

| File | Meaning |
|---|---|
| `experiments.json` | One row per research question. Includes the evidence limit. |
| `runs.jsonl` | One compact row per recovered physical run. |
| `sources.json` | Pinned source, hash, parser, and expected row count. |
| `jaxnav-manifests.json` | Index of old JaxNav manifests at an immutable Git commit. |
| `variants.json` | Fine-grained experiment names found in chats and manifests. |
| `chat-sources.json` | Chat session IDs used to find old work. Chats are not result proof. |
| `scheduler-extra-jobs.json` | Old Slurm groups found outside recovered manifests. |
| `scheduler-snapshot-20261012.json` | Dated Slurm state. This proves queue state only. |

Large logs, checkpoints, raw chats, and copied worktrees stay out of Git.
The index points back to immutable manifests or their hashes.

## Current coverage

The index holds 50 questions and 6,020 unique runs across FrozenLake, SMAX,
CVRP, RPDQN, Chess, JaxNav, DoorKey, and Connect Four lineages. Some environment
names are split when their old manifest schema differed.

The dated Slurm snapshot queried 2,893 known IDs and returned 2,970 task rows.
Slurm completion is not proof that metrics were valid.

## Rebuild and audit

```bash
python -m analysis.history_ledger rebuild --root .
python -m analysis.history_ledger audit --root .
python -m analysis.history_ledger summary --root .
```

`rebuild` reads the pinned sources and rewrites `runs.jsonl` in a stable order.
`audit` checks source hashes, row counts, duplicate job IDs, and stale output.
`summary` checks the committed files without needing old Git objects.

Refresh Slurm state only when a new dated snapshot is wanted:

```bash
python -m analysis.capture_scheduler_history --root . \
  --out experiments/history/scheduler-snapshot-YYYYMMDD.json
```

Never replace an old snapshot. It is a dated fact.

## Status rules

- `observed_log`: config came from a saved run log.
- `submitted_manifest`: config came from a submission manifest.
- `recovered_manifest`: config was recovered from pinned old source.
- `completed`: execution finished. It may still be invalid.
- `included`: evidence was accepted by that old study.
- `excluded`: known bad, cancelled, buggy, or otherwise rejected evidence.
- `unknown`: the source does not prove the fact.

The `evidence_status` on each question is the safest short answer.
