# Historical run ledger

Build one source-backed catalog of every discoverable experiment and run from
this project. Search past Codex and Claude chats, Git history and worktrees,
saved recipes and manifests, paper caches, run directories, scheduler records,
and figure provenance. Treat chat text as a lead, not proof.

The catalog will separate an experiment question from each run that tested it.
Every entry will record its environment, arms, seeds, exact config or recovery
source, job IDs, output locations, completion and validity state, known bugs,
result artifacts, and evidence level. Missing data and conflicts stay explicit.

Keep raw logs, checkpoints, and chat transcripts out of Git. Commit a small
machine-readable ledger, readable summaries, bundle links, and a repeatable
audit script. Verify known FrozenLake, JaxNav, SMAX, CVRP, Chess, DoorKey,
Connect Four, and RPDQN anchors; detect duplicate or orphan runs; then run the
catalog tests and push the result to the existing migration branch and PR.
