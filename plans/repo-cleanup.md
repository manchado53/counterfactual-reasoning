# Repository cleanup

Make PR #30 smaller and easier to trust without losing reproducibility.

Keep one live owner for each module. Remove copied source when an immutable Git
commit, path, and hash can recover the same bytes. Keep exact archived bytes only
for dirty, untracked, or otherwise unrecoverable historical source. Replace large
repeated recipe JSON with compact deterministic inputs while preserving the same
expanded rows and hashes. Remove generated profiler output, duplicate paper assets,
and stale machine-specific helpers after proving that no active command needs them.

Run the focused tests, fresh-wheel check, recipe validation, figure rebuild checks,
and JaxNav parity gate after each cleanup class. Record every removed path and its
recovery route. Do not merge until the remaining merge blockers are fixed or clearly
split into a later PR.
