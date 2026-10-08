# First migration slice — approved October 7, 2026

Build a source-pinned replay parity gate before wiring the new architecture into
training. Preserve all running jobs and old worktrees. Use the sealed October 7
JaxNav decay sources and actual saved configs as the reference, not current defaults.

This slice copies the reusable replay implementation into the new core, exercises
old/old and old/new in separate bounded CPU processes, and saves exact trace
comparisons. It also indexes worktree entrypoints and dirty work for later recovery.
No production trainer is switched. No sweep, merge, push, or cleanup is included.

The full approved direction lives in the reproducible-experiments planning worktree.
This slice does not satisfy full JaxNav parity: collector, CCE rollouts, Q updates,
evaluation, reporting and GPU checks remain to be built before that gate can pass.
The existing research branch is unchanged. Implementation is on
`feature/jaxnav-parity-migration`, based on verified default tip `0231116`.
