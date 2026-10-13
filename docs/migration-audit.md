# Migration audit

The runnable experiment architecture was built on
`feature/jaxnav-parity-migration`.

The large migration workspace is preserved by Git commit `d0c2945` and annotated
tag `migration-audit-20261008`. That commit
contains the complete discovery dumps, copied historical sources, recovered log
headers, old figure bundles, and detailed parity reports created during the move.
They are not required by the normal experiment runner.

The slim tree keeps runnable learners, environment adapters, representative
recipes, tests, and short notes about known limits. Use Git when a detailed audit
artifact is needed:

```bash
git show d0c2945:path/to/file
git worktree add /tmp/cce-migration-audit d0c2945
```

The final pre-merge checks are summarized in
[`migration-trust-gates.md`](migration-trust-gates.md).

The last full-size JaxNav GPU gate was incomplete. Two candidate tasks passed
the approved numeric contract. A third stopped when two old-code runs differed
by at most `5.090493640567788e-08`, below the approved `1e-7` raw-score
tolerance, before its candidate ran. This is tiny float noise, not evidence of
a behavior change. It is also not a complete full-budget parity run.

The bounded CPU checks and experiment smoke tests remain the merge test surface.
Historical raw logs and generated plots are research data, not runtime package
code. New runs save their own resolved config, package versions, source hashes,
source bundle, device, evaluation identity, and final status.
