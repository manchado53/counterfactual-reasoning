# Migration audit

The runnable experiment architecture was built on
`feature/jaxnav-parity-migration`.

The large migration workspace is preserved by Git commit `d0c2945`. That commit
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

The last full-size JaxNav GPU gate did not pass. Two old-code runs disagreed in
three rollout-return arrays and three CCE-score arrays before the new candidate
ran. This cleanup does not turn that failed gate into a pass.

The bounded CPU checks and experiment smoke tests remain the merge test surface.
Historical raw logs and generated plots are research data, not runtime package
code. New runs save their own resolved config, package versions, source hashes,
source bundle, device, evaluation identity, and final status.
