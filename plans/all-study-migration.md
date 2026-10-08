# All-study migration — expanded work block

Requested October 8, 2026. This expands implementation beyond the earlier
steps 1–5. The full architectural direction remains the approved reproducible
experiments plan. This document is the implementation scope for approval; no
additional environment port is implemented by writing it.

Bring every discoverable study into the same recipe, run-record, and reporting
layout. Search all local and known remote branch refs, worktrees (including
uncommitted research), notebooks, saved launch manifests, paper reproduction
inputs, raw-log figure scripts, and retained run roots. Record the search coverage
and gaps; do not equate the first list of environment names with a complete search.

Known families to cover: all JaxNav variants; deterministic, stochastic and graded
FrozenLake; DoorKey and lava variants; Connect Four; chess; SMAX and older SMAC;
CVRP/routing and budget probes; Claim-1/oracle and Claim-2/learning analyses;
covariance/theorem tests; suitability and precision-at-k analyses; RPDQN; and any
additional family discovered during the audit. Preserve failed, cancelled, invalid,
null, superseded and planned studies with their true status.

For each study, retain its original source (including required uncommitted files),
settings, seeds, dependency record, available logs/checkpoints, figure code, and
known defects before changing paths. Source-backed runnable recipes move into
configs; environment and learner code move into experiments; reusable CCE stays
in src/counterfactual_rl; analysis-only studies get analysis entry points rather
than pretend training recipes. Never replace an old algorithm with today's CCE.

Keep the recipe/run, reset/step/branch, and log/report interfaces from the existing
plan. Different rewards, legal actions, opponents, terminal rules, clocks, stopping
rules, random streams and replay equations stay explicit. Share code only when two
real users have the same meaning. Historical bugs stay named historical variants;
fixes are separate changes with their own checks.

Use three parallel workers with disjoint file ownership: small navigation/board
transfers; older multi-agent/routing learners; and historical analysis/paper studies.
The main agent owns the shared registry/runner, remaining JaxNav paths, integration,
and final checks. Work remains on the existing migration branch and draft PR.

A study is complete only when its saved recipe validates, its new entry point runs
a bounded check, old/new scientific outputs meet a declared comparison rule, and
its report/figure can be regenerated where source data exists. Track evidence
status separately from code status. A lossless archive or a wrapper around a
private worktree is not a verified new-path port. If source, data or a runtime is
missing, name the exact blocker and retain a recovery route; do not guess defaults.

Use CPU smokes with four cores and a 180-second per-process cap, fresh outputs,
and source/disk caps. Preserve the ongoing T4 parity gate and all training jobs.
No new research sweeps, hidden dependency upgrades, main-branch merge, deletion,
or caching/policy-gradient implementation is included. Push completed tested
changes to the current draft PR; keep unresolved checks and partial ports visible.
