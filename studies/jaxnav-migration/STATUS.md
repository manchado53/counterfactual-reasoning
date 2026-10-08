# Work block through FrozenLake — October 8, 2026

This is **not** a whole-migration completion claim. Production imports and running
training jobs have not been switched. Keep PR #30 draft until its open gates are
reviewed. No full training sweep, main-branch merge, or source cleanup was done.

Latest focused verification: **123 tests passed in 40.80 seconds**. This is the
migration test set, not a whole-repository certification.

| Requested step | Current evidence | Remaining gate |
|---|---|---|
| 1. JaxNav parity | CPU checks passed. T4 exact old/old exposed raw score drift; all other 3,122-event fields matched in the measured pair. First numeric old/new case passed all 3,122 events with no config overrides. | Separate user-approved numeric gate 281087 is running: 1/9 cases passed at this snapshot. No full GPU pass yet. |
| 2. Clean installation | Fresh isolated CPU venv installed the wheel; JaxNav and FrozenLake each ran twice, captured sources, and rebuilt identical plot data. | Historical JaxMARL dependency metadata conflicts remain explicitly accepted exceptions, not a resolver-clean install. Fresh GPU install untested. |
| 3. Historical figures | Exact plotted-data checks pass for the saved three-panel decay plot and the five-arm mean/median/SEM audit. All 77 bundled logs are hashed and self-contained. | Other historical figure families are not yet ported; no training parity inferred from plot reproduction. |
| 4. JaxNav studies | Existing 45 CCE decay rows plus 20 exact-header sparse/dense baseline rows validate. 377 raw historical rows preserved; 1,107 sealed source files verified. | Coverage/rarity, fresh-eval, hold wrappers and carry-pilot variants still have distinct protocols or source gaps; see history.md. This step is partial. |
| 5. FrozenLake seam | Fixed stochastic transfer and explicit legacy path added. Environment contracts and bounded old/old/new checks pass for Uniform, PER, CCE Add, Mul, and CCE-only; shared installed runner and plot path work. | No full historical trajectory, GPU, exact resume or held-out test claim. See the detailed FrozenLake reports for tested fields. |

## Evidence

- `install-evidence.json`: retained source hashes, configs, bank hashes, and result
  summaries from wheel-based runs. Larger copied outputs: `runs/install-check/`.
- `numeric-parity.md`: unchanged failed exact tests and the separate numeric rule.
- `historical-reporting.md`: exact plot rules, selected sources, and limitations.
- `history.md` / `history-ledger.json`: per-family recipe and recovery status.
- `../frozenlake-migration/parity-results.json`: source-pinned bounded comparisons.
- `../frozenlake-migration/parity-extra-arms.json`: additive-only and multiplicative extensions.
- `../../docs/migration-install.md`: install commands and declared dependency clash.

## Next decision points

1. Read the complete T4 numeric report before claiming the JaxNav GPU gate passes.
   Stop at any failed rank, sample, update, or evaluation field; do not widen the
   fixed tolerance after a failure.
2. Preserve pinned legacy suites for old hold/carry studies. Port each distinct
   loop with its own old/new check. Unknown historical defaults stay unknown.
3. Treat a resolver-compatible dependency upgrade as a separate behavior change,
   not a silent way to get a green installation check.

H100 parity job 281077 was cancelled while pending because it retained the earlier
broken launcher. Its source snapshot remains on disk. No training job was cancelled.
