# JaxNav history: step 4

Checked 2026-10-08. This step is a partial port.
No jobs were launched. No old source or logs were changed.

```text
saved config header -> 20 new runnable baseline rows
existing decay file -> 45 existing runnable CCE rows
sealed old bundles  -> 3 pinned legacy suites
old override files  -> keep raw bytes; block unresolved ports
```

The ledger is `history-ledger.json` beside this file.
It holds exact source paths, SHA-256 hashes, seeds, cells, log links, and gaps.
A hash is a file fingerprint. It does not make a backup.

## Runnable now

| Recipe under `configs/historical/runnable/` | Exact saved settings | Rows |
|---|---|---:|
| `sparse_fixed_baselines.json` | Uniform and raw-TD PER; sparse +1/0 reward | 10 |
| `dense_fixed_baselines.json` | Uniform and raw-TD PER; dense +4/-4 reward | 10 |

Each has seeds 0–4, map 8x8, fill .3, radius .8, old chunk-reset collection,
250,000 episodes, fixed eval seed 20260923, 1,000 eval maps, and Q/loss logs.
PER means replay chosen by TD error. TD error is the gap in a value update.
These PER rows use raw TD. Do not call them ranked-TD controls.

All scientific fields come from full `metrics.log` config headers.
The recipes retain those exact headers as base64 text, plus log and file hashes.
Sparse rows come from array 277733 tasks 0–9.
Dense rows come from array 278560 tasks 0–9.
Run-folder links were read from those logs, not guessed from array IDs.

Four added runner fields are shown as `adapter_fields` in each row:
`collector_variant=old`, `decay_enabled=false`, `decay_start=100000`,
and `decay_end=150000`. They select the migrated old collector and satisfy the
shared schema. The DQN build path does not read the three decay fields.
No original config field was changed or filled from current defaults.

These configs fit the migrated legacy DQN learner. Its normal DQN path handles
both uniform and raw-TD PER; it does not build the scheduled CCE class or the
hold wrapper. The existing source map pins this learner to the sealed old
decay bundle. That pin is not proof of each September run's source version.
Full old-versus-new trajectory parity and a new runtime smoke remain unchecked
by this step. The rows are runnable config ports, not a claim that those checks passed.

Validation needs no changes to `recipes.py` or `run.py` for these 20 rows:

```bash
python -m experiments.run validate --recipe configs/historical/runnable/sparse_fixed_baselines.json --task 0
python -m experiments.run validate --recipe configs/historical/runnable/dense_fixed_baselines.json --task 5
```

The existing `configs/experiment/jaxnav_decay_20261007.json` stays the sole
45-row decay recipe. This step checks all its rows and does not duplicate it.
It covers Mul/Add/CCE-only, old/fixed collectors, constant/decay CCE, sparse
reward, fixed eval, and Q logs. Read the separate parity report for its run gates.

## Family limits

| Family | New path now | Old rows still blocked from the new path |
|---|---|---|
| Coverage | Existing h60 score-on-insert decay slice | Sampled/greedy coverage and h200: launch defaults/source not pinned per run; old replay rules need a checked port |
| Radius and reward | New .8-radius sparse/dense baseline rows | .3/.5 radius rarity and crash -1 CCE cells: raw inputs kept; source/default gaps remain |
| Fixed versus fresh eval | New 1,000-map fixed baseline rows; existing fixed CCE | Fresh-eval overrides omit seed/default details; do not assume null from absence |
| Hold versus no hold | None from the hold suite | Hold 1 and hold 20 use a sealed wrapper; even hold 1 needs an RNG/collector parity check before dropping that wrapper |
| Carry versus chunk reset | New old baseline rows; existing fixed CCE rows | The carry pilot uses its own step-budget loop and endpoint checks; mapping it to the standard episode loop would change the protocol |
| Q/loss diagnostics | Both new baseline files; existing CCE | Old smoke/baseline overrides alone do not pin Q-probe defaults or launch source |
| Replay mixing | New raw-TD PER; existing ranked CCE Mul/Add/only and decay | Old unranked CCE rules and temporal hold arms cannot be routed through ranked scheduled CCE without a checked implementation |

The 15 raw archive files hold 377 source rows. Some are smoke rows or overlap
other manifests. This is not a count of unique completed runs.
Unknown settings stay unknown. A field in a baseline's config may be unused.
For example, a DQN row's CCE horizon is not evidence that it ran CCE rollouts.

## Pinned legacy recovery

| Suite | Worktree and retained bundle | Hash checks |
|---|---|---:|
| Hold 1/20 | `jaxnav-temporal-replay-arms/work/temporal_replay_20260930_v2` | 269/269 |
| Carry pilot | `jaxnav-episode-carry/work/episode_carry_pilot_20260930` | 438/438 |
| Old/fixed decay | `jaxnav-decay-carry/work/decay_20261007_v1` | 400/400 |

All roots are under the shared `.claude/worktrees/` directory.
The ledger records absolute paths and hash-manifest fingerprints.
All listed source bytes matched their sealed hash manifests in this audit.
Saved smoke/check records and dependency records are linked in the ledger.
Saved pass flags are old evidence, not new tests run here.

To recover, retain the whole listed bundle: source, vendor, scripts, manifests,
version records, and launcher. First check each file against `hashes.json`.
Restore the recorded Python/package versions and the recorded JaxMARL vendor.
Do not substitute the current worktree or regenerate a bank under new defaults.

The hold launcher sets `PYTHONPATH` to `runtime/vendor:runtime/src` and calls
`runtime/scripts/train_temporal_replay.py --work <bundle> --mode <mode>`.
The carry launcher picks the row's `variant`, sets `vendor:<variant>/src`,
and calls `scripts/train_carry_pilot.py --work <bundle> --mode <mode>`.
The decay launcher calls `scripts/decay_run.py <bundle>`; that runner checks
source hashes and package versions before building its old/fixed learner.
Each uses scheduler task/run IDs. Keep those launcher contracts on recovery.
These are recovery notes, not permission to launch.

The source bundles remain external. This step made no independent backup.
Saved bank hashes exist for hold/carry/decay checks. Full original bank content
and generator identity are still missing for some earlier runs.

## Lossless import and export

`experiments.history` uses only the standard library. It imports no learner.
It keeps exact raw bytes, source hashes, row IDs, and all row metadata.
It never merges defaults. It rejects duplicate IDs/JSON keys and changed bytes.
Export restores the original manifest format; it does not create a runnable recipe.
Both commands refuse to overwrite files.

```bash
python -m experiments.history import old-manifest.json archive.json
python -m experiments.history import old-configs.txt archive.json --encoding base64-lines
python -m experiments.history verify configs/historical/coverage.json
python -m experiments.history export configs/historical/coverage.json recovered-manifest.json
python -m pytest -q tests/test_historical_recipes.py
```

The tests restore every raw archive without the old worktrees. They check all
65 runnable rows, prove that the 20 new rows match their retained config headers,
and check tampering, duplicate IDs, missing headers, and overwrite refusal.
No simulator or GPU is needed for these tests.

Step 4 does not finish the full history port. Fresh-eval, all rarity/coverage
cells, hold wrappers, and carry-pilot protocol still need the checks listed above.

## Bounded recovery follow-up

Hold and carry now have tested bundle exports. This reduces the recovery gap.
It does not port their trainers into the current runner.

```bash
python -m experiments.history export-bundle studies/jaxnav-migration/history-ledger.json temporal_replay_v2 /tmp/hold-recovery.zip --repo .
python -m experiments.history export-bundle studies/jaxnav-migration/history-ledger.json episode_carry_pilot /tmp/carry-recovery.zip --repo .
```

Each command checks the pinned hash manifest, every sealed file, and the exact
archived recipe before writing a new ZIP. It rejects overwrite, path escapes,
symlinks, source drift, and inputs over 64 MiB. It does not launch anything.
ZIP members keep their paths relative to the old bundle root.
`recovery-info.json` lists each exported file hash.
Saved smoke reports and extra package locks are marked as unsealed additions.
No old job-submission record, run data, or checkpoint is copied unless sealed.

| Export check | Hold | Carry |
|---|---:|---:|
| Sealed files matched | 269 | 438 |
| Exported source/record members | 280 | 449 |
| Uncompressed source bytes | 9,717,068 | 9,547,562 |
| ZIP bytes in this check | 3,622,335 | 3,737,087 |
| Original source/package verifier at a fresh temp path | pass | pass |
| Original saved train-mode smoke gate | not in trainer | pass |

The ZIP adds one `recovery-info.json` member beyond those member counts.
ZIP byte counts can vary with timestamps; member bytes and hashes must match.
The checks used the recorded conda Python with isolated imports. Only the old
`digest`, `verify`, and carry `validate_smokes` functions were loaded by AST.
AST means parsed Python code; the suite's prepare/submit/main code did not run.
Both checks passed with the original source left untouched.
Scratch exports were removed by the temp-directory context after the check.
There is no new permanent backup, fresh dependency install, or GPU smoke claim.
The launchers still name this machine's original conda interpreter.

## Exact remaining migration gates

Paths below are under the named old worktree or sealed bundle. Their hashes and
absolute roots are in the ledger. These are acceptance tests to implement/run;
the recovery test above does not satisfy training-parity gates.

| Blocked family | Source contract | Acceptance needed before current-runner support |
|---|---|---|
| Hold/no-hold | Hold bundle `runtime/scripts/train_temporal_replay.py:49` and `:115` | Port the exact goal-bit/seed/step-block RNG rule and wrapper. For hold 1/20, replay the source's four 8-env x 32-step chunks at epsilon 1, 1, .8, .2; compare actions, exploration draws, states, and keys. At forced step 199 and epsilon 0, match parent reset/timeout outputs and CCE state-to-observation checks. Do not treat hold 1 as ordinary no-hold DQN. |
| Carry pilot | Carry bundle `scripts/train_carry_pilot.py:80` through its final checks | Preserve 2,000 blocks, eval every 50 blocks, 16,384,000 primitive steps, episode-based epsilon, and last-four-check endpoint. Replay old/fixed smoke paths: 8 blocks, checks at 4/8, 65,536 steps. Fixed must retain lengths over 32 and at most 200, have no ended-state transitions, and keep the same bank hash. Compare update/target clocks and collector carry; the standard episode loop is not this pilot. |
| Sampled/greedy coverage and h200 | Robotics worktree `slurm/sweep_coverage.py`, `sweep_greedy_coverage.py`, `sweep_score_on_insert.py:84` | Recover actual launch defaults/source for each raw row first. Freeze sampled versus insert scoring, rollout count/horizon/bootstrap, refresh clock, ranks, ties, and importance weights. Compare buffer insertion/refresh/wrap and sampled indices on fixed inputs; require all live entries scored for insert scoring. Never infer absent settings from today's source. |
| Radius/reward rarity | Raw `rarity_baselines`, `rarity_cce`, `rarity_cce_add_8x8`, `score_on_insert`, and `paper_reward` archives | Resolve each .3/.5/.8 radius, sparse/dense, crash 0/-1/-4 cell against launch source. Compare identical-state goal/crash/timeout transitions and reward outputs. Keep separate bank identities and arm/seed sets per cell. New .8 baseline ports do not clear all these cells. |
| Fresh/fixed eval | Robotics worktree `src/counterfactual_rl/agents/jax_nav/dqn_vectorized.py:129` | Recover the original seed/null setting and bank generator. Compare fresh-map RNG consumption, fixed-bank contents/step keys, metrics, and unchanged training key for fixed eval. Pin bank content hashes, not just seed and size. Missing raw seed fields stay unknown. |
| Q/loss diagnostics | Q-value worktree `src/counterfactual_rl/agents/jax_nav/qdiag.py`; referenced `tests/test_qdiag_identity.py::test_qdiag_td_errors_match_update_step` | Recover old probe defaults and source. Match recorded TD errors/loss against the actual update; compare enabled/disabled training states and RNG to test whether probes alter training. No purity claim from a saved Q log alone. |
| Mixing beyond current ranked decay slice | Hold trainer plus its sealed `runtime/src/counterfactual_rl/agents/shared/buffers.py`; existing raw mixing rows | Pin raw/ranked TD, normalization order, tie/epsilon rules, exponents, importance weights, and refresh behavior. Match probabilities and sampled indices for Mul/Add/CCE-only; test constant and decay endpoints with the original episode clock. The ranked scheduled CCE class cannot silently replace an older rule. |

Across all families: first establish old-versus-old repeatability under a bounded
runtime. Then compare old/new traces with tolerances fixed in advance. No new
training, full sweeps, or broad parity claim was made in this follow-up.

## Manifest size and identifier audit

Checked 18 JSON files: 17 under `configs/historical/` plus the history ledger.
Total: 1,861,279 bytes (about 1.78 MiB). No run log or checkpoint payload is embedded.
Large files keep both raw base64 bytes and parsed rows so byte-exact export works.

| File under `configs/historical/` unless noted | Bytes |
|---|---:|
| `cce_add_fixed_eval.json` | 31,104 |
| `coverage.json` | 31,478 |
| `decay_legacy.json` | 454,905 |
| `dense_fixed_eval.json` | 65,548 |
| `episode_carry_pilot.json` | 116,260 |
| `fixed_eval_8x8.json` | 31,319 |
| `greedy_coverage.json` | 53,047 |
| `paper_reward.json` | 33,053 |
| `qdiag_baselines_8x8.json` | 15,822 |
| `qdiag_smoke.json` | 4,265 |
| `rarity_baselines.json` | 75,827 |
| `rarity_cce.json` | 104,186 |
| `rarity_cce_add_8x8.json` | 38,813 |
| `runnable/dense_fixed_baselines.json` | 50,619 |
| `runnable/sparse_fixed_baselines.json` | 50,479 |
| `score_on_insert.json` | 77,070 |
| `temporal_replay_v2.json` | 233,079 |
| `studies/jaxnav-migration/history-ledger.json` (repo-relative) | 394,405 |

The scan inspected parsed strings plus 168 decoded payloads, including nested
base64 config lines. It found no private-key markers, known token prefixes,
Bearer credentials, URL passwords, email addresses, nonempty credential fields,
or user-name references outside the expected source/run paths.
Expected identifiers remain: absolute workspace/interpreter/run paths, Git hashes,
branch names, scheduler IDs, task IDs, seeds, device labels, and package versions.
This is a scoped pattern/content audit of these manifests, not a general secret
scan of every external source bundle or an assurance about arbitrary secret forms.
