# Small transfer ports — October 8, 2026

These are real new-path learners. CPU checks passed. No research run, GPU gate,
speedup, or full-study result is claimed.

| Protocol | Code | Evidence |
| --- | --- | --- |
| DoorKey fixed JaxNav transfer | Runnable, five arms, seeds 0/1 | Five reduced train/reload checks; exact old/new collection, TD updates, branch returns, initial eval |
| ConnectFour fixed JaxNav transfer | Runnable, five arms, seeds 0/1 | Five exact old/new reduced training checks, including params, optimizer, carry, replay scores, RNG and eval |
| Old DoorKey 6x6 and lava | Runnable, five arms; source Claim-1/2 recipes | Ten exact old/new reduced learn checks, params, targets, random keys, returns, lengths, eval |
| Old DoorKey sequential | Runnable; large-lava CLI instances explicit | Exact old/new reduced learning on both large lava layouts and all five arms; see extension |
| Old ConnectFour random/rule/MCTS | Runnable separate historical protocol | Source kernels unchanged apart from paths; full arm/foe checks recorded in extension evidence |
| DoorKey larger layouts/headroom | Runnable; exact source headroom recipes | Both large plain layouts with uniform/PER, both large lava layouts with five arms; see extension |
| Graded FrozenLake | Seen in sibling paper caches/manifests | Left to FrozenLake owner; no code touched |

## Source and recovery

`source-manifest.json` records source roots, Git heads, full dirty status, and
SHA-256 for the first 122 captured source files. `extra-source-manifest.json`
pins two more old ConnectFour dependencies. `sources/` has their original bytes.
The hash test reads those bytes; it does not read a private worktree at runtime.

* Root: `01a27a8a99e66dfccc0a441ea53d916ff60fca6b`.
* Fixed DoorKey and ConnectFour worktrees: base
  `0231116bd4e97cfc06d01c95ee7247763ff6bb0d`, with untracked transfer code and dirty
  shared replay sources. The base commit alone cannot recover those edits.
* Old DoorKey/lava: andon worktree at
  `3c1056c5ea70629530a35c199bf70a5bb42760f5`.

Both fixed ports use their own captured replay implementation under
`_source_shared/`. Old DoorKey uses its older replay implementation under
`legacy/shared/`. In particular, its old CCE is not today's rank CCE.
DoorKey's JSON source recipe is also embedded as `source_recipe.py`; the generic
Python source bundle therefore holds its training-control values without a
runtime JSON/package-data dependency. The JSON remains for source recovery.

`artifact-inventory.json` lists 4,543 retained files, about 3.09 GB, with paths,
sizes and timestamps. Originals remain in place. The bounded text snapshot
copied 8,388,365 bytes under `retained/`. It is not a full checkpoint backup.
Large binaries and over-cap logs have explicit retained-in-place entries.
Text copies have both original and snapshot hashes; a final newline can differ.
Recovery of uncopied checkpoints still needs the named local disk paths.
No scheduler or remote run-state audit was done here.

## Rules kept

Fixed DoorKey keeps seven actions, full-state one-hot input, fixed east start,
goal +1, and timeout at 50 actions. Timeout keeps the real next state and its
bootstrap value. Goal is terminal. CCE branches take 60 actions without applying
the collection cutoff and bootstrap live tails. Scores keep float64 differences
and ordered accumulation. Train-key splits are retained even for deterministic
branches. Target copies occur after attempts 0,125,250,... .

Fixed ConnectFour keeps PGX learner-turn transitions, legal masks, alternating
roles, opening move outside replay, and the deterministic win/block/center foe.
Tie order stays `[3,2,4,1,5,0,6]`. No illegal action enters scoring or bootstraps.
Terminal transitions absorb. Fixed evaluation repeats two role cases 500 times
each; it does not provide 1,000 independent cases.

Both fixed full recipes keep 2,000 blocks, 256 environments, 32 collection
steps, evaluation every 50 blocks, and no early stop. Smoke changes only block
count, eval cadence, environment count and replay capacity: 2 blocks, 4 envs,
32 steps, capacity 256. Network, batch size, update cadence, score horizon,
opponent, discount and evaluation bank stay at source values. Thus each smoke
has 256 transitions and 16 real updates. It is not the old 65,536-step GPU gate.

Old DoorKey keeps its episode budget and source early-stop checks. The vector
collector resets episode return/length at each chunk; it is not silently fixed.
Its eval cap stays 200 and eval consumes the training random key. The identity
hook reports this and does not invent a fixed bank. Legacy smoke keeps task,
rewards, start rules and early stopping but reduces network, rollout and replay
sizes; the old/new comparison applies the exact same reductions to both sides.

## First-slice checks and scope

The follow-up ports, wider seed scope and later checks are described in
`EXTENSION.md` and `extension-evidence.json`. The counts below are the first
slice's checks; the extended recipe set now holds 353 explicit rows.

All tests used `PYTHONNOUSERSITE=1`, CPU affinity restricted to four cores, and a
180-second process cap. DoorKey used `/tmp/cce-clean-install-uudpga/venv/bin/python`.
ConnectFour used `/home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python`.
The clean venv lacks PGX. No dependencies were changed.

Both runtimes: JAX/JAXlib 0.9.1, Flax 0.10.4, Optax 0.2.5, NumPy 2.4.0,
SciPy 1.16.3. Historical conda has PGX 2.6.0. It emits a CUDA plugin warning
on this CPU host, but the requested CPU backend runs and all checks pass.

* Five fixed DoorKey arm checks passed (with one old lava uniform check:
  six checks in 49.11 seconds).
* Fixed ConnectFour uniform passed in 25.88 seconds. The four other arms passed
  together in 100.36 seconds.
* Eight old DoorKey/lava PER/CCE checks passed in 50.01 seconds; the remaining
  no-lava uniform check passed in 8.99 seconds. Lava uniform is in the first group.
* Two hash/control checks passed in 0.91 seconds. They validate all 126 recipe
  rows, source hashes, unchanged transition kernels, and fixed smoke controls.
* Three fresh shared-runner CCE-multiplicative smokes completed. See
  `validation-evidence.json` for configs, paths and source hashes.

Exact means `assert_array_equal`, not a loose loss tolerance. DoorKey fixed does
not claim an exact old/new whole training trace: its source hard-codes an
8,192-transition block. The reduced check compares its untouched collector,
actual TD update and horizon-60 return functions, then runs/reloads the new
learner. ConnectFour and legacy DoorKey compare complete reduced learning paths.
Only seed 0 was trained in these checks. Seed 1 recipes validate but were not run.
Legacy stochastic Claim-1 rows validate but were not trained here.

Original DoorKey plot code also regenerated the retained October 1 CPU smoke:
`/tmp/cce-doorkey-source-replot-8gxhq5wm/doorkey_curves.png`.
Input: `/tmp/doorkey-transfer-cpu-cli-20261001`.
That source record says CPU correctness passed and GPU gate `passed=false`.
The plot is a smoke plot, not a full-study result. Old full-study figures and
ConnectFour historical outcomes were not regenerated here.

## Hooks and rerun

Both `build.py` modules provide lazy `validate_config`, `build_agent`,
`smoke_overrides`, `simulator_sources`, and `evaluation_axis`. Agents preserve
the resolved config exactly and support `learn(verbose=False)` and `evaluate`.
Fixed agents also expose fixed evaluation identity. The shared runner owns
fresh output paths through `CCE_RUN_DIR`, attempt/source records and backend.
DoorKey legacy writes its original metrics, checkpoint and timing files there.

```bash
python studies/small-transfers-migration/check.py \
  --python /tmp/cce-clean-install-uudpga/venv/bin/python \
  --recipe configs/experiment/doorkey-fixed-transfer.json --task 4

python studies/small-transfers-migration/check.py \
  --python /home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python \
  --test 'tests/test_connect_four_transfer_migration.py::test_fixed_source_and_smoke[cce_mul]'
```

The helper always makes a fresh `/tmp/cce-small-transfer-check-*` folder and
caps each child at 180 seconds. Run a single test node per child for a firm cap.
Capture scripts are one-shot source bookkeeping tools, not runtime wrappers.
