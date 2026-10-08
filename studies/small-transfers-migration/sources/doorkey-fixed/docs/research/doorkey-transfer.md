# DoorKey fixed-recipe transfer

Custom full-state 6x6 DoorKey, not stock MiniGrid. State ID is one-hot encoded.
Fixed start (1,1), east; no lava, slip, random starts or shaping. Seven valid
actions, including blocked-action, drop and done no-ops. Goal pays +1 once.

Training and evaluation stop an episode at 50 actions. This is truncation:
the replay next state is the real state and remains bootstrappable. Only goal
blocks value. Q estimates uncapped discounted return; success measures goal
within 50 actions. CCE never resets or applies that collection cutoff. Horizon
60 includes the first action; live tail uses gamma^60 times target max Q.

One observation index identifies the full task state. Q has three 128-wide ReLU
layers, seven outputs. The fixed source recipe is preserved in source_recipe.json.
Old navigation and inactive refresh keys in that file do not enable runtime code.
The pinned shared buffer files belong to main and retain draw-log hooks.

One chunk produces exactly 256*32 real action transitions. Carry holds state,
observation index, pending reset, full episode return and length. A pending
reset occurs before the next action; it never contributes a replay row.
Target copies follow attempts 0,125,250,... after their updates. Score each new
CCE row before sampling, in chunks of 1280; retain source training-key splits.
No old-score refresh. All seven alternatives are evaluated, with the taken
action excluded from the weighted mean. With deterministic finite states,
network values may be computed as a complete table for a fixed rollout call;
each inserted transition still receives all seven horizon-60 branches.

Run from a frozen checkout with its src first in PYTHONPATH:

```bash
python analysis/claim2/train_doorkey_transfer.py --arm uniform --seed 0 --mode smoke --output /absolute/new-run --require-gpu
```

Arms: uniform, per, cce_mul, cce_add, cce_only. Smoke: seed 0, 8 blocks,
65536 transitions, 4096 Q updates, eval at blocks 4 and 8. Train: seeds 0,1,
2000 blocks, 16384000 transitions, 1024000 updates, eval every 50 blocks.
Main's central launcher freezes sources, verifies hashes and versions, and
enforces all ten task/arm smoke passes before the twenty full runs. This runner
does not submit jobs. --allow-cpu permits local diagnostics but never emits a
passing GPU gate. Output paths must be new. No resume, early stop or best model.

The 1000 independent eval keys use seed 20260923. The environment is deterministic:
all 1000 episodes repeat one greedy path. No confidence bands or unique-map claim.
Evaluation and replay diagnostics must leave training JAX and NumPy RNG intact.

Logs: provenance.json, config.json, qdiag.jsonl (each block), eval.jsonl and
replay.jsonl (each check), progress.json, last.pkl and scheduled train checkpoints.
Checkpoints contain weights, target, optimizer, carry and clocks for checking;
they omit replay and are explicitly not resumable. Reload and eval parity are
checked before complete.json. CPU checks do not count as GPU smoke passes.

Primary train endpoint is the mean of the final four scheduled eval checks per
seed. Plots keep thin seed curves and label mean/median, with no filled tails.
T4 smoke and H100 full CCE time must not become cross-hardware speed claims.

## CPU verification, 2026-10-01

66 checks passed in 63.51 seconds: 36 old task-rule tests, 24 transfer tests,
and six shared-contract checks owned by main. Transfer checks include one real
8192-transition/512-update block for each of the five arms. Random-policy
rollouts matched a stepwise-network reference for three seeds at rtol=1e-5,
atol=1e-6; greedy action choices matched for all 154 enumerated states.

The actual smoke CLI also completed on CPU for cce_mul seed 0: 65536 transitions,
4096 real updates, 100% score coverage, 50-action full episode lengths,
checkpoint reload parity, finite diagnostics and unchanged eval/diagnostic RNG.
Runtime was 47.91 seconds, first block 9.73 seconds, remaining blocks mean 5.36
seconds. These are local CPU measurements, not GPU or full-buffer cost estimates.
Local evidence: /tmp/doorkey-transfer-cpu-cli-20261001/complete.json. It records
backend=cpu and passed=false by design, so it cannot unlock the GPU gate.

Plot real/live logs with:

```bash
python analysis/claim2/plot_doorkey_transfer.py --runs /absolute/run-one /absolute/run-two --output /absolute/figures
```
