# Episode-carry pilot: launch and protocol

## Integration into the earlier working branch

The prepared source from `fix/jaxnav-episode-carry` (base `0231116`, JaxNav import
`0165ecb`) is integrated onto `research/cce-buffer-diagnosis` at `bb0d081`.
Shared buffers use a three-way merge retaining the target's `DrawLogMixin` and
all supply/draw hooks alongside the JaxNav score coverage and priority features.
The manifest test reads a small protocol-only fixture, not a live `work/` tree.
Launchers accept `CCE_PYTHON`, defaulting to the user's counterfactual conda env.
Suite preparation still needs historical manifests/logs and sibling source trees;
those experiment inputs are not bundled. Existing frozen runtimes and jobs are
unchanged. The launch status below is the original recorded status, not a fresh
scheduler check.

Branch `fix/jaxnav-episode-carry`, based on remote master0231116. Prior JaxNav source0165ecb imported for a standalone patch; production sources are frozen separately. No source or jobs in the old experiment worktrees changed. JaxNav only: the identified FrozenLake/DoorKey bugs are not fixed by this patch.

## Change and checks

Plain/PER and CCE JaxNav now initialize a five-field collection carry once after reset and return that entire carry between scans. It contains simulator state, observation, pending reset, cumulative return and episode length. Their `learn()` callers initialize the new carry. The temporal-hold subclass uses the same interface. CCE still emits pre-action simulator state at output index8 and retains the existing score-on-insert path.

The old collector failed the newly written split-collection test (episode return lost at a block boundary). After the change, all10 CPU tests pass: plain and CCE endings before/on/after the split, episodes across several blocks, no CCE pre-state already ended, and64steps versus32+32 through the temporal collector. CCE observation reconstruction is checked at1e-6 tolerance because separate JAX compilations differ by roughly4e-8; split-versus-unbroken trajectories are checked exactly. A separate manifest test passed for all20 paired configs.

Frozen old source is byte-identical to the previous temporal runtime. Fixed source differs only in the two JaxNav collector files, plus the temporal collector script. Script/source/package hashes are checked at startup. Artifacts live under `work/episode_carry_pilot_20260930/`; naming reflects when preparation began, not a claim about the current calendar date.

## Paired pilot

- Five arms: Uniform, PER, CCE multiply(1,1), additive50/50, CCE-only(additive mu1).
- Seeds0,1; old/fixed per seed =20 jobs.
- Dense8x8, fill.3, radius.8, goal+4/crash−4, inherited shaping, max200steps, exploration hold20.
- CCE horizon60, one greedy rollout/action, target bootstrap, Wasserstein weighted mean, rank normalization and full score-on-insert unchanged.
- Equal2000 blocks×256envs×32steps =16,384,000 transitions and1,024,000 Q-update attempts per job. Epsilon still decays by completed-episode count; its clock correction is part of the intervention.
- Same1000-map fixed evaluation bank(seed20260923), every50blocks. Primary endpoint is the mean of the final four scheduled checks, reported per seed then averaged. No best-checkpoint selection and no significance claim from two seeds.
- Old/fixed variants within each replay arm use the same GPU type: baselinesT4, CCE H100. Cross-arm wall-clock differences are hardware-confounded; report within-arm fix cost separately.

This is a **fresh matched-step pilot**, not a bit-identical rerun of the previous episode-budget experiment. Both variants use the same explicit runner. Evaluation/checkpoint cadence is step-based; episode-based early stopping and large raw-buffer dumps are not used. Diagnostic drift probes are not invoked in this runner (the inherited source config retains their historical values); Q/loss, replay concentration, exploration and full-coverage logs remain on. This omission is equal in both variants and is not removal of CCE replay scoring. The old drift probe's comments claim no training perturbation, but it consumes training JAX and NumPy RNG: do not treat it as a passive probe in future reproduction claims. Compare this pilot's old/fixed pairs, not differences from older curves as a pure fix effect.

## Safety gates

Each variant/arm first gets a real8-block GPU smoke test (10smokes total), checking actual learning, finite loss, fixed-bank RNG independence, checkpoint round trip, full episode lengths in fixed runs and100%CCE score coverage. Full jobs depend on successful completion of **both** smoke arrays and revalidate all10 smoke reports at startup. Failure blocks full training. No automatic requeue and no reuse of run directories.

Smoke arrays:279440(T4),279441(H100). Full array IDs and monitor PIDs are recorded in `work/episode_carry_pilot_20260930/train-jobs.json`. Each sbatch was dry-run first and immediately given a persistent state/error monitor. Monitor logs are local status logs, not a promise of push notifications.

Full arrays:279445(8Uniform/PER tasks,T4),279446(12CCE tasks,H100), each with `afterok:279440:279441`. At the last check, all four T4 smoke tasks passed; H100 smoke tasks pending for scheduler priority. Fixed Uniform logged a197-step episode, versus old Uniform's maximum32. All completed smoke evaluation bank hashes match the historical fixed bank. Full training has not started and no learning improvement is measured yet.

`terminal_start_transitions` is directly measured only for CCE, whose outputs include simulator states. The baseline runner leaves that field at0 without measuring it; do not interpret baseline0 as proof of no ghosts. The old/fixed split tests, full episode lengths and CCE state checks supply the relevant correctness evidence.

GPU smoke results were pending when this launch note was first written. Do not call the performance effect measured until real paired training logs exist.
