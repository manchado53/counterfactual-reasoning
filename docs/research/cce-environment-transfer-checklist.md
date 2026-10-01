# CCE transfer after the JaxNav audit

2026-10-01. Read-only review of earlier results and current code, not authorization to launch another environment. Historical numbers in the notebook are not revalidated here. Do not describe every recent JaxNav change as a proven cause of CCE gains.

## Best small revisit: stochastic FrozenLake

The lab notebook records a weak/no CCE advantage in slippery FrozenLake, a previously stronger deterministic result, a marginal SMAX result, and unresolved game-specific issues in Connect Four. FrozenLake is the simplest revisit: four actions, small state set, explicit transition table `env.P`, and controllable slip. Use its exact model to check the CCE estimates rather than assuming a learning curve explains their quality.

There is already an unlaunched/independently maintained graded-slip plan in `.claude/worktrees/graded-slip-frozenlake/plans/graded-slip-frozenlake.md`. Its wording predicts a monotone result too strongly. Treat the slip curve as a falsifiable test, not an expected win that must appear. That plan omits several newer safeguards below.

## What transfers, and what does not

| Item | What to preserve or test | FrozenLake-specific warning |
|---|---|---|
| Episode carry | Full state, pending reset, reward sum and length survive collection blocks; reset once | Batched FrozenLake has the same confirmed bug. Fix and test it before rerunning; sequential code is a useful reference |
| Environment contract | Real state, observation, legal actions, reward recipient, true ending vs cutoff | Integer lake state can reconstruct position, but not an elapsed episode clock. Do not assume it contains every field |
| CCE state | Score the **pre-action** state and recorded action | Test saved state/action against the actual transition, including stochastic outcomes |
| Coverage | Every new replay entry scored, including circular-buffer overwrites | Inspected main-worktree FrozenLake still samples only part of the buffer periodically. Adding `score_on_insert` to JSON alone does not implement this |
| Score freshness | Log age in Q updates and actual replay draws | Scored-once coverage is not evidence that scores remain current as the policy learns |
| Stochasticity | Separate differences caused by actions from random environment noise | One greedy rollout is deterministic in JaxNav, **not** on a slippery lake. Check repeated estimates and convergence in rollout count |
| Score distance | Explicitly choose and verify the consequence metric | Total variation of point masses is coarse. Wasserstein keeps return-distance information; it does not by itself remove noisy estimates |
| Horizon | Verify that the look-ahead can see the relevant outcome; define tail value treatment | Do not copy h=60 or h=200 mechanically. Inspected old lake rollout truncates at its horizon without JaxNav's optional target bootstrap |
| Action weights | Lock taken-action exclusion, aggregation and Q temperature | Old lake scorer does not pass JaxNav's action-probability weights. Same name `weighted_mean` does not establish equal weighting behavior |
| Priority mixing | Preserve exact additive/multiplicative formula and normalization | CCE-only is additive mu=1; 50/50 is mu=.5; multiply mu_c=mu_delta=1. Verify resolved behavior, not just labels |
| Replay correction | Check probability exponent and importance weights in the actual buffer | Our `beta` names the sampling exponent; do not confuse it with another library's importance-correction beta. Log effective sample size and realized draws |
| Training intensity | Match collected steps, actual Q updates, batch size and buffer turnover | JaxNav batch128/update-every16 and lake batch32/update-every4 both imply nominal replay ratio8, but network/target clock still differ |
| Exploration | Keep the same exploration rule across replay arms | Do **not** port hold20 by default. A lake move changes a tile at once; 20 repeated moves can hit a wall or hole |
| Evaluation | Separate RNG from training; paired evaluation conditions; held-out final test | A fixed lake map still has random slip. Pair the slip-key bank, not merely the board. Old evaluate consumes training RNG |
| Budget and stopping | Equal step/update budgets; explicit handling of solved-run stopping | Do not compare stopped tails with continuing runs as if those flat lines were new evaluations |
| Baselines | Uniform/PER receive the same bug fixes, model and evaluation | A stronger CCE must beat equally repaired baselines, not historical buggy runs |
| Reporting | Raw seeds, mean/median, learning speed, final window, uncertainty and wall time | Tiny seed counts are pilots; best checkpoint picked on validation is not an unbiased held-out result |
| Reproducibility | Freeze source, vendor code, configs, package versions and evaluation banks | Reject unknown/unused config flags. Keep one arm/seed record; never mix hardware copies or forward-fill unfinished runs |

## Branch-specific traps actually found

1. Main-worktree lake (`research/cce-buffer-diagnosis`) uses `slip_probability`; graded-slip worktree uses `slip_prob`. Both define total sideways probability p with `[p/2, 1-p, p/2]`. The legacy fully slippery lake is **p=2/3**, not p=1/3. Test the transition table AND empirical step frequencies. Passing the wrong key can silently leave the old binary behavior if defaults absorb unknown keys.
2. The main lake collector loses episode carry; its evaluation uses a 200-step scan, while its training episode loop does not enforce the same cap. Decide whether the task is infinite-horizon with evaluation cutoff or a finite-horizon task; do not silently change objectives while fixing bookkeeping.
3. The main lake CCE scorer still draws a uniform subset on scoring intervals; the JaxNav scorer has `_score_indices` and score-on-insert. Port implementation and tests, not just its config.
4. Main lake Q targets use max-next-Q in the inspected version; current JaxNav configs use Double DQN. Changing the learning rule is a separate ablation, applied to all arms, not a hidden CCE-only change.
5. Old notebook pre-flight lines mandate total variation and full-episode horizon. Those are historical choices, not universal requirements: record the intended scientific variant instead of blindly following them.
6. Connect Four/Chess need seat-correct rewards, legal-action masks and a clear opponent policy. Their fresh-games-per-chunk issue is different from the lake/JaxNav ghost episode. SMAX needs joint-action/state and team-reward checks; neither is a simple copy of single-agent navigation.
7. Diagnostic code can change training RNG. JaxNav `_score_states` splits `self._key`, and drift-probe initialization samples using the replay buffer's NumPy RNG. Test logging on/off equivalence or give probes separate random streams before calling them passive instrumentation.

## Small next design, not launched

1. Repair and test the lake collector and independent evaluation RNG. Validate terminal rewards and slip semantics against `env.P`.
2. On fixed state/policy snapshots, compare estimated action-return scores with exact finite-horizon return distributions (or exact action values when the chosen score compares means). A Q*-gap oracle alone does not validate a distribution-distance score.
3. Reproduce the old recipe with correctness fixes and repaired Uniform/PER controls. Keep reward, exploration and learning rule unchanged.
4. Add full scoring coverage as its own change. Then separately test graded return distance, look-ahead/tail treatment, and enough stochastic rollouts. Do not stack every improvement into one unexplained win.
5. Start with no-slip and legacy-full-slip anchors, five paired training seeds, the five explicit replay arms. Add middle slip levels only after endpoint checks pass. Report actual success and both step- and wall-clock cost, even if CCE still ties or loses.

Current JaxNav pilot is separate: two seeds, five arms, old/fixed collector, hold20 dense setup, equal16,384,000-step budget. Its outcome is not known yet.

## Source pointers inspected

- Main `lab-notebook.md`: status, 2026-07-05 caveat about weak three-seed reruns, 2026-06-14 noise/metric probes.
- Main `src/counterfactual_rl/agents/frozen_lake/{config,dqn,dqn_vectorized,consequence_dqn}.py` and `envs/frozen_lake.py`.
- Robotics-transfer worktree `src/counterfactual_rl/agents/jax_nav/{consequence_dqn,consequence_dqn_vectorized}.py` and `agents/shared/consequence_buffers.py`.
- Graded-slip worktree `plans/graded-slip-frozenlake.md` and `src/counterfactual_rl/envs/frozen_lake.py`.
- The separate episode-boundary audit records the runtime reproductions and environment comparison; this checklist does not claim to have fixed those other environments.
