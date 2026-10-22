# CartPole PER positive control

This study checks whether two proportional replay rules beat uniform replay on
the published ReaPER CartPole-v1 protocol.

| Arm | Sampling score | Importance weight |
|---|---|---|
| `uniform` | constant | `1` |
| `paper_per` | `(|TD| + 1e-6)^0.6` | `(N P(i))^-beta`, max-normalized; beta `0.4 -> 1.0` |
| `repo_per` | `(|TD| + 0.01)^0.25` | `1 / (N P(i))` |

All arms use the same DDQN, Huber loss, optimizer, exploration schedule,
50,000-step cap, fixed evaluation seed rule, and 20 seeds (`0..19`). Task IDs
are `3 * seed + arm_index` in the table order above.

The paper does not publish its seed values. The declared consecutive seed list
is therefore a matched reproduction choice, not a recovered paper artifact.

Smoke runs only prove that code paths execute. They are not study evidence.
The full study is complete only when all 60 tasks have valid `attempt.json`,
`summary.json`, and evaluation traces.
