# JaxNav episode carry fix and pilot

Approved by the user on 2026-09-30: fix episode state lost at collection boundaries and relaunch a subset across the previous arms. Preserve existing jobs and outputs.

Keep simulator state, observation, pending reset, accumulated reward, and episode length across collection calls in plain and CCE JaxNav collectors and the temporal-exploration collector. Keep replay insertion, CCE scoring, rewards, action keys, timeout semantics, and evaluation unchanged. Verify split versus unbroken collection with fixed parameters and keys, boundary endings, full return/length, and CCE pre-step states. Run real GPU training smoke tests before production.

Pilot: dense 8x8/fill0.3/radius0.8, exploration hold20; Uniform, PER, CCE multiply, CCE addition 50/50, and CCE-only. Seeds0 and1, each old versus fixed (20 runs). Freeze the previous runtime and resolved configs; use separate immutable old/fixed sources, unique outputs, hashes and provenance. Equal 2000 collection blocks (16,384,000 steps), unchanged episode-based epsilon rule, evaluate every50 blocks on the same1000 maps. The epsilon clock is part of the fix's total effect, not an independent intervention. Compare on environment-step axes, not the broken old episode axis. Check finite losses and every CCE arrival scored. Dry-run submissions, then start monitors after each array submission. No old job cancellations and no paper-scale claim from two seeds.

This task fixes JaxNav only. FrozenLake/DoorKey follow-ups stay separate. The repository worktree starts at current origin/master and imports the prior JaxNav source as explicit provenance, without changing its running worktree.
