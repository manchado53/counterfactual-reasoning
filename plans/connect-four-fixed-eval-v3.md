# Fixed Connect Four tests

User approved this design and launch on October 14, 2026.
Use 500 empty-board cases and 500 paired-opening cases.
Freeze 50 four-ply boards, roles, minimax depths 1–5, and per-case foe seeds.
Use seed plus absolute board ply for tie choices, never checkpoint or training RNG.
Test the same saved agent twice and require identical full paths and scores.
Report the two suites separately, including draws and actual unique path counts.

Retain v2 training settings and win/block/center training foe. Only test foes change.
Keep Uniform, PER, ranked-decay CCE, five seeds, identical budgets and checkpoints.
Use a new worktree/recipe so active v2 and DoorKey jobs keep their frozen source.
This branch starts at freshly fetched master 5f2d313 and includes v2 commit 391399e.
Run CPU regression tests and all three GPU smokes before the full 15-task array.
Test Slurm submissions first, keep total active experiment GPUs at most eight,
start a persistent read-only monitor, push the branch, and open a PR.
