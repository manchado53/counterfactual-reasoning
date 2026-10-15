# DoorKey full-state 16x16 difficulty gate

This is the next rung after the full-state 12x12 DoorKey study.

Only the task size changes:

- grid: `12x12 -> 16x16`;
- episode cap: `1,440 -> 2,560` steps;
- output and protocol names identify the new study.

Uniform, PER, and CCE keep the same network, optimizer, replay settings,
exploration schedule, sparse reward, five training seeds, and fixed 1,000-map
evaluation design as the 12x12 study.

Run `smoke.sbatch` first. Then run `gate.sbatch` for seed 0 of all three arms.
The full 15-task recipe is kept ready for expansion after the difficulty gate.

If H100 jobs cannot start, use `smoke-t4.sbatch` and `gate-t4.sbatch`. These
run the same three tasks and recipe on the `batch` T4 partition.
