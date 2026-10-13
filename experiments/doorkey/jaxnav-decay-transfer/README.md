# DoorKey transfer of the winning JaxNav schedule

This experiment copies the environment-independent configuration from task 33
of `experiments/jaxnav/ranked-td-decay/training.json`: CCE-only replay with the
fixed collector, followed by a linear transition to ranked TD replay from
100,000 through 150,000 completed episodes.

The study uses Uniform, PER, and scheduled CCE on two deterministic DoorKey
layouts: clean 6x6 and the matched 6x6 map with one fatal lava tile. Seeds 0-4
are paired across every arm and layout. The only intentional differences from
the JaxNav source are simulator-bound fields: environment, state/action space,
layout, and the DoorKey episode cap.

`python -m training.generate_doorkey_decay_transfer` rebuilds the checked-in
recipe. Tests compare every copied field against the JaxNav source row and
reject drift.

## Submit

Run a scheduler dry run first:

```bash
sbatch --test-only experiments/doorkey/jaxnav-decay-transfer/train.sbatch
```

Then submit the 30-task array:

```bash
sbatch experiments/doorkey/jaxnav-decay-transfer/train.sbatch
```

Outputs are written under `experiments/doorkey/jaxnav-decay-transfer/runs/` and
are ignored by Git.
