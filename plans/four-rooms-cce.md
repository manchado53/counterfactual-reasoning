# Sparse Four Rooms CCE transfer

Use the latest origin/master (the user approved this base because origin/main does not exist) in an isolated task branch. Add the classic Four Rooms map as a shared environment, explicitly label the +1 terminal reward and 200-step cap as our Sparse Four Rooms variant. Reuse shared DQN updates, CCE scoring and replay priorities. Freeze equal training settings and a separate fixed evaluation bank for Uniform, rank PER and full CCE, with five matched seeds.

Verify the map, stochastic transitions, termination, evaluation identity, replay-only arm differences, and CPU/GPU startup before submitting at most eight concurrent jobs. Save raw logs and provenance, report only observed checkpoints, and produce success/length curves, paired bootstrap comparisons, map and reproduction recipe. Push small commits and open a PR without merging. Monitor only failures, milestones and completion.
