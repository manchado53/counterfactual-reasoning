# Record + commit the JaxNav robotics-transfer experiment

We just got a real result (CCE beats PER on the "holes" map, loses on the easy map) and the
user wants it captured so it's reproducible and won't get lost. Two things are true about this
repo that shape how: `**/runs` and `**/experiments/` are gitignored (raw checkpoints/manifests
live only on Rosie disk, which the lab notebook already flags as a place results have
vanished from before), and the established pattern for "durable, look-back-able" results is a
small committed cache next to the code (mirrors how `paper/repro/` freezes the FrozenLake
results) plus figures under `docs/figures/real/claim2/<env>/`.

So: commit the actual code (new env adapter, trainers, buffer fix, Claim-2 wiring, slurm
scripts) so the experiment is re-runnable; commit the figures and a small results-data cache
(the curve JSONs + manifests, a few hundred KB total) so the numbers survive even if the Rosie
run directories are ever cleaned up or lost; and write it into `lab-notebook.md` the way every
other experiment in this project is recorded, with exact commands and job IDs so it can be
repeated. Raw checkpoints and per-episode logs stay on Rosie only (that's normal — they're
large and regenerable from the recorded config + seed).

This is a local commit on the current worktree branch (`worktree-research+cce-robotics-transfer`,
off `research/cce-robotics-transfer`) — not main. Whether to also push the branch to `origin` is
a separate, explicit question for the user, since pushing is a shared/visible action.
