# DoorKey transfer of the winning JaxNav protocol

Run Uniform, PER, and the winning JaxNav CCE-only-to-ranked-TD schedule on
DoorKey. Preserve every environment-independent learner, replay, optimizer,
exploration, scoring, and schedule value from the JaxNav result. Change only
the simulator-bound fields needed for DoorKey.

Compare the clean 6x6 map with the matched 6x6 one-lava map. Use seeds 0-4,
the same fixed evaluation identity across replay arms, complete buffer scoring,
and the existing diagnostic logs. Add parity tests that fail if any copied
JaxNav field drifts. Run CPU correctness tests, GPU smoke tasks, then submit the
30-run Slurm array only after a scheduler dry run. Save the exact recipe and
submission script in the experiment bundle.
