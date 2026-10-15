# DoorKey full-state 16x16 experiment

Extend the successful full-state DoorKey 12x12 study to 16x16 on the same experiment branch. Keep the learner, replay rules, reward, seeds, and fixed evaluation design unchanged. Change only the map size, the MiniGrid-style episode cap, and output names.

First verify the 16x16 environment and GPU path. Then launch one 1M-step seed for Uniform, PER, and CCE as a difficulty gate. Use the gate to decide whether the full five-seed study has useful headroom.

The run must use the migrated shared TD, PER, and CCE logic. Do not launch from stale environment-local replay code.
