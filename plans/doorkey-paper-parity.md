# DoorKey paper-parity experiment

Add a separate DoorKey 12x12 experiment that matches the VLM-RB MiniGrid setup instead of replacing the existing partial-view stress test. The policy will receive the full symbolic grid, use only the five useful actions, and use a spatial encoder plus the paper's DQN training settings. Uniform, PER, and CCE will share the same maps, seeds, learner, reward, and fixed evaluation bank.

First verify environment and observation parity in tests. Then run GPU smoke tests and one full Uniform seed as a learning gate. Launch the five-seed Uniform/PER/CCE comparison only if that gate learns. Keep the old partial-view experiment unchanged and record both protocols separately.
