# DoorKey 12x12 benchmark

Replace the recent fixed 6x6 mechanism test with a researcher-style 12x12 DoorKey study. Each training episode will draw a fresh procedural wall split, door, key, agent position, and direction. The task will use deterministic MiniGrid actions, the official sparse time-scaled goal reward, the official 1,440-step limit, and a fixed held-out bank of procedural maps for paired evaluation.

Keep the winning JaxNav replay design unchanged: Uniform, ranked-TD PER, and CCE-only replay that fades to ranked TD. The learner will consume MiniGrid's symbolic 7x7 agent view. Replay rows will also retain the hidden simulator state and map description so CCE can branch from the exact transition without giving that hidden state to the policy.

Build parity tests against official MiniGrid generation, observation, reward, and transition behavior. Add matched-seed smoke tests for all three arms, verify CCE scores every inserted transition, verify evaluation does not touch training randomness, and measure GPU speed before launching five paired seeds per arm.

Use a one-million-environment-step budget and researcher-style reporting: success rate over a shared held-out map bank, mean and median seed curves, all seed traces, and uncertainty across seeds. Do not launch the full sweep until the official-environment parity gate and GPU smoke gate pass.
