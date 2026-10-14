# Procedural DoorKey 12x12 at 10M steps

This repeats `procedural-12x12` from fresh initialization with one intentional
change: each run trains for 10,000,000 environment steps instead of 1,000,000.

Everything else stays fixed: official procedural maps, sparse reward, 1,440-step
cap, symbolic 7x7 partial observation, learner, replay settings, decay from 500k
to 750k, five paired seeds, and the shared 1,000-map evaluation bank.

Run the smoke array first, then the full 15-task array after `sbatch --test-only`.
