# Procedural DoorKey 12x12 benchmark

This is the hard DoorKey transfer.

- Fresh procedural 12x12 map each episode.
- Official MiniGrid sparse, time-scaled goal reward.
- Official 1,440-step cap.
- Symbolic 7x7 partial observation for the Q-network.
- Hidden simulator state is stored only for exact CCE branches.
- Uniform, ranked-TD PER, and CCE-only fading to ranked TD.
- Five paired train seeds and one shared bank of 1,000 evaluation maps.
- One million environment steps per run.

CCE scores every inserted transition. Its replay mix fades linearly to pure
ranked TD from 500,000 to 750,000 environment steps.

Rebuild the manifest with:

```bash
python -m training.generate_doorkey_procedural_benchmark
```

Run `sbatch --test-only` before each submission. Smoke all three arms first.
