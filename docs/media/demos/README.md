# README demo provenance

These files are generated from saved checkpoints and raw evaluation logs.
They are not hand-drawn examples.

## Comparison rules

```text
FrozenLake
  same training seed -> same deterministic map -> final checkpoint

JaxNav curve
  same seeds 0,1,2 -> same 1,000 maps -> 5k bins -> mean +/- SEM

JaxNav rollout
  same training seed -> checkpoint nearest 125k -> fixed map index 0
```

The JaxNav map is not searched for a favorable CCE outcome. A single rollout
is an illustration. The learning curve is the comparison.

## Source runs

| Environment | Arm | Runs | Checkpoint rule |
|---|---|---|---|
| FrozenLake | Uniform | 257500 | final |
| FrozenLake | PER | 257525 | final |
| FrozenLake | CCE multiplicative | 257600 | final |
| JaxNav | Uniform | 277734–277736 | nearest 125k for video |
| JaxNav | PER | 277751–277753 | nearest 125k for video |
| JaxNav | CCE Only + ranked-TD decay | 281092, 281093, 280874 | nearest 125k for video |

The JSON files beside the media hold checkpoint or log hashes and the measured
outcomes. The paths are recorded as `run-id/file`, not as machine-specific home
directories.

## Rebuild

Use the historical environment that can load the checkpoints. Force JAX to CPU
when rendering on a login node.

```bash
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=.:src

python analysis/demos/render_readme_comparison.py frozenlake \
  --uniform /path/to/runs/257500/last.pkl \
  --per /path/to/runs/257525/last.pkl \
  --cce /path/to/runs/257600/last.pkl \
  --output-dir docs/media/demos

python analysis/demos/render_readme_comparison.py jaxnav \
  --uniform /path/to/runs/277734/checkpoints/ckpt_0125000.pkl \
  --per /path/to/runs/277751/checkpoints/ckpt_0125048.pkl \
  --cce /path/to/runs/281092/checkpoints/ckpt_0124039.pkl \
  --eval-seed 20260923 \
  --eval-episodes 1000 \
  --map-index 0 \
  --output-dir docs/media/demos
```
