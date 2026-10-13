"""Build the README policy-comparison media from saved checkpoints.

FrozenLake uses one shared deterministic reset. JaxNav can render a disclosed
list of maps from the fixed 1,000-map evaluation bank. All three panels receive
the same initial state and step keys on each map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence


LABELS = {
    "uniform": "Uniform replay",
    "per": "PER",
    "cce": "CCE",
}
COLORS = {
    "uniform": "#4D4D4D",
    "per": "#E68613",
    "cce": "#8E44AD",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int, bold: bool = False):
    names = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        (
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
        ),
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _save_media(frames: list[Image.Image], stem: Path, fps: int) -> dict:
    stem.parent.mkdir(parents=True, exist_ok=True)
    gif = stem.with_suffix(".gif")
    mp4 = stem.with_suffix(".mp4")
    duration = round(1000 / fps)
    frames[0].save(
        gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )

    import imageio.v2 as imageio

    with imageio.get_writer(mp4, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            rgb = np.asarray(frame.convert("RGB"))
            # H.264 requires even dimensions.
            writer.append_data(rgb[: rgb.shape[0] // 2 * 2, : rgb.shape[1] // 2 * 2])
    return {"gif": gif.name, "mp4": mp4.name, "frames": len(frames), "fps": fps}


def _load_checkpoint(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _checkpoint_record(path: Path) -> dict:
    parts = path.parts
    try:
        run_index = parts.index("runs") + 1
        reference = "/".join(parts[run_index:])
    except ValueError:
        reference = path.name
    return {"reference": reference, "sha256": file_sha256(path)}


def _pad_and_join(
    panel_frames: dict[str, list[Image.Image]], tail_frames: int = 10
) -> list[Image.Image]:
    count = max(len(frames) for frames in panel_frames.values())
    padded = {
        arm: frames + [frames[-1]] * (count - len(frames))
        for arm, frames in panel_frames.items()
    }
    result = []
    for index in range(count):
        panels = [padded[arm][index] for arm in ("uniform", "per", "cce")]
        width = sum(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (width, height), "white")
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        result.append(canvas)
    result.extend([result[-1]] * tail_frames)
    return result


def _frozen_rollout(checkpoint: Path, reset_seed: int) -> tuple[object, list[tuple]]:
    from counterfactual_rl.agents.frozen_lake.dqn import _QNetwork
    from counterfactual_rl.envs.frozen_lake import FrozenLakeEnv

    saved = _load_checkpoint(checkpoint)
    cfg = saved["config"]
    env = FrozenLakeEnv(
        map_name=cfg.get("map_name", "8x8"),
        is_slippery=cfg.get("is_slippery", False),
    )
    network = _QNetwork(
        n_states=env.n_states,
        hidden_dim=cfg.get("hidden_dim", 64),
        n_layers=cfg.get("n_layers", 2),
        n_actions=env.n_actions,
    )
    params = jax.tree.map(jnp.asarray, saved["params"])
    key = jax.random.PRNGKey(reset_seed)
    _, state = env.reset(key)
    state = int(state)
    trajectory = [(state, None, 0.0, False)]
    for _ in range(200):
        action = int(jnp.argmax(network.apply(params, jnp.int32(state))))
        key, step_key = jax.random.split(key)
        _, next_state, reward, done, _ = env.step(
            step_key, jnp.int32(state), jnp.int32(action)
        )
        state = int(next_state)
        trajectory.append((state, action, float(reward), bool(done)))
        if bool(done):
            break
    return env, trajectory


def _frozen_frame(env, item: tuple, arm: str, step: int) -> Image.Image:
    state, _, _, done = item
    tile = env.desc[state // env.ncols][state % env.ncols]
    if tile == "G":
        status = "GOAL"
    elif tile == "H":
        status = "HOLE"
    elif step >= 200:
        status = "TIMEOUT"
    else:
        status = "RUNNING"
    size = 46
    top = 78
    panel = Image.new("RGB", (env.ncols * size + 24, env.nrows * size + top + 20), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), LABELS[arm], fill=COLORS[arm], font=_font(22, True))
    draw.text((12, 38), f"step {step}  |  {status}", fill="#222222", font=_font(16))
    tile_colors = {"S": "#8ecae6", "F": "#edf2f4", "H": "#293241", "G": "#55a630"}
    for row, values in enumerate(env.desc):
        for col, value in enumerate(values):
            x0, y0 = 12 + col * size, top + row * size
            draw.rounded_rectangle(
                (x0 + 2, y0 + 2, x0 + size - 2, y0 + size - 2),
                radius=5,
                fill=tile_colors[value],
                outline="#c4cbd2",
            )
            draw.text(
                (x0 + 17, y0 + 12),
                value,
                fill="white" if value in "HG" else "#555",
                font=_font(15, True),
            )
    row, col = divmod(state, env.ncols)
    cx, cy = 12 + col * size + size // 2, top + row * size + size // 2
    draw.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill="#d62828", outline="white", width=2)
    return panel


def build_frozen(args) -> dict:
    checkpoints = {arm: Path(getattr(args, arm)) for arm in LABELS}
    panel_frames = {}
    outcomes = {}
    for arm, checkpoint in checkpoints.items():
        env, trajectory = _frozen_rollout(checkpoint, args.reset_seed)
        panel_frames[arm] = [
            _frozen_frame(env, item, arm, step)
            for step, item in enumerate(trajectory)
        ]
        state = trajectory[-1][0]
        final_tile = env.desc[state // env.ncols][state % env.ncols]
        outcome = (
            "goal"
            if final_tile == "G"
            else "hole"
            if final_tile == "H"
            else "timeout"
        )
        outcomes[arm] = {"outcome": outcome, "steps": len(trajectory) - 1}
    frames = _pad_and_join(panel_frames)
    media = _save_media(frames, Path(args.output_dir) / "frozenlake_comparison", args.fps)
    return {
        "environment": "FrozenLake 8x8 deterministic",
        "selection_rule": "same training seed; final checkpoint; one shared deterministic reset",
        "reset_seed": args.reset_seed,
        "checkpoints": {arm: _checkpoint_record(path) for arm, path in checkpoints.items()},
        "outcomes": outcomes,
        "media": media,
    }


def _jaxnav_policy(checkpoint: Path):
    from training.learners.jaxnav.dqn import _QNetwork

    saved = _load_checkpoint(checkpoint)
    cfg = saved["config"]
    network = _QNetwork(
        obs_dim=205,
        hidden_dim=cfg.get("hidden_dim", 128),
        n_layers=cfg.get("n_layers", 3),
        n_actions=15,
    )
    return network, jax.tree.map(jnp.asarray, saved["params"]), cfg


def _jaxnav_rollout(env, network, params, init_state, step_keys):
    state = init_state
    obs = env.get_obs(state)
    raw_obs = [env._env.get_obs(state)]
    states = [state]
    rewards = [0.0]
    outcome = "timeout"
    for step, key in enumerate(step_keys):
        action = int(jnp.argmax(network.apply(params, obs)))
        obs, state, reward, done, _ = env.step(key, state, action)
        raw_obs.append(env._env.get_obs(state))
        states.append(state)
        rewards.append(float(reward))
        if bool(done):
            if bool(np.asarray(state.goal_reached).ravel()[0]):
                outcome = "goal"
            elif bool(np.asarray(state.move_term).ravel()[0]):
                outcome = "collision"
            else:
                outcome = "timeout"
            break
    return raw_obs, states, rewards, outcome, step + 1


def _jaxnav_frames(
    env,
    obs,
    states,
    rewards,
    arm: str,
    outcome: str,
    steps: int,
    map_index: int,
    frame_stride: int,
) -> list[Image.Image]:
    from jaxmarl.environments.jaxnav import JaxNavVisualizer

    with tempfile.NamedTemporaryFile(suffix=".gif") as handle:
        title = f"Map {map_index} | {LABELS[arm]} | {outcome.upper()} in {steps} steps"
        visualizer = JaxNavVisualizer(
            env._env,
            obs,
            states,
            rewards,
            title_text=title,
            plot_lidar=False,
            plot_reward=False,
        )
        visualizer.animate(save_fname=handle.name)
        image = Image.open(handle.name)
        all_frames = [
            frame.convert("RGB").resize((420, 420), Image.Resampling.LANCZOS)
            for frame in ImageSequence.Iterator(image)
        ]
        image.close()
    frames = all_frames[::frame_stride]
    if frames[-1] != all_frames[-1]:
        frames.append(all_frames[-1])
    return frames


def validate_map_indices(map_indices: list[int], eval_episodes: int) -> list[int]:
    if not map_indices:
        raise ValueError("at least one JaxNav map index is required")
    if len(set(map_indices)) != len(map_indices):
        raise ValueError("JaxNav map indices must be unique")
    invalid = [index for index in map_indices if not 0 <= index < eval_episodes]
    if invalid:
        raise ValueError(
            f"JaxNav map indices must be in [0, {eval_episodes}): {invalid}"
        )
    return map_indices


def build_jaxnav(args) -> dict:
    from training.envs.jaxnav import JaxNavEnv

    checkpoints = {arm: Path(getattr(args, arm)) for arm in LABELS}
    policies = {arm: _jaxnav_policy(path) for arm, path in checkpoints.items()}
    cfg = policies["cce"][2]
    env = JaxNavEnv(
        map_id=cfg.get("map_id", "Grid-Rand-Poly"),
        map_size=tuple(cfg.get("map_size", (8, 8))),
        fill=cfg.get("fill", 0.3),
        goal_radius=cfg.get("goal_radius", 0.8),
        goal_rew=cfg.get("goal_rew", 1.0),
        coll_rew=cfg.get("coll_rew", 0.0),
        max_steps=cfg.get("max_steps", 200),
        sparse_reward=cfg.get("sparse_reward", True),
    )
    n_eval = args.eval_episodes
    all_keys = jax.random.split(
        jax.random.PRNGKey(args.eval_seed), n_eval * (env.max_steps + 1)
    )
    map_indices = validate_map_indices(args.map_indices, n_eval)
    step_keys = all_keys[n_eval:].reshape(n_eval, env.max_steps, 2)
    frames = []
    maps = []
    for map_index in map_indices:
        _, init_state = env.reset(all_keys[map_index])
        panel_frames = {}
        outcomes = {}
        for arm, (network, params, _) in policies.items():
            obs, states, rewards, outcome, steps = _jaxnav_rollout(
                env, network, params, init_state, step_keys[map_index]
            )
            panel_frames[arm] = _jaxnav_frames(
                env,
                obs,
                states,
                rewards,
                arm,
                outcome,
                steps,
                map_index,
                args.frame_stride,
            )
            outcomes[arm] = {"outcome": outcome, "steps": steps}
        frames.extend(_pad_and_join(panel_frames, tail_frames=args.fps))
        maps.append({"map_index": map_index, "outcomes": outcomes})
    media = _save_media(frames, Path(args.output_dir) / "jaxnav_comparison", args.fps)
    return {
        "environment": "JaxNav Grid-Rand-Poly 8x8, fill 0.3, radius 0.8",
        "selection_rule": (
            "training seed 0; checkpoint nearest 125k; disclosed CCE-success "
            "examples from the fixed evaluation bank; illustration only"
        ),
        "eval_seed": args.eval_seed,
        "eval_episodes": n_eval,
        "map_indices": map_indices,
        "frame_stride": args.frame_stride,
        "checkpoints": {arm: _checkpoint_record(path) for arm, path in checkpoints.items()},
        "maps": maps,
        "media": media,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("environment", choices=("frozenlake", "jaxnav"))
    result.add_argument("--uniform", required=True)
    result.add_argument("--per", required=True)
    result.add_argument("--cce", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--fps", type=int, default=10)
    result.add_argument("--reset-seed", type=int, default=1)
    result.add_argument("--eval-seed", type=int, default=20260923)
    result.add_argument("--eval-episodes", type=int, default=1000)
    result.add_argument("--map-indices", type=int, nargs="+", default=[11, 12, 13, 17])
    result.add_argument("--frame-stride", type=int, default=2)
    return result


def main() -> None:
    args = parser().parse_args()
    record = (
        build_frozen(args)
        if args.environment == "frozenlake"
        else build_jaxnav(args)
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / f"{args.environment}_provenance.json"
    provenance.write_text(json.dumps(record, indent=2) + "\n")
    print(provenance)


if __name__ == "__main__":
    main()
