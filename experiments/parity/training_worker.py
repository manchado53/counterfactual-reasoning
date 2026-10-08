"""Bounded actual JaxNav learn() trace. Run only through training.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--kind", choices=["reference", "candidate"], required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", choices=["smoke", "full-size"], default="smoke")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--gpu-type", choices=["h100", "t4"], default="h100")
    p.add_argument("--chunks", type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.chunks <= 3:
        p.error("Gate is limited to 1..3 collection chunks")
    # GPU memory mappings need a large address range; SLURM bounds physical RAM.
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, available[:8 if args.device == "gpu" else 4])
    if args.device == "cpu":
        resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (14400, 14400))
    args.output.mkdir()
    rows = json.loads((args.reference/"manifest.json").read_text())["rows"]
    row = next(r for r in rows if r["task"] == args.task and r["mode"] == "train")
    cfg = row["config"].copy()
    overrides = dict(n_envs=2, collect_steps=4, batch_size=4, buffer_capacity=17,
        max_steps=8, n_episodes=6, eval_episodes=2, eval_interval=2,
        n_steps_per_update=4, target_update_freq=8, n_checkpoints=0, n_buffer_snapshots=0,
        decay_start=2, decay_end=4, qdiag_every=1, cf_score_chunk=8)
    if args.profile == "full-size":
        overrides = {}
    cfg.update(overrides)
    sys.path.insert(0, str(args.reference/"vendor"))
    os.environ["SLURM_JOB_ID"] = "parity-" + args.output.name
    os.environ["CCE_RUN_DIR"] = str(args.output/"logs")
    if args.kind == "reference":
        sys.path[:0] = [str(args.source/row["collector"]/"src"), str(args.source/"scripts")]
        from counterfactual_rl.agents.jax_nav import consequence_dqn as module
        from counterfactual_rl.agents.jax_nav import dqn as logger_module
        from counterfactual_rl.agents.jax_nav.consequence_dqn_vectorized import JaxNavConsequenceDQNVectorized as Base
        from decay_buffer import DecayBuffer, blend_weight
        module.ConsequenceReplayBuffer = DecayBuffer
        # Historical logger derives its output from __file__. Redirect only that
        # metadata so the frozen reference never receives diagnostic run outputs.
        logger_module.__file__ = str(args.output / "reference_dqn.py")

        class Scheduled(Base):
            def _update(self):
                self.buffer.blend = blend_weight(self._current_episode, cfg["decay_enabled"],
                                                 cfg["decay_start"], cfg["decay_end"])
                return super()._update()
        agent = Scheduled(cfg)
    else:
        sys.path[:0] = [str(args.source), str(args.source/"src")]
        from experiments.learners.jaxnav.build import build_agent
        agent = build_agent(cfg)
    import jax
    import numpy as np
    if jax.default_backend() != args.device:
        raise RuntimeError("Requested parity device is unavailable")
    if args.device == "gpu" and not all(args.gpu_type.upper() in d.device_kind for d in jax.devices()):
        raise RuntimeError("Full-size GPU gate requires the requested " + args.gpu_type)
    if agent.config != cfg:
        raise ValueError("Resolved config drift")
    events = []
    chunks_seen = 0
    updates_seen = 0

    class PrefixComplete(Exception):
        pass

    def tree(value):
        leaves = []
        for leaf in jax.tree.leaves(value):
            a = np.ascontiguousarray(np.asarray(leaf))
            if a.dtype.hasobject:
                raise ValueError("Object dtype cannot form a stable trace")
            leaves.append(dict(shape=list(a.shape), dtype=a.dtype.str,
                               sha256=hashlib.sha256(a.tobytes()).hexdigest()))
        # Dataclass module names differ after a move; compare leaf paths, not repr(treedef).
        paths = [str(path) for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]]
        return dict(paths=paths, leaves=leaves)

    def emit(kind, **values):
        events.append(dict(kind=kind, **{k: tree(v) for k,v in values.items()}))

    # Instrument existing compiled functions, with no extra random draws.
    update = agent._update_step
    def traced_update(*values):
        nonlocal updates_seen
        if args.profile == "full-size" and updates_seen == 0:
            # Independent diagnostic from the actual batch; not a second optimizer step.
            params, target, _, states, actions, rewards, nexts, dones, weights = values
            def diagnostic_loss(params):
                q = jax.vmap(agent.network.apply, in_axes=(None,0))(params,states)
                next_target = jax.vmap(agent.network.apply,in_axes=(None,0))(target,nexts)
                if cfg["double_dqn"]:
                    next_online = jax.vmap(agent.network.apply,in_axes=(None,0))(params,nexts)
                    choice = jax.numpy.argmax(jax.lax.stop_gradient(next_online),axis=-1)
                    best = next_target[jax.numpy.arange(next_target.shape[0]),choice]
                else:
                    best = next_target.max(axis=-1)
                targets = rewards + agent.gamma * best * (1-dones)
                td = targets - q[jax.numpy.arange(q.shape[0]),actions]
                return jax.numpy.mean(weights*td**2), (targets,td)
            diagnostic, gradients = jax.value_and_grad(diagnostic_loss,has_aux=True)(params)
            emit("gradient-diagnostic", loss_targets_td=diagnostic, gradients=gradients)
        result = update(*values)
        emit("update", inputs=values, outputs=result)
        updates_seen += 1
        return result
    agent._update_step = traced_update
    build_collect = agent._build_collect_fn
    def traced_build_collect():
        build_collect()
        collect = agent._collect_fn
        def traced_collect(*values):
            nonlocal chunks_seen
            if args.profile == "full-size" and chunks_seen >= args.chunks:
                raise PrefixComplete()
            result = collect(*values)
            chunks_seen += 1
            emit("collect", inputs=values, outputs=result)
            return result
        agent._collect_fn = traced_collect
    agent._build_collect_fn = traced_build_collect
    build_rollout = agent._build_rollout_fn
    def traced_build_rollout():
        build_rollout()
        rollout = agent._compiled_rollout_fn
        def traced_rollout(*values):
            result = rollout(*values)
            emit("rollout", outputs=result)
            return result
        agent._compiled_rollout_fn = traced_rollout
    agent._build_rollout_fn = traced_build_rollout
    add = agent._add_chunk_to_buffer
    def traced_add(outputs):
        add(outputs)
        n = len(agent.buffer)
        emit("insert", scores=agent.buffer.consequence_scores[:n],
             td=agent.buffer.td_magnitudes[:n], scored_at=agent.buffer.scored_at[:n],
             key=agent._key, rng=np.random.get_state()[1:])
    agent._add_chunk_to_buffer = traced_add
    evaluate = agent.evaluate
    def traced_evaluate(n_episodes=2):
        result = evaluate(n_episodes)
        emit("eval", result=result, bank=(agent._fixed_eval_init_states, agent._fixed_eval_step_keys),
             key=agent._key, rng=np.random.get_state()[1:])
        return result
    agent.evaluate = traced_evaluate
    sample = agent.buffer.sample
    def traced_sample(n):
        result = sample(n)
        emit("sample", indices=result[1], weights=result[2], blend=agent.buffer.blend)
        return result
    agent.buffer.sample = traced_sample
    emit("initial", params=agent.params, target=agent.target_params, optimizer=agent.opt_state, key=agent._key)
    try:
        agent.learn(verbose=False)
    except PrefixComplete:
        # Stop only at a collection boundary, after the last chunk's real updates.
        agent.evaluate(cfg["eval_episodes"])
        agent._write_qdiag(agent._current_episode,agent.q_update_count)
        # Exercise original decay clocks on the captured state, not 150k fake episodes.
        saved_episode = agent._current_episode
        for episode in (cfg["decay_start"]-1,cfg["decay_start"],
                        (cfg["decay_start"]+cfg["decay_end"])//2,
                        cfg["decay_end"],cfg["decay_end"]+1):
            agent._current_episode = episode
            emit("controlled-clock",episode=episode)
            agent._update()
        agent._current_episode = saved_episode
        agent.metrics_logger.close()
    emit("final", params=agent.params, target=agent.target_params, optimizer=agent.opt_state,
         key=agent._key, rng=np.random.get_state()[1:], lengths=agent.episode_lengths,
         returns=agent.episode_returns, steps=agent.total_steps)
    # Logs contain wall times and paths. Compare only scientific log streams.
    logdir = Path(agent.metrics_logger.dir)
    for name in ("eval.jsonl", "qdiag.jsonl"):
        records = [json.loads(line) for line in (logdir/name).read_text().splitlines()] if (logdir/name).exists() else []
        emit(name, records=records)
    result = dict(task=row["task"], overrides=overrides, config=cfg, events=events,
                  device=[d.device_kind for d in jax.devices()], full_production_budget=False,
                  profile=args.profile,chunks_seen=chunks_seen)
    (args.output/"trace.json").write_text(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
