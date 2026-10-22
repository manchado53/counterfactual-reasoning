"""Run a score-only intervention on sealed historical learners."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]
WORK = ROOT/'work/value-score-replay'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((BUNDLE/'manifest.json').read_text())
    row = manifest['rows'][args.task]
    env, arm = row['environment'], row['arm']
    for name, expected in manifest.get('driver_hashes', {}).items():
        assert hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest() == expected, name
    assert sys.version == manifest['versions']['python'], 'Python runtime differs from sealed reference'
    for name, expected in manifest['source_hashes'].items():
        if name.startswith(env+'/'):
            assert hashlib.sha256((WORK/name).read_bytes()).hexdigest() == expected, name
    for package, version in manifest['versions']['packages'].items():
        assert importlib.metadata.version(package) == version, package
    runtime = WORK/'jaxnav/fixed/src' if env == 'jaxnav' else WORK/'frozenlake/src'
    sys.path[:0] = [str(runtime), str(WORK/'jaxnav/vendor'), str(WORK/'jaxnav/scripts')]
    import jax
    import numpy as np
    from value_scores import FrozenLakeValue, JaxNavValue
    assert jax.default_backend() == ('cpu' if args.cpu else 'gpu')
    if env == 'jaxnav':
        from counterfactual_rl.agents.jax_nav import dqn as base
        from counterfactual_rl.agents.jax_nav import consequence_dqn as cce
        from counterfactual_rl.agents.jax_nav.consequence_dqn_vectorized import JaxNavConsequenceDQNVectorized
        from decay_buffer import DecayBuffer, blend_weight
        cce.ConsequenceReplayBuffer = DecayBuffer
        class Parent(JaxNavConsequenceDQNVectorized):
            def _update(self):
                self.buffer.blend = blend_weight(self._current_episode, self.config['decay_enabled'],
                                                self.config['decay_start'], self.config['decay_end'])
                return super()._update()
        Value = JaxNavValue
    else:
        from counterfactual_rl.agents.frozen_lake import dqn as base
        from counterfactual_rl.agents.frozen_lake.consequence_dqn_vectorized import FrozenLakeConsequenceDQNVectorized as Parent
        Value = FrozenLakeValue
    assert Path(base.__file__).resolve().is_relative_to(runtime)
    cfg = row['config'].copy()
    overrides = {}
    if args.smoke:
        overrides = dict(n_envs=2, collect_steps=4, buffer_capacity=32, batch_size=4,
                         hidden_dim=8, n_layers=1, n_episodes=8, eval_interval=2, eval_episodes=4,
                         n_checkpoints=0, save_every=1000, cf_horizon=4, cf_n_rollouts=2,
                         n_score_sample=4, score_interval=1, target_update_freq=8,
                         n_steps_per_update=4)
        if env == 'jaxnav':
            overrides.update(cf_score_chunk=4, cf_drift_probe_n=2, cf_drift_probe_start_ep=0,
                             decay_start=2, decay_end=4, qdiag_every=1)
        cfg.update(overrides)
    scheduler_job = os.environ.get('SLURM_JOB_ID')
    run_id = (scheduler_job or 'cpu-'+str(time.time_ns())) + f'-task{args.task}' + ('-smoke' if args.smoke else '')
    run = Path(base.__file__).parent/'runs'/run_id
    assert not run.exists(), f'Refusing reused run {run}'
    os.environ['SLURM_JOB_ID'] = run_id
    run.mkdir(parents=True)
    started = time.monotonic()
    ancestor = type('ValueAgent', (Value, Parent), {}) if arm == 'value' else Parent
    class Agent(ancestor):
        def evaluate(self, n_episodes=1000):
            result = super().evaluate(n_episodes)
            record = dict(result, env_steps=self.total_steps, episode=self._current_episode,
                          seconds=time.monotonic()-started, score_calls=getattr(self,'value_score_calls',0))
            if env == 'jaxnav':
                h = hashlib.sha256()
                for leaf in jax.tree.leaves((self._fixed_eval_init_states, self._fixed_eval_step_keys)):
                    a = np.asarray(leaf); h.update(str((a.shape,a.dtype)).encode()); h.update(a.tobytes())
                record['eval_bank_hash'] = h.hexdigest()
                if not args.smoke:
                    assert record['eval_bank_hash'] == '357092b89bd5524ee397c69ae45a8bf86524ca369eef8b0c71a5507e77d39cf5'
            with (run/'study_eval.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            print('PROGRESS '+json.dumps(record), flush=True)
            return result
    agent = Agent(cfg)
    assert agent.config == cfg, 'Historical defaults changed the recovered config'
    provenance = dict(row=row, smoke=args.smoke, overrides=overrides, effective_config=cfg,
                      device=[d.device_kind for d in jax.devices()], python=sys.version,
                      manifest_sha256=hashlib.sha256((BUNDLE/'manifest.json').read_bytes()).hexdigest())
    (run/'study_provenance.json').write_text(json.dumps(provenance,indent=2))
    status_dir = WORK/'status'; status_dir.mkdir(exist_ok=True)
    status_path = status_dir/f'{run_id}.json'
    status = dict(task=args.task, environment=env, arm=arm, seed=row['seed'], run=str(run),
                  smoke=args.smoke, job=run_id, scheduler_job=scheduler_job, state='running')
    status_path.write_text(json.dumps(status,indent=2))
    print('RUN '+str(run), flush=True)
    try:
        agent.learn(verbose=False)
        if arm == 'value':
            assert agent._compiled_rollout_fn is None
            assert agent.value_score_calls > 0
        status.update(state='complete', seconds=time.monotonic()-started,
                      episodes=len(agent.episode_returns), env_steps=agent.total_steps,
                      value_score_calls=getattr(agent,'value_score_calls',0),
                      value_score_seconds=getattr(agent,'value_score_seconds',0.))
    except Exception as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        status_path.write_text(json.dumps(status,indent=2))
    print('COMPLETE '+json.dumps(status),flush=True)


if __name__ == '__main__':
    main()
