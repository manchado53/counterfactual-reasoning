"""Executable smoke/full runner. Scheduler and cross-task gate belong to main."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import jax
import numpy as np

from .transfer import ARMS, RECIPE, Trainer


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def append_json(path, value):
    with path.open('a') as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


def source_hashes():
    package = Path(__file__).resolve().parents[2]
    root = package.parent
    files = [Path(__file__), Path(__file__).with_name('transfer.py'),
             package / 'envs/connect_four_transfer.py']
    files += [root / 'src/counterfactual_rl/replay/buffers.py',
              root / 'src/counterfactual_rl/replay/consequence_buffers.py']
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def bank_hash():
    # Role, fixed empty opening, fixed foe and greedy tie rule define the bank.
    spec = dict(seed=20260923, roles=[0, 1] * 500, start='empty',
                opponent=RECIPE['opponent'], ties=RECIPE['opponent_ties'], learner_ties='lowest-legal')
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def effective_config(arm, seed):
    cce = arm.startswith('cce')
    return dict(RECIPE, arm=arm, seed=seed, cce_enabled=cce,
                normalise_priorities='rank' if cce else 'none', score_on_insert=cce,
                mu=1. if arm == 'cce_only' else .5, mu_c=1., mu_delta=1.,
                priority_mixing='multiplicative' if arm == 'cce_mul' else 'additive',
                active_features=dict(cce=cce, raw_td_per=arm == 'per',
                                     uniform_replay=arm == 'uniform', refresh=False),
                inactive_keys=[] if cce else ['cf_horizon', 'cf_n_rollouts', 'cf_gamma',
                    'cf_rollout_temperature', 'cf_action_prob_temperature', 'cf_score_chunk',
                    'cf_bootstrap', 'consequence_metric', 'consequence_aggregation',
                    'mu', 'mu_c', 'mu_delta', 'priority_mixing'])


def checked_eval(agent):
    before_key = np.asarray(agent.key).copy()
    before_numpy = np.random.get_state()
    metrics = agent.evaluate()
    np.testing.assert_array_equal(agent.key, before_key)
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0] and before_numpy[2:] == after_numpy[2:]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--seed', required=True, type=int, choices=(0, 1))
    parser.add_argument('--mode', required=True, choices=('smoke', 'train'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--require-gpu', action='store_true')
    args = parser.parse_args()
    if not args.output.is_absolute():
        parser.error('--output must be absolute')
    if args.require_gpu and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU requested, but unavailable')
    args.output.mkdir(parents=True, exist_ok=False)
    config = effective_config(args.arm, args.seed)
    versions = {name: importlib.metadata.version(name) for name in
                ('jax', 'jaxlib', 'flax', 'optax', 'pgx', 'numpy', 'scipy')}
    provenance = dict(task='connect_four', environment='connect_four', mode=args.mode, config=config,
        source_commit='73bb2f385c1f1711b763e03dff2037b5ca089556',
        base_commit='0231116bd4e97cfc06d01c95ee7247763ff6bb0d',
        source_manifest_sha256='b867cb1bd7b6d89789a61997dbe4902a381d028a3d5ecefcfa54d57e7979a894',
        source_manifest='episode_carry_pilot_20260930/manifest.json: variant=fixed',
        source_hashes=source_hashes(), packages=versions, python=platform.python_version(),
        devices=list(map(str, jax.devices())), backend=jax.default_backend(), eval_bank_hash=bank_hash(),
        entry_source=str(Path(__file__).resolve()),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        units='one learner decision = learner move plus reply; opening excluded from replay',
        test_scope='1000 repeats, two deterministic role cases; no independent-trial claim')
    write_json(args.output / 'provenance.json', provenance)
    print('PROVENANCE: ' + json.dumps(provenance), flush=True)
    np.random.seed(args.seed)
    start = time.monotonic()
    agent = Trainer(args.arm, args.seed)
    initial = [np.asarray(x).copy() for x in jax.tree.leaves(agent.params)]
    blocks, cadence = (8, 4) if args.mode == 'smoke' else (2000, 50)
    evals, durations = [], []
    for block in range(1, blocks + 1):
        block_start = time.monotonic()
        agent.learn_block()
        elapsed = time.monotonic() - block_start
        durations.append(elapsed)
        row = dict(agent.diagnostics(), block=block, seconds=time.monotonic() - start,
                   block_seconds=elapsed, arm=args.arm, seed=args.seed)
        assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
        append_json(args.output / 'progress.jsonl', row)
        print('PROGRESS: ' + json.dumps(row, allow_nan=False), flush=True)
        if block % cadence == 0:
            metrics = dict(checked_eval(agent), block=block, steps=agent.steps,
                           arm=args.arm, seed=args.seed)
            evals.append(metrics)
            append_json(args.output / 'eval.jsonl', metrics)
            agent.save(args.output / 'last.pkl')
            if args.mode == 'train':
                agent.save(args.output / f'ckpt_steps_{agent.steps:09d}.pkl')
            print('EVAL: ' + json.dumps(metrics), flush=True)
    assert agent.steps == blocks * 256 * 32
    assert agent.updates == agent.attempts == agent.steps // 16
    assert any(not np.array_equal(a, np.asarray(b)) for a, b in zip(initial, jax.tree.leaves(agent.params)))
    if agent.cce:
        assert np.all(agent.buffer.scored_at[:len(agent.buffer)] >= 0)
    restored = Trainer.load(args.output / 'last.pkl')
    assert checked_eval(restored) == checked_eval(agent)
    for a, b in zip(jax.tree.leaves(agent.params), jax.tree.leaves(restored.params)):
        np.testing.assert_array_equal(a, b)
    complete = dict(passed=True, task='connect_four', environment='connect_four',
        backend=jax.default_backend(), arm=args.arm, seed=args.seed, mode=args.mode,
        blocks=blocks, steps=agent.steps, env_steps=agent.steps, actual_updates=agent.updates,
        update_attempts=agent.attempts, checkpoint_reload_passed=True, eval_rng_unchanged=True,
        finite_diagnostics=True,
        parameters_changed=True, resumable=False, score_coverage=1. if agent.cce else None,
        source_hashes=provenance['source_hashes'], config=config, eval_bank_hash=bank_hash(),
        seconds=time.monotonic() - start, first_block_seconds=durations[0],
        steady_block_seconds=float(np.median(durations[1:])),
        endpoint=float(np.mean([e['win_rate'] for e in evals[-4:]])) if args.mode == 'train' else None,
        endpoint_checks=[e['block'] for e in evals[-4:]] if args.mode == 'train' else [],
        run=str(args.output))
    write_json(args.output / 'complete.json', complete)
    print('COMPLETE: ' + json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
