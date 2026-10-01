import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_paired_configs_and_budget():
    # Protocol-only fixture: never read or ship a live runtime snapshot.
    manifest = json.loads((ROOT / 'tests/fixtures/carry_pilot_manifest.json').read_text())
    rows = manifest['rows']
    assert len(rows) == 20
    assert {r['arm'] for r in rows} == {'uniform', 'per', 'cce_mul', 'cce_add', 'cce_only'}
    for arm in {r['arm'] for r in rows}:
        for seed in [0, 1]:
            pair = [r for r in rows if r['arm'] == arm and r['seed'] == seed]
            assert {r['variant'] for r in pair} == {'old', 'fixed'}
            assert pair[0]['config'] == pair[1]['config']
            cfg = pair[0]['config']
            assert cfg['explore_hold_steps'] == 20
            assert cfg['epsilon_decay_episodes'] == 62500
            assert cfg['n_envs'] * cfg['collect_steps'] * manifest['blocks'] == manifest['env_steps']
            assert cfg['eval_episodes'] == 1000 and cfg['fixed_eval_seed'] == 20260923
            if arm.startswith('cce'):
                assert cfg['score_on_insert'] is True
                assert cfg['cf_horizon'] == 60 and cfg['cf_n_rollouts'] == 1
                assert cfg['consequence_metric'] == 'wasserstein'
                if arm == 'cce_mul':
                    assert cfg['priority_mixing'] == 'multiplicative'
                    assert cfg['mu_c'] == cfg['mu_delta'] == 1.
                else:
                    assert cfg['priority_mixing'] == 'additive'
                    assert cfg['mu'] == (1. if arm == 'cce_only' else .5)
