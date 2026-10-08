"""Synthetic log contracts; no trainer import and no frozen-runtime edits."""
import importlib.util
import json
from pathlib import Path

import pytest

PLOT = Path(__file__).resolve().parents[1] / 'analysis/claim2/plot_doorkey_transfer.py'
spec = importlib.util.spec_from_file_location('doorkey_plot', PLOT)
plotter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotter)


@pytest.fixture
def make_run(tmp_path):
    def make(arm='cce_add', seed=0, *, change=None, remove=None, provenance=None,
             eval_hash='fixed-bank'):
        cfg = {key: 'shared-test-value' for key in plotter.COMMON}
        cfg['cce_enabled'] = arm.startswith('cce')
        if cfg['cce_enabled']:
            cfg.update(cf_horizon=60, cf_n_rollouts=1, cf_gamma=.99,
                cf_rollout_temperature=0, cf_action_prob_temperature=.5,
                consequence_metric='wasserstein', consequence_aggregation='weighted_mean',
                cf_bootstrap=True, normalise_priorities='rank', score_on_insert=True,
                cf_score_chunk=1280, refresh_enabled=False, drift_probe_enabled=False,
                mu=1. if arm == 'cce_only' else .5, mu_c=1., mu_delta=1.,
                priority_mixing='multiplicative' if arm == 'cce_mul' else 'additive')
        cfg.update(change or {})
        if remove:
            cfg.pop(remove)
        p = dict(environment='doorkey', arm=arm, seed=seed, mode='train',
                 config=cfg, source_hashes={'runtime.py':'frozen-hash'},
                 eval_bank_hash='fixed-bank', versions={'jax':'0.9.1','numpy':'2.4.0'})
        p.update(provenance or {})
        path = tmp_path / f'{arm}-{seed}-{len(list(tmp_path.iterdir()))}'
        path.mkdir()
        (path/'provenance.json').write_text(json.dumps(p))
        row = dict(arm=arm, seed=seed, env_steps=409600, unique_tasks=1,
                   eval_episodes=1000, eval_bank_hash=eval_hash, win_rate=0., avg_length=50.)
        (path/'eval.jsonl').write_text(json.dumps(row)+'\n')
        return path
    return make


def test_accept_all_arms_and_seeds_with_expected_mixing(make_run):
    paths = [make_run(arm,seed) for arm in plotter.ARMS for seed in (0,1)]
    assert len(plotter.load_runs(paths)) == 10


@pytest.mark.parametrize('field,value', [
    ('cf_horizon',59), ('cf_n_rollouts',2), ('cf_gamma',.95),
    ('cf_rollout_temperature',.1), ('cf_action_prob_temperature',1.),
    ('consequence_metric','total_variation'), ('consequence_aggregation','max'),
    ('cf_bootstrap',False), ('normalise_priorities','none'), ('score_on_insert',False),
    ('cf_score_chunk',64), ('refresh_enabled',True), ('drift_probe_enabled',True),
    ('mu_c',.5), ('mu_delta',.5), ('cce_enabled',False),
])
def test_reject_mixed_cce_settings(make_run,field,value):
    paths = [make_run('cce_mul'),make_run('cce_add',change={field:value})]
    with pytest.raises(ValueError,match='CCE'):
        plotter.load_runs(paths)


@pytest.mark.parametrize('arm,field,value', [
    ('cce_mul','priority_mixing','additive'), ('cce_add','priority_mixing','multiplicative'),
    ('cce_only','priority_mixing','multiplicative'), ('cce_mul','mu',1.),
    ('cce_add','mu',1.), ('cce_only','mu',.5),
])
def test_reject_wrong_arm_mixing_even_for_single_run(make_run,arm,field,value):
    with pytest.raises(ValueError,match=field):
        plotter.load_runs([make_run(arm,change={field:value})])


@pytest.mark.parametrize('field', list(plotter.CCE_EXPECTED)+['mu','mu_c','mu_delta','priority_mixing'])
def test_reject_missing_cce_settings(make_run,field):
    with pytest.raises(ValueError,match=field):
        plotter.load_runs([make_run(remove=field)])


@pytest.mark.parametrize('arm', ['per','cce_mul'])
@pytest.mark.parametrize('metadata', [
    {'eval_bank_hash':'different-bank'},
    {'versions':{'jax':'0.9.2','numpy':'2.4.0'}},
])
def test_reject_mixed_bank_or_versions_across_arms(make_run,arm,metadata):
    with pytest.raises(ValueError,match='mixed'):
        plotter.load_runs([make_run('uniform'),make_run(arm,provenance=metadata)])


@pytest.mark.parametrize('metadata', [{'eval_bank_hash':None},{'eval_bank_hash':''},
                                      {'versions':None},{'versions':{}}])
def test_reject_missing_bank_or_versions(make_run,metadata):
    with pytest.raises(ValueError,match='missing'):
        plotter.load_runs([make_run(provenance=metadata)])


def test_reject_eval_row_bank_mismatch(make_run):
    with pytest.raises(ValueError,match='eval bank differs'):
        plotter.load_runs([make_run(eval_hash='wrong-bank')])


def test_reject_duplicate_arm_seed(make_run):
    with pytest.raises(ValueError,match='duplicate arm/seed'):
        plotter.load_runs([make_run(),make_run()])

