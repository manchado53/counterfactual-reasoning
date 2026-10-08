"""Reduced source equivalence; no private worktree, training, or network needed."""
import ast
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import warnings

if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pytest
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPORT = json.loads((ROOT/'studies/historical-analysis-migration/source-report.json').read_text())


def record(module):
    return next(r for r in REPORT['sources'] if r['port'].endswith('/'+module+'.py'))


def original(module, **namespace):
    """Execute original function bodies, avoiding historical import side effects."""
    r = record(module)
    tree = ast.parse((ROOT/r['snapshot']).read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and (r['functions'] is None or n.name in r['functions'])]
    ns = dict(np=np, os=os, json=json, warnings=warnings,
              contextmanager=contextmanager, spearmanr=spearmanr,
              EPS=0.01, BETA=0.25, NOISE_PEAK=2/3, **namespace)
    exec(compile(ast.Module(body=functions, type_ignores=[]), r['snapshot'], 'exec'), ns)
    return ns


@pytest.mark.parametrize('r', REPORT['sources'], ids=lambda r:Path(r['port']).stem)
def test_original_source_hash_and_function_bodies(r):
    raw = (ROOT/r['snapshot']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == r['sha256']
    old = {n.name:ast.dump(n, include_attributes=False) for n in ast.parse(raw).body
           if isinstance(n, ast.FunctionDef)}
    new = {n.name:ast.dump(n, include_attributes=False)
           for n in ast.parse((ROOT/r['port']).read_text()).body if isinstance(n, ast.FunctionDef)}
    for name, body in new.items():
        assert body == old[name], (r['port'], name)


def test_covariance_and_flatness_against_source():
    from experiments.studies import covariance as new, priority_flatness as flat
    old = original('covariance')
    old_flat = original('priority_flatness')
    rng = np.random.default_rng(41)
    for n in (3, 19, 101):
        c, d, u = rng.random(n), rng.random(n), rng.normal(size=n)
        for name in ('predicate', 'deployed_predicate', 'check_identities'):
            assert getattr(new,name)(c,d,u) == old[name](c,d,u)
        assert new.check_identities(c,d,u)['ok']
        assert flat.summarise(c) == old_flat['summarise'](c)


def test_suitability_nan_legal_actions_and_precision_ties():
    from experiments.studies import suitability as new, precision, diagnostic_metrics
    old = original('suitability', gini=diagnostic_metrics.gini,
                   spearman_boot=diagnostic_metrics.spearman_boot)
    old_precision = original('precision')
    returns = np.array([[[0,0],[1,1],[np.nan,np.nan]], [[0,1],[0,0],[1,1]]])
    np.testing.assert_equal(new.stakes_C(returns), old['stakes_C'](returns))
    assert new.snr(returns) == old['snr'](returns)
    assert new.concentration([0,1,np.nan,3]) == old['concentration']([0,1,np.nan,3])
    for k in (0.05,0.5,1):
        assert precision.precision_at_k([0,1,1,0],[1,0,1,0],k) == old_precision['precision_at_k']([0,1,1,0],[1,0,1,0],k)


def test_frozen_lake_oracle_matches_original_and_paper_cache():
    from counterfactual_rl.envs.frozen_lake import FrozenLakeEnv
    from experiments.studies.frozen_lake_oracle import compute_oracle
    old = original('frozen_lake_oracle', FrozenLakeEnv=FrozenLakeEnv)
    q, score, states = compute_oracle()
    oldq, oldscore, oldstates = old['compute_oracle']()
    np.testing.assert_array_equal(q, oldq)
    assert score == oldscore and states == oldstates
    with np.load(ROOT/'paper/repro/cache/claim1_frozen_lake_oracle.npz') as z:
        np.testing.assert_allclose(q,z['Q'],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(states,z['non_terminal'])


def test_claim2_actual_cache_points_and_censoring():
    pytest.importorskip('rliable')
    from experiments.studies import claim2_metrics as new
    from rliable import library as rly, metrics as rl_metrics
    from typing import Dict, Optional, Tuple
    old = original('claim2_metrics', Dict=Dict, Optional=Optional, Tuple=Tuple,
                   rly=rly, rl_metrics=rl_metrics)
    with np.load(ROOT/'paper/repro/cache/claim2_frozen_lake_no_slip.npz') as z:
        labels=json.loads(str(z['meta']))['labels']
        raw={a:z[f'raw_{i}'] for i,a in enumerate(labels)}
        steps={a:z[f'steps_{i}'] for i,a in enumerate(labels)}
    a,b = new.final_iqm(raw,reps=8), old['final_iqm'](raw,reps=8)
    assert {k:v[0] for k,v in a.items()} == {k:v[0] for k,v in b.items()}
    assert new.steps_to_threshold(raw,steps,0.75) == old['steps_to_threshold'](raw,steps,0.75)


def test_cvrp_oracle_against_source():
    from counterfactual_rl.envs.cvrp import CVRPEnv
    from experiments.studies.cvrp_oracle import compute_oracle
    from typing import Dict, List, Optional, Tuple
    old=original('cvrp_oracle', CVRPEnv=CVRPEnv, Dict=Dict, List=List,
                 Optional=Optional, Tuple=Tuple)
    env=CVRPEnv()
    a,b=compute_oracle(env),old['compute_oracle'](env)
    np.testing.assert_array_equal(a[0],b[0])
    assert a[1:] == b[1:]


def test_chess_oracle_sign_and_chosen_action_on_toy_records():
    import jax
    import jax.numpy as jnp
    from typing import NamedTuple, List
    from types import SimpleNamespace
    from experiments.studies.chess_oracle import compute_oracle_for_records
    class State(NamedTuple):
        legal_action_mask: object
        observation: object
    def step(state,action):
        return State(state.legal_action_mask,jnp.array([1.0+2.0*action]))
    env=SimpleNamespace(step=step)
    model=lambda observations:(jnp.zeros((len(observations),2)),observations[:,0])
    records=[SimpleNamespace(pgx_state=State(jnp.array([True,True]),jnp.array([0.])),action=0)]
    old=original('chess_oracle',jax=jax,jnp=jnp,List=List)
    assert compute_oracle_for_records(records,env,model,chunk_size=1) == old['compute_oracle_for_records'](records,env,model,chunk_size=1) == [2.0]


def test_retained_sources_are_lossless():
    import base64
    folder=ROOT/'studies/historical-analysis-migration/retained'
    p=json.loads((folder/'provenance.json').read_text())
    assert hashlib.sha256((folder/'rpdqn/274714/metrics.log').read_bytes()).hexdigest() == p['log_sha256']
    blob=base64.b64decode((folder/'hero_scores.npz.base64').read_text())
    assert hashlib.sha256(blob).hexdigest() == p['hero_sha256']
    import io
    with np.load(io.BytesIO(blob)) as z:
        data=json.loads((folder/'hero-scores.json').read_text())
        for name in z.files:
            np.testing.assert_array_equal(z[name],data[name])


def test_graded_cache_and_entropy():
    from experiments.studies import graded_slip as new
    old = original('graded_slip')
    path=ROOT/'paper/repro/cache/claim2_graded_slip.npz'
    a,b = new._load_cache(path),old['_load_cache'](path)
    for p in a[0]:
        for alg in a[0][p]:
            np.testing.assert_array_equal(a[0][p][alg], b[0][p][alg])
        assert new.slip_entropy(p) == old['slip_entropy'](p)
    assert new._split_main_probe([0,2/3,1]) == ([0,2/3],[1])


def test_rpdqn_dirty_source_parser_and_edge_smoothing():
    from experiments.studies import rpdqn_logs as new
    new.R = str(ROOT/'studies/historical-analysis-migration/fixtures/rpdqn')
    old = original('rpdqn_logs', R=new.R)
    a,b = new.load('fixture'),old['load']('fixture')
    for key in ('ep','gs','wr','ret'):
        np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_array_equal(new.smooth(np.arange(21)),old['smooth'](np.arange(21)))


@pytest.mark.parametrize('name', ['covariance','suitability-precision','rpdqn-logs'])
def test_new_entrypoint_reports_input_hashes_and_synthetic_scope(name,tmp_path):
    from experiments.studies.run import run
    config=json.loads((ROOT/f'configs/studies/{name}.json').read_text())
    out=tmp_path/name
    report=run(config,out)
    assert report['evidence_status'].startswith('synthetic_')
    assert report['inputs'][0]['sha256']
    assert json.loads((out/'report.json').read_text()) == report


def test_full_driver_function_bodies_match_saved_sources():
    records=json.loads((ROOT/'studies/historical-analysis-migration/full-driver-sources.json').read_text())
    for r in records:
        raw=(ROOT/r['snapshot']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==r['sha256']
        normalized=raw.decode()
        for before,after in r['transforms']:
            normalized=normalized.replace(before,after)
        old={n.name:ast.dump(n) for n in ast.parse(normalized).body if isinstance(n,ast.FunctionDef)}
        new={n.name:ast.dump(n) for n in ast.parse((ROOT/r['port']).read_text()).body if isinstance(n,ast.FunctionDef)}
        assert old==new, r['port']


def test_full_missing_inputs_are_exact(tmp_path):
    from experiments.studies.full_drivers import execute
    missing=tmp_path/'absent-checkpoint.pkl'
    with pytest.raises(FileNotFoundError,match='absent-checkpoint.pkl'):
        execute({'study':'covariance_checkpoints','checkpoints':[str(missing)]},tmp_path,tmp_path,lambda p:p)


def test_explicit_data_root_and_project_source_bundle(tmp_path):
    import tarfile
    from experiments.studies.run import run
    config=json.loads((ROOT/'configs/studies/covariance.json').read_text())
    report=run(config,tmp_path/'result',data_root=ROOT)
    assert report['data_root']==str(ROOT)
    assert any(s['module']=='experiments.studies.covariance' for s in report['imported_project_sources'])
    with tarfile.open(report['source_bundle']['path']) as archive:
        for src in report['imported_project_sources']:
            assert hashlib.sha256(archive.extractfile(src['bundle_member']).read()).hexdigest()==src['sha256']


def test_cli_supervisor_wall_cap(tmp_path):
    import subprocess,sys
    p=subprocess.run([sys.executable,'-m','experiments.studies.run',
        '--config',str(ROOT/'configs/studies/covariance.json'),'--out',str(tmp_path/'out'),
        '--max-wall','0.001'],capture_output=True,text=True,timeout=10,cwd=ROOT)
    assert p.returncode==124 and 'process group stopped' in p.stderr


def _full_original(module):
    records=json.loads((ROOT/'studies/historical-analysis-migration/full-driver-sources.json').read_text())
    r=next(r for r in records if r['port'].endswith('/'+module.__name__.split('.')[-1]+'.py'))
    tree=ast.parse((ROOT/r['snapshot']).read_text())
    ns=vars(module).copy()
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),r['snapshot'],'exec'),ns)
    return ns


def test_real_checkpoint_covariance_source_equivalence():
    from experiments.studies import checkpoint_covariance as new
    from experiments.studies.run import _json
    old=_full_original(new)
    path=str(ROOT/'paper/repro/cache/checkpoints/seed_0/trained.pkl')
    a=new.run_checkpoint(path,'trained',n_rollouts=2,verify=True)
    b=old['run_checkpoint'](path,'trained',n_rollouts=2,verify=True)
    assert _json(a)==_json(b)


def test_real_checkpoint_suitability_source_equivalence():
    from types import SimpleNamespace
    from experiments.studies import suitability_driver as new
    from experiments.studies.suitability_envs import make_frozenlake_adapter
    from experiments.studies.run import _json
    old=_full_original(new)
    path=str(ROOT/'paper/repro/cache/checkpoints/seed_0/trained.pkl')
    args=SimpleNamespace(metric='total_variation',slip_prob=None,cf_n_rollouts=2,
                         batch=256,visit_episodes=1,horizon_states=2,horizons=[2])
    def evaluate(fn):
        agent=new.build_agent_for_env(path,True,args)
        new._load_params(agent,path)
        agent.cf_horizon=2;agent._build_rollout_fn()
        adapter=make_frozenlake_adapter(agent,'FL-stoch',True)
        return _json(fn(agent,adapter,args))
    assert evaluate(new.compute_all_metrics)==evaluate(old['compute_all_metrics'])


def test_raw_rebuild_matches_source_on_retained_real_log(tmp_path):
    from experiments.studies import raw_logs
    from experiments.studies.full_drivers import rebuild
    config=json.loads((ROOT/'configs/studies/claim2-rebuild.json').read_text())
    config.update(smoke=True,runs_roots=['studies/historical-analysis-migration/retained/frozen_lake'])
    result,evidence=rebuild(config,tmp_path,ROOT,lambda p:Path(p) if Path(p).is_absolute() else ROOT/p)
    old=_full_original(raw_logs)
    old['_find_run_dir']=lambda job,env:str(ROOT/'studies/historical-analysis-migration/retained/frozen_lake'/job)
    expected=old['load_manifest'](str(tmp_path/'manifest_frozen_lake_no_slip.json'))
    with np.load(result['groups']['frozen_lake_no_slip']['cache']) as z:
        labels=json.loads(str(z['meta']))['labels']
        for i,label in enumerate(labels):
            np.testing.assert_array_equal(z[f'raw_{i}'],expected['raw'][label])
            np.testing.assert_array_equal(z[f'steps_{i}'],expected['eval_steps'][label])
    assert evidence=='raw_log_cache_rebuild_no_training'
    assert result['headers']['257499']['logged_header']['is_slippery'] is False
    assert not result['headers']['257499']['launch_source_proven']


def test_config_relative_root_works_away_from_repo(tmp_path):
    import subprocess,sys
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT)+os.pathsep+str(ROOT/'src')
    p=subprocess.run([sys.executable,'-m','experiments.studies.run',
        '--config',str(ROOT/'configs/studies/covariance.json'),'--out',str(tmp_path/'out')],
        capture_output=True,text=True,cwd=tmp_path,env=env,timeout=30)
    assert p.returncode==0,p.stderr
    assert json.loads((tmp_path/'out/report.json').read_text())['data_root']==str(ROOT)


def test_checkpoint_loader_methods_match_source():
    r=json.loads((ROOT/'studies/historical-analysis-migration/checkpoint-loader-source.json').read_text())
    raw=(ROOT/r['snapshot']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==r['sha256']
    def methods(text):
        cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='FrozenLakeDQN')
        return {n.name:ast.dump(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in r['methods']}
    assert methods(raw)==methods((ROOT/r['port']).read_text())
