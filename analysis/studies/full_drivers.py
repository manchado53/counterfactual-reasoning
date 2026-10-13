"""Executable historical pipelines with explicit data inputs and smoke scope."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace


def path_at(value, root):
    p = Path(value)
    return p if p.is_absolute() else root / p


def headers(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('# ') and ': ' in line:
            key, value = line[2:].split(': ', 1)
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            result[key] = value
    return result


def require_files(paths):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError('Missing analysis inputs:\n'+'\n'.join(missing))


def execute(config, out, root, source):
    if config['study'] == 'claim2_rebuild':
        return rebuild(config, out, root, source)
    paths = [path_at(p, root) for p in config.get('checkpoints', [])]
    # Explicit checkpoints avoid historical private-worktree/default directory lookup.
    if not paths:
        raise ValueError('checkpoints must list saved checkpoint paths explicitly')
    if config.get('smoke'):
        paths = paths[:1]
    require_files(paths)
    for p in paths:
        source(str(p))
    if config.get('validate_inputs_only'):
        return {'validated_files':[str(p) for p in paths]}, 'input_presence_only_not_execution'
    if config['study'] == 'covariance_checkpoints':
        from .checkpoint_covariance import run_checkpoint
        records = [run_checkpoint(str(p), p.stem,
                     n_rollouts=2 if config.get('smoke') else config['n_rollouts'],
                     verify=config.get('verify_linearisation', True)) for p in paths]
        # Full run_checkpoint retains checkpoint cf_horizon; smoke only cuts N and files.
        return {'records':records, 'smoke':config.get('smoke',False),
                'effective_n_rollouts':2 if config.get('smoke') else config['n_rollouts'],
                'horizon_policy':'checkpoint cf_horizon unchanged',
                'rng_policy':'Source cce_matrix and linearisation use seed=0; agent uses saved checkpoint seed.'}, 'fresh_checkpoint_gradient_analysis_no_training'
    from . import suitability_driver as driver
    from .suitability_envs import make_frozenlake_adapter
    from .suitability_scorecard import plot_scorecard
    options = dict(metric='total_variation', slip_prob=None, cf_n_rollouts=None,
                   batch=256, visit_episodes=100, eval_episodes=50,
                   horizon_states=12, horizons=[2,4,8,16,32])
    options.update(config['options'])
    if config.get('smoke'):
        options.update(cf_n_rollouts=2, visit_episodes=1, eval_episodes=1,
                       horizon_states=2, horizons=[2])
    args = SimpleNamespace(**options)
    result = {'schema_version':1, 'config':options, 'envs':{},
              'smoke_cf_horizon':2 if config.get('smoke') else None,
              'rng_policy':'Agent key from saved checkpoint config; horizon state subset seed=0, as source.',
              'smoke':config.get('smoke',False)}
    for env_name in config['envs']:
        agent = driver.build_agent_for_env(str(paths[0]), driver.ENV_SLIPPERY[env_name], args)
        if config.get('smoke'):
            agent.cf_horizon = 2
            agent._build_rollout_fn()
        adapter = make_frozenlake_adapter(agent,env_name,exact_truth=True)
        records = []
        for p in paths:
            driver._load_params(agent,str(p))
            metrics = driver.compute_all_metrics(agent,adapter,args)
            evaluation = agent.evaluate(n_episodes=args.eval_episodes)
            records.append({'ckpt':str(p), 'phase':p.stem, 'episode':None,
                            'win_rate_eval':float(evaluation['win_rate']),
                            'logged_win_pct':None, 'metrics':metrics})
        result['envs'][env_name] = {'slippery':driver.ENV_SLIPPERY[env_name],
                                    'checkpoints':records}
    plot_scorecard(result,str(out/'scorecard.png'))
    return result, 'fresh_checkpoint_rollouts_no_training'


def rebuild(config, out, root, source):
    import numpy as np
    from . import raw_logs
    manifest_path = source(config['manifest'])
    manifest = json.loads(manifest_path.read_text())
    jobs = config.get('smoke_jobs') if config.get('smoke') else config.get('jobs')
    if jobs is not None:
        manifest = {str(j):manifest[str(j)] for j in jobs}
    if not manifest:
        raise ValueError('Empty selected manifest')
    roots = [path_at(p,root) for p in config['runs_roots']]
    selected, missing, header_records = {}, [], {}
    for job,cfg in manifest.items():
        candidates = [r/str(job)/'metrics.log' for r in roots]
        present = [p for p in candidates if p.is_file()]
        if not present:
            missing.append({'job':job,'searched':[str(p) for p in candidates]})
            continue
        if len(present)>1 and len({p.read_bytes() for p in present})>1:
            raise ValueError(f'Conflicting raw logs for {job}: {present}')
        p = source(str(present[0])); selected[job] = p.parent
        h = headers(p)
        mismatches = {k:{'manifest':v,'header':h[k]} for k,v in cfg.items()
                      if k in h and v != h[k]}
        if mismatches:
            raise ValueError(f'Manifest/header conflict for {job}: {mismatches}')
        header_records[job] = {'logged_header':h, 'manifest_overrides':cfg,
                              'not_in_header':[k for k in cfg if k not in h],
                              'launch_source_proven':False}
    if missing:
        raise FileNotFoundError('Missing raw inputs: '+json.dumps(missing,sort_keys=True))
    if config.get('validate_inputs_only'):
        return {'headers':header_records,'selected_jobs':list(selected)}, 'input_presence_and_headers_only'
    if config['completion_policy'] not in ('historical_all', 'complete_only'):
        raise ValueError('completion_policy must be historical_all or complete_only')
    # Change only input lookup. All parser, completion and padding equations are original.
    previous = raw_logs._find_run_dir
    raw_logs._find_run_dir = lambda job,env: str(selected[job]) if job in selected else None
    try:
        dropped = []
        if config['completion_policy']=='complete_only':
            manifest,dropped = raw_logs.filter_complete_runs(manifest)
        groups = {}
        for job,cfg in manifest.items():
            label = str(cfg['slip_prob']) if config.get('group_by_slip') else config['env']
            groups.setdefault(label,{})[job] = cfg
        outputs = {}
        for label, rows in groups.items():
            # A duplicate seed within an arm is not another independent replicate.
            seen = set()
            for job,cfg in rows.items():
                arm=(cfg.get('algorithm'),cfg.get('priority_mixing','additive'),cfg.get('mu',.25),cfg.get('seed'))
                if arm in seen:
                    raise ValueError(f'Duplicate arm/seed in {label}: {arm}')
                seen.add(arm)
            subset = out/f'manifest_{label}.json'
            subset.write_text(json.dumps(rows,indent=2)+'\n')
            data = raw_logs.load_manifest(str(subset))
            labels = list(data['raw'])
            if not labels:
                raise ValueError(f'No parsed data for {label}')
            meta = {'manifest':str(manifest_path),'env':label,'threshold':config['threshold'],
                    'labels':labels,'env_type':data['env_type'],
                    'n_seeds':{a:data['raw'][a].shape[0] for a in labels},
                    'n_checkpoints':{a:data['raw'][a].shape[2] for a in labels},
                    'historical_padding':'repeat last value; longest-seed step grid',
                    'completion_policy':config['completion_policy'],
                    'smoke':config.get('smoke',False)}
            arrays = {'meta':json.dumps(meta)}
            for i,a in enumerate(labels):
                for key,prefix in [('raw','raw'),('eval_steps','steps'),('raw_length','len'),('raw_allies','allies')]:
                    arrays[f'{prefix}_{i}'] = data[key][a]
            target = out/f'{label}.npz'
            np.savez_compressed(target,**arrays)
            outputs[label] = {'cache':str(target), 'metadata':meta}
        if not outputs:
            raise ValueError('Completion filter left no runs: '+json.dumps(dropped))
        return {'groups':outputs,'headers':header_records,'dropped_runs':dropped,
                'warning':'Historical clocks/padding retained. Rebuilt analysis is not a training port.'}, 'raw_log_cache_rebuild_no_training'
    finally:
        raw_logs._find_run_dir = previous
