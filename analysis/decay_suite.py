"""Seal data-derived run configs and submit shared-cap mixed smoke/train arrays."""
import argparse,ast,hashlib,importlib.metadata,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path('/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning')
OLD=PROJECT/'.claude/worktrees/research+cce-robotics-transfer'
PILOT=PROJECT/'.claude/worktrees/jaxnav-episode-carry/work/episode_carry_pilot_20260930'
WORK=ROOT/'work/decay_20261007_v1'
LOG=Path('/home/ad.msoe.edu/manchadoa/jaxnav_decay_logs')
PY='/home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python'

def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2))

def prepare():
    assert not WORK.exists(),'Refusing to overwrite suite'
    refs={}
    for array in ['277733','278035']:
        for log in sorted((Path.home()/'jaxnav_logs').glob(array+'_*.out')):
            text=log.read_text();launch=json.loads(next(l[8:] for l in text.splitlines() if l.startswith('CONFIG: ')))
            if launch['algorithm']!='consequence-dqn':continue
            run=Path(re.search(r'Run dir: (.+)',text)[1].strip());cfg={}
            for line in (run/'metrics.log').read_text().splitlines():
                if not line.startswith('#'):break
                z=re.match(r'^# ((?:[a-z_][a-z_0-9]*|PER_parameters)): (.*)$',line)
                if z:
                    try:cfg[z[1]]=ast.literal_eval(z[2])
                    except (SyntaxError,ValueError):cfg[z[1]]=z[2]
            arm='mul' if cfg['priority_mixing']=='multiplicative' else ('only' if cfg['mu']==1 else 'add')
            assert cfg['sparse_reward'] and cfg['cf_horizon']==60 and cfg['normalise_priorities']=='rank'
            assert cfg['fixed_eval_seed']==20260923 and cfg['eval_episodes']==1000
            assert (arm,cfg['seed']) not in refs
            refs[arm,cfg['seed']]=(cfg,str(log),str(run))
    assert len(refs)==15
    WORK.mkdir(parents=True);LOG.mkdir(exist_ok=True)
    ignore=shutil.ignore_patterns('runs','__pycache__','*.pyc')
    for collector in ['old','fixed']:
        shutil.copytree(OLD/'src',WORK/collector/'src',ignore=ignore)
    # Only the two carry-fix trainer files differ between collector runtimes.
    for name in ['dqn_vectorized.py','consequence_dqn_vectorized.py']:
        rel=Path('src/counterfactual_rl/agents/jax_nav')/name
        shutil.copy2(PROJECT/rel,WORK/'fixed'/rel)
    changed=[str(p.relative_to(WORK/'old')) for p in (WORK/'old').rglob('*.py')
        if p.read_bytes()!=(WORK/'fixed'/p.relative_to(WORK/'old')).read_bytes()]
    assert sorted(changed)==['src/counterfactual_rl/agents/jax_nav/consequence_dqn_vectorized.py','src/counterfactual_rl/agents/jax_nav/dqn_vectorized.py']
    shutil.copytree(PILOT/'vendor',WORK/'vendor',ignore=ignore)
    (WORK/'scripts').mkdir()
    for name in ['decay_run.py','decay_buffer.py','test_decay_buffer.py','decay_suite.py']:
        shutil.copy2(ROOT/'analysis'/name,WORK/'scripts'/name)
    shutil.copy2(ROOT/'slurm/decay.sbatch',WORK/'decay.sbatch')
    rows=[]
    for gpu in ['h100','v100','t4']:
        # Smokes and full jobs share ONE array per GPU type, hence ONE cap.
        seeds={'h100':[0,1,2],'v100':[3],'t4':[4]}[gpu]
        for mode in ['smoke','train']:
            for arm in ['mul','add','only']:
                for collector,decay in [('old',True),('fixed',False),('fixed',True)]:
                    for seed in ([seeds[0]] if mode=='smoke' else seeds):
                        cfg,log,run=refs[arm,seed];cfg=cfg.copy()
                        cfg.update(n_episodes=3000 if mode=='smoke' else 200000,
                            decay_start=1000 if mode=='smoke' else 100000,
                            decay_end=2000 if mode=='smoke' else 150000,
                            decay_enabled=decay,collector_variant=collector,
                            experiment='ranked_td_decay_20261007')
                        if mode=='smoke':cfg.update(n_buffer_snapshots=0,n_checkpoints=0)
                        rows.append(dict(task=len(rows),mode=mode,gpu=gpu,arm=arm,collector=collector,
                            decay=decay,seed=seed,config=cfg,parent_log=log,parent_run=run))
    assert sum(r['mode']=='train' for r in rows)==45
    dump(WORK/'manifest.json',dict(rows=rows,caps={'h100':6,'v100':2,'t4':2},
        controls='15 historical old/no-decay runs; crop at 200k; hardware/source limitations apply',
        gate='No GPU smoke dependency by user request; static tests required',
        source_commits={'old':subprocess.check_output(['git','-C',str(OLD),'rev-parse','HEAD'],text=True).strip(),
                        'fixed':subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()}))
    dump(WORK/'versions.json',dict(python=sys.version,packages={n:importlib.metadata.version(n) for n in ['jax','jaxlib','jaxmarl','flax','optax','numpy','scipy','chex']}))
    hashes={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in WORK.rglob('*') if p.is_file()}
    dump(WORK/'hashes.json',hashes)
    print('PREPARED',WORK,len(rows),'tasks')

def monitor(work,job):
    previous=None
    while True:
        r=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode'],capture_output=True,text=True)
        if r.stdout!=previous:
            print(time.strftime('%Y-%m-%d %H:%M:%S'),r.stdout,flush=True);previous=r.stdout
        if r.returncode==0 and r.stdout.strip() and not any(s in r.stdout for s in ['PENDING','RUNNING','COMPLETING','CONFIGURING','REQUEUED','SUSPENDED']):return
        time.sleep(45)

def seal():
    assert not (WORK/'jobs.json').exists(),'Never reseal submitted jobs'
    for name in ['decay_run.py','decay_buffer.py','test_decay_buffer.py','decay_suite.py']:
        shutil.copy2(ROOT/'analysis'/name,WORK/'scripts'/name)
    manifest=json.loads((WORK/'manifest.json').read_text())
    rows=[]
    # Interleave independent smoke and full jobs so free slots can start both.
    for gpu in ['h100','v100','t4']:
        smokes=[r for r in manifest['rows'] if r['gpu']==gpu and r['mode']=='smoke']
        trains=[r for r in manifest['rows'] if r['gpu']==gpu and r['mode']=='train']
        while smokes or trains:
            if smokes:rows.append(smokes.pop(0))
            if trains:rows.append(trains.pop(0))
    for task,row in enumerate(rows):row['task']=task
    manifest['rows']=rows;dump(WORK/'manifest.json',manifest)
    paths=list((WORK/'old').rglob('*.py'))+list((WORK/'fixed').rglob('*.py'))+list((WORK/'vendor').rglob('*.py'))+list((WORK/'scripts').glob('*.py'))
    paths += [WORK/'manifest.json',WORK/'versions.json',WORK/'decay.sbatch']
    dump(WORK/'hashes.json',{str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print('SEALED',len(rows),'tasks, interleaved smoke and full training')

def submit(dry):
    assert not (WORK/'jobs.json').exists(),'Already submitted'
    for name,h in json.loads((WORK/'hashes.json').read_text()).items():
        assert hashlib.sha256((WORK/name).read_bytes()).hexdigest()==h,name
    rows=json.loads((WORK/'manifest.json').read_text())['rows']
    jobs=[]
    for gpu,partition,cap in [('h100','dgxh100',6),('v100','dgx',2),('t4','teaching',2)]:
        ids=[r['task'] for r in rows if r['gpu']==gpu]
        cmd=['sbatch','--parsable','--job-name=cce-decay-'+gpu,'--partition='+partition,
            '--account=undergrad_research','--qos=normal',
            '--gres=gpu:'+gpu+':1','--cpus-per-task=8','--mem=48G','--time=3-00:00:00',
            '--array='+','.join(map(str,ids))+'%'+str(cap),
            '--output='+str(LOG/'%A_%a.out'),'--error='+str(LOG/'%A_%a.err')]
        if dry:cmd.append('--test-only')
        cmd += [str(WORK/'decay.sbatch'),str(WORK)]
        print('COMMAND',cmd,flush=True)
        result=subprocess.run(cmd,capture_output=True,text=True,check=True)
        print(result.stdout,result.stderr,flush=True)
        if not dry:
            job=result.stdout.strip().split(';')[0];assert job.isdigit()
            fh=(WORK/f'monitor-{job}.log').open('a')
            proc=subprocess.Popen([PY,str(WORK/'scripts/decay_suite.py'),'monitor','--job',job],stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append(dict(gpu=gpu,job=job,monitor_pid=proc.pid,cap=cap))
            dump(WORK/'jobs.json',jobs)
    if dry:dump(WORK/'dry-run.json',dict(passed=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','seal','dry-run','submit','monitor']);p.add_argument('--job');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='seal':seal()
    elif a.action=='monitor':monitor(WORK,a.job)
    else:
        if a.action=='submit':assert (WORK/'dry-run.json').exists()
        submit(a.action=='dry-run')
