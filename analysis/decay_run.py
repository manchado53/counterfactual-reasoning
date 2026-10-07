"""Run sealed historical/fixed collector with unchanged inherited training loop."""
import os, sys, json, hashlib, time
from pathlib import Path
import numpy as np

work=Path(sys.argv[1]).resolve()
task=int(os.environ['SLURM_ARRAY_TASK_ID'])
manifest=json.loads((work/'manifest.json').read_text())
row=manifest['rows'][task]
assert row['task']==task
for name,digest in json.loads((work/'hashes.json').read_text()).items():
    assert hashlib.sha256((work/name).read_bytes()).hexdigest()==digest,name
runtime=work/row['collector']/'src'
sys.path[:0]=[str(runtime),str(work/'vendor')]
import importlib.metadata
versions=json.loads((work/'versions.json').read_text())
assert sys.version==versions['python']
for name,value in versions['packages'].items():
    assert importlib.metadata.version(name)==value,name
import jax
import jaxmarl
from counterfactual_rl.agents.jax_nav import consequence_dqn as module
from counterfactual_rl.agents.jax_nav import dqn as base
from counterfactual_rl.agents.jax_nav.consequence_dqn_vectorized import JaxNavConsequenceDQNVectorized
from decay_buffer import DecayBuffer,blend_weight
assert Path(base.__file__).resolve().is_relative_to(runtime)
assert Path(jaxmarl.__file__).resolve().is_relative_to(work/'vendor')
assert jax.default_backend()=='gpu'
device='; '.join(d.device_kind for d in jax.devices()); assert row['gpu'].lower() in device.lower(),device
run=Path(base.__file__).parent/'runs'/os.environ['SLURM_JOB_ID']
assert not run.exists(),f'Refusing reused output {run}'
module.ConsequenceReplayBuffer=DecayBuffer
started=time.monotonic()

class Agent(JaxNavConsequenceDQNVectorized):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.ghosts=0;self.valid_ends=0;self.blends_seen=set()
        self.bank_digest=None

    def _add_chunk_to_buffer(self,outputs):
        ended=np.asarray(outputs[7]).astype(bool)
        ghosts=np.asarray(outputs[8].done).astype(bool).reshape(ended.shape)
        self.ghosts+=int(ghosts.sum())
        self.valid_ends+=int((ended & ~ghosts).sum())
        if row['collector']=='fixed':assert not ghosts.any(),'Ended state stepped'
        super()._add_chunk_to_buffer(outputs)
        assert np.all(self.buffer.scored_at[:len(self.buffer)]>=0),'Coverage lost'

    def _update(self):
        self.buffer.blend=blend_weight(self._current_episode,row['decay'],
            self.config['decay_start'],self.config['decay_end'])
        self.blends_seen.add('zero' if self.buffer.blend==0 else ('one' if self.buffer.blend==1 else 'middle'))
        return super()._update()

    def evaluate(self,n_episodes=1000):
        key=np.asarray(self._key).copy();rng=np.random.get_state()
        result=super().evaluate(n_episodes)
        np.testing.assert_array_equal(key,np.asarray(self._key))
        after=np.random.get_state();np.testing.assert_array_equal(rng[1],after[1]);assert rng[2:]==after[2:]
        if self.bank_digest is None:
            h=hashlib.sha256()
            for leaf in jax.tree.leaves((self._fixed_eval_init_states,self._fixed_eval_step_keys)):
                a=np.asarray(leaf);h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
            self.bank_digest=h.hexdigest()
        result.update(env_steps=self.total_steps,blend=self.buffer.blend,
            terminal_start_transitions=self.ghosts,valid_terminal_events=self.valid_ends,
            collector=row['collector'],schedule='decay' if row['decay'] else 'constant',
            eval_bank_hash=self.bank_digest,seconds=time.monotonic()-started)
        progress=dict(result,episode=self._current_episode,task=task,arm=row['arm'],seed=row['seed'],device=device)
        if hasattr(self,'metrics_logger'):
            (Path(self.metrics_logger.dir)/'progress.json').write_text(json.dumps(progress))
        print('PROGRESS '+json.dumps(progress),flush=True)
        return result

cfg=row['config'].copy()
agent=Agent(cfg)
assert agent.config==cfg,'Resolved config changed'
print('CONFIG: '+json.dumps(cfg),flush=True)
print('EXPERIMENT: '+json.dumps(row),flush=True)
print('DEVICE: '+device,flush=True)
run.mkdir(parents=True)
(run/'provenance.json').write_text(json.dumps(dict(row=row,device=device,manifest=str(work/'manifest.json')),indent=2))
agent.learn(verbose=False)
assert len(agent.episode_lengths)>=cfg['n_episodes']
assert np.isfinite(agent.buffer.consequence_scores[:len(agent.buffer)]).all()
qrows=[json.loads(l) for l in (run/'qdiag.jsonl').read_text().splitlines()]
assert qrows and all(r.get('loss') is not None and np.isfinite(r['loss']) for r in qrows)
if row['collector']=='fixed':assert 32<max(agent.episode_lengths)<=200
else:assert max(agent.episode_lengths)<=32
if row['decay']:assert {'zero','middle','one'}<=agent.blends_seen
record=dict(passed=True,task=task,run=str(run),seconds=time.monotonic()-started,
    episode=len(agent.episode_lengths),steps=agent.total_steps,bank=agent.bank_digest,
    blends=sorted(agent.blends_seen),max_length=max(agent.episode_lengths),ghosts=agent.ghosts)
(run/'complete.json').write_text(json.dumps(record,indent=2))
(work/f'complete-{task}.json').write_text(json.dumps(record,indent=2))
print('COMPLETE '+json.dumps(record),flush=True)
