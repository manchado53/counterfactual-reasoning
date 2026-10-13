"""Add step-indexed eval records; leave the source's chunk schedule untouched."""
import json
import math
import os
from pathlib import Path
import numpy as np
from .dqn import Connect4DQN
from .consequence_dqn import Connect4ConsequenceDQN


class Records:
    def learn(self,*args,**kwargs):
        # The original CLI seeds only JAX. Record, rather than replace, the
        # inherited global NumPy state used by its replay buffer.
        state = np.random.get_state()
        (Path(os.environ['CCE_RUN_DIR'])/'numpy_initial_state.json').write_text(json.dumps(dict(
            kind=state[0],keys=state[1].tolist(),position=state[2],
            has_gauss=state[3],cached_gaussian=state[4]))+'\n')
        self._record_evals = True
        try:
            return super().learn(*args,**kwargs)
        finally:
            self._record_evals = False

    def evaluate(self,*args,**kwargs):
        metrics = super().evaluate(*args,**kwargs)
        if getattr(self,'_record_evals',False):
            # Missing role cases remain NaN in the original result/logger. JSON
            # uses null solely for serialization, with no change to training.
            row = {k:(None if isinstance(v,float) and not math.isfinite(v) else v)
                   for k,v in metrics.items()}
            row.update(env_steps=self.total_steps,source_chunk=self._current_episode+1,
                       eval_seed=kwargs.get('seed',42),opponent=self.config['opponent'])
            with (Path(os.environ['CCE_RUN_DIR'])/'eval.jsonl').open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n')
        return metrics


class DQN(Records,Connect4DQN):
    pass


class CCE(Records,Connect4ConsequenceDQN):
    pass
