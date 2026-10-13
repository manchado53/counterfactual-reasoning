"""Run-record adapter; the source learner owns every update and random draw."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .transfer import Trainer


class Agent(Trainer):
    def __init__(self, config):
        self.config = config
        super().__init__(config['arm'], config['seed'], n_envs=config['n_envs'],
                         capacity=config['buffer_capacity'])

    def evaluation_identity(self):
        spec = dict(seed=20260923, roles=[0, 1] * 500, start='empty',
                    opponent=self.config['opponent'], ties=self.config['opponent_ties'],
                    learner_ties='lowest-legal')
        return dict(eval_bank_hash=hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
                    eval_unique_cases=2, eval_independent_trials=False)

    def learn(self, verbose=False):
        np.random.seed(self.config['seed'])
        out = Path(os.environ['CCE_RUN_DIR'])
        for block in range(1, self.config['blocks'] + 1):
            self.learn_block(self.config['collect_steps'])
            with (out / 'progress.jsonl').open('a') as f:
                f.write(json.dumps(dict(self.diagnostics(), block=block), allow_nan=False) + '\n')
            if block % self.config['eval_every_blocks'] == 0 or block == self.config['blocks']:
                with (out / 'eval.jsonl').open('a') as f:
                    f.write(json.dumps(dict(self.evaluate(), block=block, env_steps=self.steps), allow_nan=False) + '\n')
                self.save(out / 'last.pkl')
        return self
