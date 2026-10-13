"""Run-record adapter around the independently pinned fixed transfer learner."""
import json
import os
from pathlib import Path
import numpy as np
from .train import Trainer


class Agent(Trainer):
    def __init__(self, config):
        super().__init__(config['arm'], config['seed'], config=config)

    def evaluation_identity(self):
        return dict(eval_bank_hash=self.eval_bank_hash, eval_unique_cases=1,
                    eval_independent_trials=False)

    def learn(self, verbose=False):
        np.random.seed(self.config['seed'])
        out = Path(os.environ['CCE_RUN_DIR'])
        for block in range(1, self.config['blocks'] + 1):
            row = dict(self.train_block(), block=block)
            with (out / 'progress.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            if block % self.config['eval_every_blocks'] == 0 or block == self.config['blocks']:
                with (out / 'eval.jsonl').open('a') as f:
                    f.write(json.dumps(dict(self.evaluate(), block=block, env_steps=self.steps), allow_nan=False) + '\n')
                self.save(out / 'last.pkl')
        return self
