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
        if self.config['training_budget_kind'] == 'episodes':
            return self._learn_episode_budget(out)
        for block in range(1, self.config['blocks'] + 1):
            row = dict(self.train_block(), block=block)
            with (out / 'progress.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            if block % self.config['eval_every_blocks'] == 0 or block == self.config['blocks']:
                with (out / 'eval.jsonl').open('a') as f:
                    f.write(json.dumps(dict(self.evaluate(), block=block, env_steps=self.steps), allow_nan=False) + '\n')
                self.save(out / 'last.pkl')
        return self

    def _learn_episode_budget(self, out):
        target = self.config['n_episodes']
        next_eval = self.config['eval_interval']
        next_save = self.config['save_every']
        block = 0
        checkpoints = []
        while self.episodes < target:
            block += 1
            row = dict(self.train_block(), block=block)
            with (out / 'progress.jsonl').open('a') as handle:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
            if self.episodes >= next_eval or self.episodes >= target:
                evaluation = dict(
                    self.evaluate(),
                    block=block,
                    episode=self.episodes,
                    env_steps=self.steps,
                    decay_blend=(self.buffer.blend if self.arm == 'cce_decay' else None),
                )
                with (out / 'eval.jsonl').open('a') as handle:
                    handle.write(json.dumps(evaluation, allow_nan=False) + '\n')
                with (out / 'replay.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(
                        self.diagnostics(),
                        block=block,
                        episode=self.episodes,
                        env_steps=self.steps,
                    ), allow_nan=False) + '\n')
                self.save(out / 'last.pkl')
                while next_eval <= self.episodes:
                    next_eval += self.config['eval_interval']
            if self.episodes >= next_save:
                checkpoint = out / f'checkpoint_{self.episodes:09d}.pkl'
                self.save(checkpoint)
                checkpoints.append(checkpoint)
                while len(checkpoints) > self.config['n_checkpoints']:
                    checkpoints.pop(0).unlink()
                while next_save <= self.episodes:
                    next_save += self.config['save_every']
        return self
