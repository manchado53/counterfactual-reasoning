"""Report observed runs only; never fill missing checkpoints or failed tails."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
work = ROOT/'work/value-score-replay'
records = []
for p in sorted((work/'status').glob('*.json')):
    status = json.loads(p.read_text())
    if status['smoke']:
        continue
    log = Path(status['run'])/'study_eval.jsonl'
    points = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    records.append(dict(status, observed_checkpoints=len(points), last=points[-1] if points else None))
print(json.dumps(records, indent=2))
if '--plot' in sys.argv and records:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row in records:
        log=Path(row['run'])/'study_eval.jsonl'
        if not log.exists():continue
        points=[json.loads(l) for l in log.read_text().splitlines()]
        i=0 if row['environment']=='frozenlake' else 1
        for j,x in enumerate(('env_steps','seconds')):
            axes[i,j].plot([p[x] for p in points],[p['win_rate'] for p in points],
                           label=f"{row['arm']} seed {row['seed']} ({row['state']})",
                           color='#1b9e77' if row['arm']=='value' else '#7570b3',alpha=.65)
            axes[i,j].set(title=row['environment'],xlabel=x,ylabel='success rate',ylim=(0,1))
    for ax in axes.flat:
        ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.suptitle('Observed checkpoints only; three-seed pilot')
    fig.tight_layout(); fig.savefig(work/'learning-curves.png',dpi=160)
