"""Bounded saved-task smoke; use shared runner after cvrp_budget registration."""
import argparse
import json
import os
from pathlib import Path
import signal
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=['budget', 'option_a'])
    args = parser.parse_args()
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError('150-second smoke cap')))
    signal.alarm(150)
    from experiments.learners.cvrp_budget.build import build_agent, smoke_overrides
    rows = json.loads(Path('configs/experiment/cvrp_budget-logged-settings.json').read_text())['rows']
    row = next(r for r in rows if r['config']['learner_variant'] == args.variant)
    cfg = {**row['config'], **smoke_overrides(row['config'])}
    out = Path(tempfile.mkdtemp(prefix=f'cvrp-budget-{args.variant}-'))
    os.environ['CCE_RUN_DIR'] = str(out)
    record = dict(status='starting', smoke=True, task=row['task'], variant=args.variant,
                  source_header_sha256=row['source_header_sha256'], config=cfg,
                  source_manifest=row['port_source_manifest'], output=str(out))
    try:
        agent = build_agent(cfg)
        assert agent.config == cfg
        agent.learn(verbose=False)
        record.update(status='completed', evaluation=agent.evaluate(1))
    except BaseException as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        signal.alarm(0)
        (out / 'attempt.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(status=record['status'], output=str(out))), flush=True)


if __name__ == '__main__':
    main()
