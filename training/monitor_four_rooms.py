"""Durable, quiet monitor: failures, quarter-budget milestones, completion only.

Writes status atomically on each poll and events only when something changes.
It never cancels/requeues jobs or starts new training. Reports use observed data.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def scheduler_states(job):
    result=subprocess.run(['sacct','-j',str(job),'--format=JobID,State,ExitCode',
                           '--parsable2','--noheader'],capture_output=True,text=True,check=True)
    states={}
    for line in result.stdout.splitlines():
        fields=line.split('|')
        if len(fields)<3 or '_' not in fields[0] or '.' in fields[0]: continue
        suffix=fields[0].split('_')[-1]
        if suffix.isdigit(): states[int(suffix)]=dict(state=fields[1].split()[0].rstrip('+'),exit=fields[2])
    return states


def monitor(root,job,interval):
    from analysis.studies.four_rooms import read_jsonl,report
    base=root/'runs/four-rooms';dest=base/'monitor';dest.mkdir(parents=True,exist_ok=True)
    state_file=dest/'state.json'
    saved=json.loads(state_file.read_text()) if state_file.exists() else {}
    seen=set(saved.get('seen',[]))
    def event(key,kind,detail):
        if key in seen: return
        seen.add(key)
        entry=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind=kind,detail=detail)
        with (dest/'events.jsonl').open('a') as f: f.write(json.dumps(entry)+'\n')
        print(json.dumps(entry),flush=True)
    while True:
        try:
            sched=scheduler_states(job)
            progress={}; attempts={}
            for task in range(15):
                folder=base/'full'/f'task_{task}'
                logs=read_jsonl(folder/'progress.jsonl')
                progress[task]=logs[-1]['env_steps'] if logs else 0
                path=folder/'attempt.json'
                if path.exists():
                    try: attempts[task]=json.loads(path.read_text())
                    except json.JSONDecodeError: continue
                rec=attempts.get(task,{})
                status=rec.get('status')
                scheduler=sched.get(task,{}).get('state','UNKNOWN')
                if status=='failed' or scheduler in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}:
                    kind='startup_failure' if progress[task]==0 else 'job_failure'
                    event(f'failure-{task}',kind,dict(task=task,scheduler=scheduler,error=rec.get('error')))
            fraction=sum(progress.values())/(15*524288)
            changed=False
            for percent in (25,50,75,100):
                key=f'milestone-{percent}'
                if fraction>=percent/100 and key not in seen:
                    event(key,'progress',dict(percent=percent,steps=progress));changed=True
            terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}
            finished=len(sched)==15 and all(x['state'] in terminal for x in sched.values())
            if changed or finished:
                report(base/'full',base/'report')
            if finished:
                completed=sum(x.get('status')=='completed' for x in attempts.values())
                event('finished','completion',dict(completed=completed,total=15,report=str(base/'report')))
            tmp=state_file.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(pid=os.getpid(),job=job,seen=sorted(seen),
                progress=progress,scheduler=sched,finished=finished),indent=2));tmp.replace(state_file)
            if finished: return
        except Exception as exc:
            event('monitor-error-'+str(exc),'monitor_failure',str(exc))
        time.sleep(interval)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--job',type=int,required=True);p.add_argument('--interval',type=int,default=120)
    a=p.parse_args();monitor(a.root,a.job,a.interval)
