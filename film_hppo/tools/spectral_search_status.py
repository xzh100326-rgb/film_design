"""Compact live status for the bounded HPPO spectral experiment."""
import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads((args.root/'plan.json').read_text(encoding='utf-8'))
    rows=[]
    for batch in plan['batches']:
        for entry in batch['manifest']['runs']:
            record=Path(entry['record'])
            state=json.loads(record.read_text(encoding='utf-8')) if record.exists() else {'status':'pending'}
            row={'variant':batch['label'],'seed':entry['seed'],'status':state['status']}
            run=Path(entry['run_dir'])
            designs=[]
            for path in (run/'designs').glob('*.json'):
                try:
                    d=json.loads(path.read_text(encoding='utf-8'))
                    designs.append((d['score'],d['layer_count']))
                except (ValueError,KeyError):pass
            if designs:row.update(best_score=max(designs)[0],best_layers=max(designs)[1])
            log=run/'log/serial'
            if log.exists():
                acc=EventAccumulator(str(log),size_guidance={'scalars':1});acc.Reload()
                tags=acc.Tags()['scalars']
                for target,pattern in [('steps','collector_iter/total_envstep_count'),('episode_steps','collector_iter/avg_envstep_per_episode'),
                                       ('episode_reward','collector_iter/reward_mean'),('eval_mean','evaluator_iter/reward_mean')]:
                    matches=[t for t in tags if t==pattern]
                    if matches:
                        values=acc.Scalars(matches[0])
                        if values:row[target]=values[-1].value
                row['optimizer_step']=max((acc.Scalars(t)[-1].step for t in tags if t.startswith('learner_iter/') and acc.Scalars(t)),default=0)
            rows.append(row)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
