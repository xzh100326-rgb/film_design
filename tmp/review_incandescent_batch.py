"""Read-only training snapshot; writes diagnostics, never changes a run."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

project = Path(r'E:\lithography_machine_lens\film_hppo')
batch = project / 'outputs/incandescent_task/20261020_212346'
manifest = json.loads((batch/'manifest.json').read_text())
stamp = datetime.datetime.now().astimezone()
out = batch/'diagnostics'/stamp.strftime('snapshot_%Y%m%d_%H%M%S')
out.mkdir(parents=True, exist_ok=False)
rows = []
panels = [('evaluator_iter/reward_mean', 'Evaluation mean score'),
          ('evaluator_iter/reward_max', 'Best score in each evaluation'),
          ('collector_iter/avg_envstep_per_episode', 'Collection episode steps (includes EOS)'),
          ('learner_iter/entropy_loss_avg', 'Logged hybrid entropy')]
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
for entry in manifest['runs']:
    status_path = Path(entry['record'])
    status = json.loads(status_path.read_text()) if status_path.exists() else entry
    row = {'seed': entry['seed'], 'status': status['status'], 'pid': status.get('training_pid')}
    run = Path(entry['run_dir'])
    designs = []
    for path in (run/'designs').glob('best_*.json'):
        report = json.loads(path.read_text(encoding='utf-8-sig'))
        bands = report['target_errors']
        assert abs(report['score']-(1-sum(t['normalized_weight']*t['mae'] for t in bands))) < 1e-12
        designs.append({'file': str(path), 'score': report['score'], 'layers': report['layer_count'],
                        'total_thickness_nm': report['total_thickness_nm'], 'format': report.get('design_format'),
                        'bands': bands, 'materials': [x['material'] for x in report['layers']]})
    row['designs'] = sorted(designs, key=lambda d: d['score'], reverse=True)
    events = run/'log/serial'
    if events.exists():
        acc = EventAccumulator(str(events), size_guidance={'scalars':0})
        acc.Reload()
        tags = acc.Tags()['scalars']
        row['metrics'] = {}
        for tag in tags:
            if '_iter/' not in tag:
                continue
            values = acc.Scalars(tag)
            assert all(math.isfinite(e.value) for e in values), (entry['seed'], tag)
            row['metrics'][tag] = [{'step':e.step, 'value':e.value, 'wall_time':e.wall_time} for e in values]
        for axis, (tag, title) in zip(axes.flat, panels):
            data = row['metrics'].get(tag, [])
            if data:
                axis.plot([x['step'] for x in data], [x['value'] for x in data], label=f'seed{entry["seed"]}')
        text = Path(entry['train_log']).read_text(encoding='utf-8', errors='replace')
        row['kl_early_stops'] = text.count('rollout reuse stopped')
        row['tracebacks'] = text.count('Traceback (most recent call last)')
        row['scalar_values_all_finite'] = True
    rows.append(row)
for axis, (_, title) in zip(axes.flat, panels):
    axis.set(title=title, xlabel='Logged learner iteration')
    axis.grid(alpha=.25)
    if axis.lines:
        axis.legend()
fig.suptitle('Incandescent Task 2 | weights 1:4:3 | incomplete batch snapshot')
fig.savefig(out/'training_snapshot.png', dpi=150)
plt.close(fig)
hash_unchanged = all((project/p).is_file() and hashlib.sha256((project/p).read_bytes()).hexdigest()==sha
                     for p,sha in manifest['source_hashes'].items())
snapshot = {'captured_at':stamp.isoformat(), 'batch':str(batch), 'source_files_unchanged':hash_unchanged,
            'all_complete':all(row['status']=='complete' for row in rows), 'runs':rows,
            'warning':'Incomplete training snapshot. Best saved design is not the average policy performance.'}
(out/'snapshot.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
print('OUTPUT',str(out))
for row in rows:
    if not row['designs']:
        continue
    metrics=row['metrics']
    def latest(tag):
        data=metrics.get(tag,[])
        return data[-1] if data else None
    d=row['designs'][0]
    print(json.dumps({'seed':row['seed'], 'best_score':d['score'], 'layers':d['layers'],
          'visible_R':d['bands'][1]['mean_R'], 'visible_T':d['bands'][1]['mean_T'],
          'visible_A':d['bands'][1]['mean_A'], 'IR_R':d['bands'][2]['mean_R'],
          'latest_eval':latest('evaluator_iter/reward_mean'),
          'latest_collection_steps':latest('collector_iter/avg_envstep_per_episode'),
          'latest_env_steps':latest('collector_iter/total_envstep_count'),
          'latest_kl':latest('learner_iter/approx_kl_avg'),
          'best_materials':d['materials'][:8]}))
