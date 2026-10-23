"""Read-only physics/timing checks before the HPPO configuration experiment."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from film_rl.config import load_config
from film_rl.design import compact_design
from film_rl.optics import FilmOptics

torch.set_num_threads(1)


def main():
    output = PROJECT / 'outputs/incandescent_hppo_experiments' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(PROJECT / 'configs/incandescent_task.json')
    cfg['spectral_targets'][1].update(quantity='T', target=1.0)
    cfg['_reference'].update(
        target='Application objective: visible T=1, ultraviolet and infrared R=1.',
        objective_note='Band weights 1:4:3, uniform means within each band. Not the paper enhancement factor.')
    (output/'evaluation_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    sources = {
        'paper42': PROJECT/'outputs/references/oml_ppo_task2/table3_current_materials/input_design.json',
        'old_seed0': PROJECT/'runs/incandescent_task/20261020_212346/seed0/designs/best_a24032d0f34548c4a86f586f7fd57ca3.json',
        'old_single': PROJECT/'runs/incandescent_task/20261020_212346/seed1/designs/best_da0e342b.json',
    }
    singles = list((PROJECT/'runs/incandescent_task/20261020_212346/seed1/designs').glob('*.json'))
    sources['old_single'] = max(singles, key=lambda p: json.loads(p.read_text(encoding='utf-8'))['score'])
    rows = []
    for tag, path in sources.items():
        design = json.loads(path.read_text(encoding='utf-8-sig'))
        layers = [(item['material_id'], item['thickness_nm']) for item in design['layers']]
        reference = None
        for mode in ['dense_cpu', 'dense_cuda', 'midpoint18_cpu']:
            settings = deepcopy(cfg)
            if mode == 'dense_cuda':
                settings['tmm_device'] = 'cuda'
            if mode == 'midpoint18_cpu':
                settings['angles_deg'] = list(range(2, 90, 5))
            optics = FilmOptics(settings)
            times = []
            for _ in range(3):
                start = time.perf_counter()
                report = optics.evaluate(layers)
                times.append(time.perf_counter()-start)
            if reference is None:
                reference = report
                (output/f'{tag}_dense.json').write_text(json.dumps(compact_design(report), indent=2), encoding='utf-8')
            row = dict(design=tag, source=str(path), mode=mode, layers=len(layers),
                       seconds=times, score=report['score'],
                       bands=[{k:b[k] for k in ['mean_R','mean_T','mean_A']} for b in compact_design(report)['target_errors']],
                       score_difference=report['score']-reference['score'])
            if mode == 'dense_cuda':
                row['max_spectral_difference'] = max(float(np.max(np.abs(np.asarray(report[q])-reference[q]))) for q in 'RTA')
                assert row['max_spectral_difference'] < 1e-8, row
            rows.append(row)
            print(json.dumps(row), flush=True)
    (output/'benchmark.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('EXPERIMENT_ROOT='+str(output), flush=True)


if __name__ == '__main__':
    main()
