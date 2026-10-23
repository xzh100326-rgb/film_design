"""Bounded native HPPO experiments, physical-metric selection and dense validation.

No thickness optimizer, imitation, material changes or paper initialization.
Trial artifacts are intentionally separated so that only this job's temporary
directories can be removed after its decisions are recorded in PROCESS.md.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch
from film_rl.config import load_config, optical_data_files
from film_rl.hppo import build_model
from film_rl.improve import problem_fingerprint, layers_of
from film_rl.optics import FilmOptics
from film_rl.design import compact_design
from film_rl.lamp_metrics import lamp_metrics, lamp_key, dominates, IR_BANDS


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(40):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def split_targets():
    specs = [('R', 300., 470., 1., 1.), ('R', 480., 700., 0., 1.),
             ('T', 480., 700., 1., 7.)]
    specs += [('R', lo, hi, 1., weight) for (lo, hi), weight in
              zip(IR_BANDS, [2., 1.5, 1., .9, .6])]
    return [dict(quantity=q, wavelength_range_nm=[lo, hi], target=target, weight=weight)
            for q, lo, hi, target, weight in specs]


def trial(root, name, sigma, lr, split=False, steps=36000, source=None):
    cfg = load_config(PROJECT/'outputs/lamp_four_targets/20261021_195347/evaluation_config.json')
    cfg['angles_deg'] = list(range(2, 90, 5))
    cfg['design_selection'] = 'lamp_pareto'
    if split:
        cfg['spectral_targets'] = split_targets()
    else:
        cfg['spectral_targets'][2]['weight'] = 7.
    cfg['_reference'] = {'method': 'Native HPPO model-weight transfer; physical-metric Pareto selection',
        'note': 'Optimizer/RNG reset; entire VAC transferred, sigma explicitly reinitialized. No improve or imitation.',
        'visible_window_nm': [480, 700], 'split_infrared': split}
    run = PROJECT/'runs'/root.name/name
    folder = root/'trials'/name if name != 'final' else root/'final'
    folder.mkdir(parents=True, exist_ok=False)
    source = Path(source) if source else PROJECT/'runs/lamp_four_targets__narrow/20261021_202859/seed0/ckpt/iteration_4176.pth.tar'
    payload = torch.load(source, map_location='cpu', weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(payload['model'], strict=True)
    torch.save({'model': model.state_dict(), 'problem_fingerprint': problem_fingerprint(cfg),
                'source_checkpoint': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'method': 'Native HPPO transfer; no optimizer or RNG restored'}, folder/'initial_model.pt')
    cfg['training'].update(seed=19, initial_model=str(folder/'initial_model.pt'), initial_sigma=sigma,
        learning_rate=lr, n_sample=3000, batch_size=256, epoch_per_collect=3,
        gae_lambda=1., entropy_weight_discrete=.003, entropy_weight_continuous=0.,
        max_env_step=steps, max_train_iter=1000000, eval_freq=360,
        exp_name=str(run.relative_to(PROJECT)).replace('\\', '/'))
    config_path = folder/'training_config.json'
    write(config_path, cfg)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    start = time.perf_counter()
    with (folder/'training.log').open('w', encoding='utf-8') as stream:
        process = subprocess.Popen([sys.executable, '-B', '-m', 'film_rl', 'train', '--config',
                                    str(config_path), '--non-interactive'], cwd=PROJECT,
                                   env=env, stdout=stream, stderr=subprocess.STDOUT)
        write(folder/'status.json', dict(status='running', pid=process.pid, run=str(run), source=str(source)))
        code = process.wait()
    if code:
        write(folder/'status.json', dict(status='failed', exit_code=code))
        raise RuntimeError(f'{name} failed; inspect {folder / "training.log"}')
    records = []
    for path in (run/'designs').glob('*.json'):
        data = read(path)
        records += data if isinstance(data, list) else [data]
    # Re-evaluate all archived designs against a common fixed optical grid.
    optics = FilmOptics(cfg)
    candidates = []
    seen = set()
    for data in records:
        layers = layers_of(data)
        signature = tuple((i, round(d, 5)) for i, d in layers)
        if signature in seen:
            continue
        seen.add(signature)
        report = optics.evaluate(layers)
        report['application_metrics'] = lamp_metrics(report)
        candidates.append(report)
    candidates.sort(key=lambda r: lamp_key(r['application_metrics']), reverse=True)
    for rank, report in enumerate(candidates[:6]):
        write(folder/f'candidate_{rank}.json', compact_design(report))
    result = dict(name=name, run=str(run), folder=str(folder), sigma=sigma, lr=lr, split=split,
                  budget=steps, elapsed_seconds=time.perf_counter()-start,
                  best=candidates[0]['application_metrics'], best_score=candidates[0]['score'],
                  candidate_count=len(candidates), config=str(config_path))
    write(folder/'status.json', dict(status='complete', **result))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def final_checkpoint(run):
    return max((Path(run)/'ckpt').glob('iteration_*.pth.tar'),
               key=lambda p: int(p.name.split('_')[1].split('.')[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='lamp_physical_20261021')
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = PROJECT/'outputs'/args.name
    root.mkdir(parents=True, exist_ok=False)
    plan = {'status': 'screening', 'method': 'pure HPPO, model-weight continuation', 'stages': [],
            'cleanup_scope': [str(root/'trials'), str(PROJECT/'runs'/args.name)],
            'physical_targets': 'visible T>=.70,R<=.25,A<=.05; UV R>=.70; IR mean>=.85 and every subband>=.70',
            'note': 'Targets are aspirational band-mean acceptance criteria, not guaranteed pointwise performance.'}
    plan['material_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in optical_data_files(
        load_config(PROJECT/'outputs/lamp_four_targets/20261021_195347/evaluation_config.json'))}
    write(root/'progress.json', plan)
    jobs = [('control_sigma08', .08, 3e-5), ('narrow_sigma03', .03, 1e-5)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(trial, root, *job) for job in jobs]
        for future in futures:
            plan['stages'].append(future.result())
            write(root/'progress.json', plan)
    # Previous nine trials and this bounded control are the stagnation audit.
    # A strict visible gain with sustained broadband reflection avoids rewarding
    # a nearly transparent film as a successful lamp filter.
    progress = any(r['best']['visible_T'] >= .38 and r['best']['IR_R'] >= .85
                   and r['best']['UV_R'] >= .70 for r in plan['stages'])
    if not progress:
        plan.update(status='split_band_screening', split_reason='No >=.38 visible T with IR>=.85 and UV>=.70 in controls')
        write(root/'progress.json', plan)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(trial, root, name, sigma, lr, True, 48000)
                       for name, sigma, lr in [('split_sigma04', .04, 3e-5), ('split_sigma08', .08, 3e-5)]]
            for future in futures:
                plan['stages'].append(future.result())
                write(root/'progress.json', plan)
    winner = max(plan['stages'], key=lambda row: lamp_key(row['best']))
    plan.update(status='final_training', selected_stage=winner['name'])
    write(root/'progress.json', plan)
    result = trial(root, 'final', winner['sigma'], winner['lr'], winner['split'], 180000,
                   final_checkpoint(winner['run']))
    plan['stages'].append(result)
    plan.update(status='training_complete_requires_dense_validation')
    write(root/'progress.json', plan)


if __name__ == '__main__':
    main()
