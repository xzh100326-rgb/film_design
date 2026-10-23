"""Configuration-only HPPO comparisons, using the project's existing batch runner."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from run_seed_batch import prepare, run_one, refresh_summary
from film_rl.config import load_config


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=60000)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--supplement', action='store_true', help='Single-seed entropy/minimum-layer diagnostics after the initial six runs')
    parser.add_argument('--final-sweep', action='store_true', help='Final bounded sweep: entropy .01, minimum two layers, or learning rate 3e-4')
    args = parser.parse_args()
    if args.supplement and args.final_sweep:
        parser.error('Choose one stage')
    root = args.root.resolve()
    extra_stage = args.supplement or args.final_sweep
    if not extra_stage and (root/'plan.json').exists():
        raise FileExistsError('This experiment already has a plan; preserve it and use a new root for a new experiment')
    previous = json.loads((root/'plan.json').read_text(encoding='utf-8')) if extra_stage else None
    previous_records = [Path(e['record']) for b in previous['batches'] for e in b['manifest']['runs']] if previous else []
    cfg = load_config(root/'evaluation_config.json')
    cfg['angles_deg'] = list(range(2,90,5))
    cfg['_reference']['angle_grid_note'] = ('Training: centers of 18 consecutive five-degree bins of the original '
                                            '0..89 degree grid. All saved designs are re-evaluated on 0..89 degrees.')
    cfg['_reference']['training_note'] = ('Fresh native HPPO/VAC; terminal linear reward; '
                                          'no thickness optimizer, imitation, architecture or loss changes.')
    batches = []
    variants = ([dict(label='e5_h0p003', epochs=5, entropy=.003, minimum=1),
                 dict(label='e5_ir6', epochs=5, entropy=.001, minimum=1, infrared_weight=6),
                 dict(label='e5_min4', epochs=5, entropy=.001, minimum=4)] if args.supplement else
                [dict(label=f'e{e}', epochs=e, entropy=.001, minimum=1) for e in (10,5)])
    if args.final_sweep:
        variants = [dict(label='e5_h0p01', epochs=5, entropy=.01, minimum=1),
                    dict(label='e5_min2', epochs=5, entropy=.001, minimum=2),
                    dict(label='e5_lr3em4', epochs=5, entropy=.001, minimum=1, learning_rate=.0003)]
    if previous:
        existing={b.get('label','e'+str(b['epochs'])) for b in previous['batches']}
        if any(v['label'] in existing for v in variants):
            raise FileExistsError('This stage is already prepared; existing configurations/results are preserved')
    seeds = [0] if extra_stage else [0,1,2]
    for variant in variants:
        epochs = variant['epochs']
        trial = deepcopy(cfg)
        trial['min_layers'] = variant['minimum']
        if 'infrared_weight' in variant:
            trial['spectral_targets'][2]['weight'] = variant['infrared_weight']
            trial['_reference']['objective_note'] = 'Training band weights 1:4:6; final comparison uses the original application weights 1:4:3.'
        trial['training'].update(epoch_per_collect=epochs, max_env_step=args.budget,
                                  max_train_iter=1000000, eval_freq=24*epochs,
                                  entropy_weight=variant['entropy'])
        if 'learning_rate' in variant:
            trial['training']['learning_rate'] = variant['learning_rate']
        trial['training'].pop('initial_model', None)
        path = root/'configs'/f'transmission_{variant["label"]}.json'
        write(path, trial)
        batch = prepare(path, group=f'incandescent_T_{variant["label"]}', seeds=seeds)
        manifest = json.loads((batch/'manifest.json').read_text(encoding='utf-8'))
        batches.append(dict(**variant, batch=str(batch), manifest=manifest))
    plan = dict(algorithm='Unchanged FilmPPOPolicy + FilmVAC', budget_env_steps=args.budget,
                seeds=seeds, workers=args.workers,
                training_angles=cfg['angles_deg'], validation_angles=list(range(90)),
                objectives=cfg['spectral_targets'], batches=batches)
    if extra_stage:
        previous['batches'].extend(batches)
        previous['supplement_note'] = ('Single-seed diagnostics: existing entropy_weight, min_layers, infrared target weight, '
                                       'and learning_rate; all stages use fresh networks. No architecture or PPO loss changes.')
        write(root/'plan.json', previous)
    else:
        write(root/'plan.json', plan)
    started = time.perf_counter()
    # Interleave variants to avoid comparing runs started at very different times.
    jobs = [(item, entry) for seed in (0,1,2) for item in batches
            for entry in item['manifest']['runs'] if entry['seed']==seed]
    lock = threading.Lock()
    active = 0

    def limited_run(item, entry):
        nonlocal active
        while True:
            with lock:
                older_remaining = sum(
                    not path.exists() or json.loads(path.read_text(encoding='utf-8'))['status'] not in ('complete','failed')
                    for path in previous_records)
                if older_remaining + active < args.workers:
                    active += 1
                    break
            time.sleep(5)
        try:
            return run_one(entry, Path(item['batch']), item['manifest']['runtime_environment'])
        finally:
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(limited_run, item, entry) for item, entry in jobs}
        while futures:
            done, futures = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            for item in batches:
                refresh_summary(Path(item['batch']), item['manifest'], args.workers, started)
    print('TRIALS_COMPLETE='+str(root), flush=True)


if __name__ == '__main__':
    main()
