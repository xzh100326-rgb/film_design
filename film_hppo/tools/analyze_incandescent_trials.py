"""Validate every saved trial design on the unchanged dense optical grid."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from film_rl.config import load_config
from film_rl.design import compact_design
from film_rl.optics import FilmOptics

torch.set_num_threads(1)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def evaluate_paths(optics, paths, tag, output):
    candidates = []
    for path in paths:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        layers = [(layer['material_id'],layer['thickness_nm']) for layer in data['layers']]
        if any(layer.get('material',optics.names[layer['material_id']]) != optics.names[layer['material_id']]
               for layer in data['layers']):
            raise ValueError(f'Material ordering mismatch: {path}')
        report = optics.evaluate(layers)
        candidates.append((report['score'],path,report))
    if not candidates:
        return None
    _, path, report = max(candidates, key=lambda item:item[0])
    summary = compact_design(report)
    summary['origin'] = str(path)
    write(output/f'{tag}.json', summary)
    np.savez_compressed(output/f'{tag}.npz', wavelengths_nm=optics.wavelengths_nm,
                        angles_deg=optics.angles_deg, **{q:np.asarray(report[q]) for q in 'RTA'})
    bands = summary['target_errors']
    return dict(experiment=tag, score=report['score'], layers=report['layer_count'],
                thickness_nm=report['total_thickness_nm'], UV_R=bands[0]['mean_R'],
                visible_R=bands[1]['mean_R'], visible_T=bands[1]['mean_T'],
                visible_A=bands[1]['mean_A'], IR_R=bands[2]['mean_R'],
                source=str(path), dense_design=str(output/f'{tag}.json'),
                candidates_recomputed=len(candidates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root/'validation'/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(root/'evaluation_config.json')
    optics = FilmOptics(cfg)
    rows = []
    reference = PROJECT/'outputs/references/oml_ppo_task2/table3_current_materials/input_design.json'
    rows.append(evaluate_paths(optics,[reference],'paper42',output))
    old = PROJECT/'runs/incandescent_task/20261020_212346'
    old_paths = list(old.glob('seed*/designs/*.json'))
    old_original_path = max(old_paths, key=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))['score'])
    old_original = evaluate_paths(optics,[old_original_path],'old_R_objective_winner',output)
    rows.append(old_original)
    for seed in range(10):
        row = evaluate_paths(optics, sorted((old/f'seed{seed}'/'designs').glob('*.json')),
                             f'old_seed{seed}', output)
        if row:
            rows.append(row)
    plan = json.loads((root/'plan.json').read_text(encoding='utf-8'))
    complete = True
    for batch in plan['batches']:
        for entry in batch['manifest']['runs']:
            record = Path(entry['record'])
            state = json.loads(record.read_text(encoding='utf-8')) if record.exists() else {'status':'pending'}
            status = state['status']
            complete = complete and status == 'complete'
            row = evaluate_paths(optics, sorted((Path(entry['run_dir'])/'designs').glob('*.json')),
                                 f'{batch.get("label", "e"+str(batch["epochs"]))}_seed{entry["seed"]}', output)
            if row:
                row['status'] = status
                trial_cfg = load_config(entry['config'])
                row.update(training_config=entry['config'],
                           training_min_layers=trial_cfg['min_layers'],
                           training_epoch_per_collect=trial_cfg['training']['epoch_per_collect'],
                           training_entropy_weight=trial_cfg['training']['entropy_weight'],
                           training_IR_weight=trial_cfg['spectral_targets'][2]['weight'],
                           actual_env_steps=state.get('result',{}).get('env_step'),
                           optimizer_updates=state.get('result',{}).get('train_iter'),
                           training_seconds=state.get('training_seconds'),
                           last_evaluation_mean=state.get('result',{}).get('eval_value'))
                rows.append(row)
    write(output/'comparison.json', dict(all_complete=complete, evaluation_config=cfg, results=rows))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    reference = rows[0]
    old_best = max((r for r in rows if r['experiment'].startswith('old_')), key=lambda r:r['score'])
    trials = [r for r in rows if r['experiment'].startswith('e')]
    group_statistics = []
    for label in sorted({r['experiment'].rsplit('_seed',1)[0] for r in trials}):
        values = [r for r in trials if r['experiment'].rsplit('_seed',1)[0] == label]
        group_statistics.append(dict(group=label, seeds=len(values),
                                     complete_seeds=sum(r['status']=='complete' for r in values),
                                     mean_score=float(np.mean([r['score'] for r in values])),
                                     min_score=min(r['score'] for r in values),
                                     max_score=max(r['score'] for r in values)))
    write(output/'group_statistics.json',group_statistics)
    plot_rows = [reference, old_original, old_best]+trials
    def short_label(name):
        if name=='paper42':return 'Paper 42'
        if name=='old_R_objective_winner':return 'Old R-best'
        if name.startswith('old_'):return 'Old T-best'
        group,seed=name.rsplit('_seed',1)
        group={'e5_h0p003':'H.003','e5_h0p01':'H.01','e5_min4':'Min4',
               'e5_min2':'Min2','e5_ir6':'IR6','e5_lr3em4':'LR3e-4'}.get(group,group.upper())
        return f'{group}/s{seed}'
    labels = [short_label(r['experiment']) for r in plot_rows]
    fig, axes = plt.subplots(1,3,figsize=(15,4.8))
    colors=['#888888','#bd7094','#b39ddb']+['#2f77b5' if r['experiment'].startswith('e10') else '#db7f20' for r in trials]
    axes[0].bar(labels,[r['score'] for r in plot_rows],color=colors)
    axes[0].set_title('Dense-grid target score');axes[0].set_ylim(0,1)
    x=np.arange(len(labels));width=.36
    axes[1].bar(x-width/2,[r['visible_T'] for r in plot_rows],width,label='Visible T')
    axes[1].bar(x+width/2,[r['IR_R'] for r in plot_rows],width,label='Infrared R')
    axes[1].set_xticks(x,labels);axes[1].set_ylim(0,1);axes[1].legend()
    axes[1].set_title('Application metrics: uniform angle means')
    axes[2].bar(labels,[r['layers'] for r in plot_rows],color=colors)
    axes[2].set_title('Layers in best saved design');axes[2].set_ylim(0,48)
    for ax in axes:
        ax.tick_params(axis='x',rotation=65);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=170);plt.close(fig)
    best=max(trials,key=lambda r:r['score']) if trials else None
    if best:
        fig, axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
        for row,color,style,label in [(best,'#1976b9','-',best['experiment']),
                                      (reference,'#555555','--','Paper42, current n/k'),
                                      (old_best,'#a057a8',':','Previous best under T target')]:
            arrays=np.load(output/f'{row["experiment"]}.npz')
            wave=arrays['wavelengths_nm']
            for ax,q in zip(axes,'RTA'):
                ax.plot(wave,arrays[q].mean(axis=0),color=color,linestyle=style,label=label)
        for ax,q in zip(axes,'RTA'):
            ax.axvspan(480,700,color='#81c784',alpha=.17)
            ax.set_ylabel(q);ax.set_ylim(-.02,1.02);ax.grid(alpha=.2)
        axes[0].plot(wave,np.where((wave<=470)|(wave>=710),1,np.nan),'k-.',lw=1,label='Target R')
        axes[1].plot(wave,np.where((wave>=480)&(wave<=700),1,np.nan),'k-.',lw=1,label='Target T')
        axes[0].legend(loc='lower right',fontsize=8)
        axes[-1].set_xlabel('Wavelength (nm)')
        axes[0].set_title('Dense validation: uniform 0-89 degree mean, unpolarized')
        fig.tight_layout();fig.savefig(output/'spectra_comparison.png',dpi=170);plt.close(fig)
    report = ['# HPPO configuration experiment', '',
              'No network, PPO loss, material data, thickness bounds, or reward timing changes.',
              'Visible target changed from R=0 to T=1. All comparison scores use weights 1:4:3, including the separate IR-weight-6 training experiment.',
              'Training uses 18 angle-bin centers; all results below were recomputed on the original 90 angles and 271 wavelengths.',
              'Comparison score = (UV mean R + 4 * visible mean T + 3 * infrared mean R) / 8.',
              'No thickness optimizer, structure search, demonstration or warm-start model was used.', '',
              f'All runs complete: {complete}', '',
              '| Experiment | Dense score | Visible T | Visible A | IR R | Layers |',
              '|---|---:|---:|---:|---:|---:|']
    for row in plot_rows:
        report.append(f'| {row["experiment"]} | {row["score"]:.6f} | {row["visible_T"]:.2%} | '
                      f'{row["visible_A"]:.2%} | {row["IR_R"]:.2%} | {row["layers"]} |')
    report += ['', 'Paper42 and previous designs are references only, never training inputs.',
               'A high visible T alone is insufficient if infrared reflection falls; use the combined score and band metrics.',
               'These experiments do not establish a global optimum or the paper luminous enhancement factor.']
    (output/'README.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    write(root/'latest_validation.json', dict(directory=str(output), all_complete=complete, best_trial=best,
                                            best_previous_under_new_objective=old_best))
    print(json.dumps(dict(output=str(output),complete=complete,best=best,previous=old_best),indent=2))


if __name__=='__main__':
    main()
