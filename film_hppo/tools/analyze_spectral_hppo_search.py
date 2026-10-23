"""Re-evaluate HPPO-generated candidates only; no gradient or design search."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from film_rl.config import load_config
from film_rl.optics import FilmOptics
from film_rl.design import compact_design


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    cfg=load_config(root/'evaluation_config.json')
    torch.set_num_threads(1)
    optics=FilmOptics(cfg)
    output=root/'dense_validation';output.mkdir(exist_ok=True)
    cfg_digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    rows=[];best_report=None;best_row=None
    sources=[('previous_pure_hppo',None,[root/'previous_pure_hppo_best.json'])]
    for batch in plan['batches']:
        for entry in batch['manifest']['runs']:
            sources.append((f"{batch['label']}_seed{entry['seed']}",entry,list((Path(entry['run_dir'])/'designs').glob('*.json'))))
    # Extra candidates may come from ordinary inference of these same HPPO
    # checkpoints. They are evaluated on the identical original dense grid.
    inference={}
    for summary_path in (root/'policy_evaluation').glob('*/summary.json'):
        sampled=json.loads(summary_path.read_text(encoding='utf-8'))
        label=sampled['experiment']+'_inference'
        inference[label]=sampled
        sources.append((label,None,[Path(sampled['best_sampled_design'])]))
    for label,entry,paths in sources:
        candidates=[]
        for path in paths:
            raw=path.read_bytes();d=json.loads(raw.decode('utf-8-sig'))
            if any(x['material']!=optics.names[x['material_id']] for x in d['layers']):
                raise ValueError(f'Material mismatch: {path}')
            key=hashlib.sha256(raw+cfg_digest.encode()).hexdigest()
            cached=output/'cache'/f'{key}.json'
            if cached.exists():
                summary=json.loads(cached.read_text(encoding='utf-8'))
            else:
                report=optics.evaluate([(x['material_id'],x['thickness_nm']) for x in d['layers']])
                summary=compact_design(report);write(cached,summary)
            candidates.append((summary['score'],path,summary))
        if not candidates:continue
        _,path,d=max(candidates,key=lambda x:x[0])
        record=json.loads(Path(entry['record']).read_text(encoding='utf-8')) if entry else {}
        row={'experiment':label,'status':record.get('status','reference'),
             'score':d['score'],'mae':d['mae'],'layers':d['layer_count'],
             'UV_R':d['target_errors'][0]['mean_R'],'visible_R':d['target_errors'][1]['mean_R'],
             'visible_T':d['target_errors'][1]['mean_T'],'visible_A':d['target_errors'][1]['mean_A'],
             'IR_R':d['target_errors'][2]['mean_R'],'source':str(path.resolve()),
             'run_dir':entry['run_dir'] if entry else None,'training_config':entry['config'] if entry else None,
             'env_steps':record.get('result',{}).get('env_step'),'candidates_recomputed':len(candidates)}
        if label in inference:
            sampled=inference[label]
            row.update(status='inference_complete',run_dir=sampled['run_dir'],
                       training_config=sampled['training_config'])
        rows.append(row);write(output/f'{label}.json',{**d,'source':str(path.resolve())})
        if best_row is None or row['score']>best_row['score']:
            best_row=row;best_report=d
    if not rows:raise ValueError('No candidates')
    write(output/'comparison.json',{'results':rows,'evaluation_config_sha256':cfg_digest})
    write(root/'best_design.json',{**best_report,'origin':best_row['source'],'method':'Pure HPPO; no design post-optimization'})
    write(root/'best_result.json',best_row)
    with (root/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Recompute only the winning case for plotted per-wavelength spectra.
    report=optics.evaluate([(x['material_id'],x['thickness_nm']) for x in best_report['layers']])
    np.savez_compressed(root/'best_spectra.npz',wavelengths_nm=optics.wavelengths_nm,angles_deg=optics.angles_deg,
                        **{q:np.asarray(report[q]) for q in 'RTA'})
    previous=json.loads((root/'previous_pure_hppo_best.json').read_text(encoding='utf-8'))
    old=optics.evaluate([(x['material_id'],x['thickness_nm']) for x in previous['layers']])
    reference=PROJECT/'outputs/references/oml_ppo_task2/figure7_omlppo_f1_vector.csv'
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    wave=optics.wavelengths_nm
    ax.plot(wave,np.asarray(old['R']).mean(0),label='Previous pure HPPO best',alpha=.8)
    ax.plot(wave,np.asarray(report['R']).mean(0),label=f"Best pure HPPO ({best_row['experiment']})")
    if reference.exists():
        curve=np.loadtxt(reference,delimiter=',',skiprows=1)
        ax.plot(curve[:,0],curve[:,1],':',color='black',label='Paper Fig.7 (plot-derived)')
    ax.plot(wave,np.where((wave>=480)&(wave<=700),0.,1.),'k--',lw=1,label='Target R')
    ax.axvspan(480,700,color='green',alpha=.1);ax.set(xlabel='Wavelength (nm)',ylabel='Uniform 0-89 deg mean R',ylim=(-.02,1.02))
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.savefig(root/'reflection_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    labels=[r['experiment'] for r in rows]
    axes[0].bar(labels,[r['mae'] for r in rows]);axes[0].set(ylabel='Weighted spectral MAE',ylim=(0,1),title='Lower is better')
    axes[1].bar(labels,[r['layers'] for r in rows]);axes[1].set(ylabel='Layers',title='Layer count is not a quality metric')
    for ax in axes:ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.2)
    fig.savefig(root/'comparison.png',dpi=170);plt.close(fig)
    lines=['# 纯HPPO光谱误差实验','',
           '目标保持R：300–470 nm为1，480–700 nm为0，710–3000 nm为1；权重1:4:3。',
           '全部候选按90角度、271波长、s/p平均复核。无improve、厚度精调、论文结构训练起点或模仿学习。','',
           '| 实验 | 状态 | 加权MAE↓ | score↑ | UV R | 可见R | 可见T | 可见A | IR R | 层数 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['experiment']} | {r['status']} | {r['mae']:.6f} | {r['score']:.6f} | {r['UV_R']:.2%} | {r['visible_R']:.2%} | {r['visible_T']:.2%} | {r['visible_A']:.2%} | {r['IR_R']:.2%} | {r['layers']} |")
    lines+=['',f"当前最好：{best_row['experiment']}，MAE={best_row['mae']:.8f}，score={best_row['score']:.8f}。",
            '训练中使用18角度近似网格，最终使用90角度；不是对训练所得厚度再优化。',
            'max_layers=80、min_layers=1、材料/厚度/目标/权重不变。不同seed反映训练随机性，单次最好不等于稳定成功率。',
            'best_design.json保存完整结构；best_result.json保存来源和run目录；训练曲线由原有批量脚本生成。',
            '若状态仍为training或pending，本报告仅是中间结果。',
            f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。"]
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'best':best_row,'completed':sum(r['status']=='complete' for r in rows),'total_cases':len(rows)-1},indent=2))


if __name__=='__main__':main()
