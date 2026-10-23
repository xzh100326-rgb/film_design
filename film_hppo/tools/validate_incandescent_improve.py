"""Dense-grid validation of incandescent improve runs, with no PPO/VAC training."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from film_rl.config import load_config
from film_rl.design import compact_design
from film_rl.improve import layers_of, refine
from film_rl.optics import FilmOptics


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--iterations',type=int,default=80)
    parser.add_argument('--case-iterations',nargs='*',default=[],help='Optional case=iterations overrides for expensive long stacks')
    args=parser.parse_args()
    overrides={}
    for value in args.case_iterations:
        name,count=value.split('=',1)
        overrides[name]=int(count)
        if overrides[name]<1:parser.error('Iteration budgets must be positive')
    root=args.root.resolve()
    output=root/'dense_validation'
    output.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    cfg=load_config(root/'evaluation_config.json')
    optics=FilmOptics(cfg)
    baseline_path=PROJECT/'outputs/incandescent_hppo_experiments/20261021_004517/best_training_design.json'
    baseline=optics.evaluate(layers_of(json.loads(baseline_path.read_text(encoding='utf-8'))))
    write(output/'pure_hppo.json',compact_design(baseline))
    np.savez_compressed(output/'pure_hppo.npz',wavelengths_nm=optics.wavelengths_nm,
                        angles_deg=optics.angles_deg,**{q:np.asarray(baseline[q]) for q in 'RTA'})
    def row_of(name,report):
        bands=compact_design(report)['target_errors']
        return dict(case=name,score=report['score'],layers=report['layer_count'],
                    total_thickness_nm=report['total_thickness_nm'],UV_R=bands[0]['mean_R'],
                    visible_T=bands[1]['mean_T'],visible_A=bands[1]['mean_A'],IR_R=bands[2]['mean_R'])
    rows=[row_of('pure_hppo',baseline)]
    names=args.cases or ['search0','search1','medium12_thickness','long45_thickness','medium12_structure','long45_structure']
    finished=[]
    for name in names:
        source=root/name/'best_design.json'
        if not (root/name/'summary.json').exists():
            if args.cases:raise FileNotFoundError(f'Incomplete search: {name}')
            continue
        case=output/name
        case.mkdir(exist_ok=True)
        iterations=overrides.get(name,args.iterations)
        digest=hashlib.sha256(source.read_bytes()+json.dumps(cfg,sort_keys=True).encode()+str(iterations).encode()).hexdigest()
        if (case/'summary.json').exists() and json.loads((case/'summary.json').read_text())['input_digest']==digest:
            summary=json.loads((case/'summary.json').read_text())
            report=json.loads((case/'best_design.json').read_text())
            rows.append(summary['result'])
            finished.append(name)
            continue
        data=json.loads(source.read_text(encoding='utf-8'))
        initial=optics.evaluate(layers_of(data))
        print(json.dumps(dict(event='dense_initial',case=name,score=initial['score'])),flush=True)
        started=time.perf_counter()
        report=refine(optics,layers_of(data),max_iterations=iterations)
        elapsed=time.perf_counter()-started
        assert report['score']>=initial['score']-1e-12
        write(case/'before_dense.json',compact_design(initial))
        write(case/'best_design.json',compact_design(report))
        np.savez_compressed(case/'spectra.npz',wavelengths_nm=optics.wavelengths_nm,
                            angles_deg=optics.angles_deg,**{q:np.asarray(report[q]) for q in 'RTA'})
        with (case/'angle_averaged_spectra.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['wavelength_nm','R','T','A'])
            writer.writerows(zip(optics.wavelengths_nm,*[np.asarray(report[q]).mean(0) for q in 'RTA']))
        row=row_of(name,report)
        rows.append(row)
        summary=dict(input_digest=digest,source=str(source),before_dense_score=initial['score'],
                     dense_refinement_seconds=elapsed,max_iterations=iterations,optimization=report['optimization'],result=row)
        write(case/'summary.json',summary)
        finished.append(name)
        print(json.dumps(dict(event='dense_refined',**summary)),flush=True)
    write(output/'comparison.json',dict(evaluation_config=cfg,results=rows))
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    best=max(rows,key=lambda r:r['score'])
    best_path=output/'pure_hppo.json' if best['case']=='pure_hppo' else output/best['case']/'best_design.json'
    write(root/'best_design.json',json.loads(best_path.read_text(encoding='utf-8')))
    labels=[r['case'].replace('medium12','12L').replace('long45','45L').replace('_thickness','-d').replace('_structure','-struct') for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    x=np.arange(len(rows))
    axes[0].bar(x,[r['score'] for r in rows]);axes[0].set_title('Dense score: fixed weights 1:4:3');axes[0].set_ylim(0,1)
    axes[1].bar(x-.18,[r['visible_T'] for r in rows],.36,label='Visible T')
    axes[1].bar(x+.18,[r['IR_R'] for r in rows],.36,label='IR R')
    axes[1].legend();axes[1].set_ylim(0,1);axes[1].set_title('Application metrics: uniform angle means')
    axes[2].bar(x,[r['layers'] for r in rows]);axes[2].set_title('Layer count')
    for ax in axes:
        ax.set_xticks(x,labels,rotation=55,ha='right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=170);plt.close(fig)
    arrays=np.load(output/'pure_hppo.npz' if best['case']=='pure_hppo' else output/best['case']/'spectra.npz')
    wave=arrays['wavelengths_nm']
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for ax,q in zip(axes,'RTA'):
        ax.plot(wave,np.asarray(baseline[q]).mean(0),label='Pure HPPO best')
        ax.plot(wave,arrays[q].mean(0),label='Best improve + dense refinement')
        ax.axvspan(480,700,color='green',alpha=.1);ax.set_ylim(-.02,1.02);ax.set_ylabel(q);ax.grid(alpha=.2)
    axes[0].legend();axes[0].set_title('90 angles (0-89 deg), 271 wavelengths, unpolarized')
    axes[-1].set_xlabel('Wavelength (nm)');fig.tight_layout();fig.savefig(root/'spectra.png',dpi=170);plt.close(fig)
    trace_fig,trace_ax=plt.subplots(figsize=(9,4.5))
    for name in finished:
        events=[json.loads(line) for line in (root/name/'search.jsonl').read_text(encoding='utf-8').splitlines()]
        events=[e for e in events if e['event']=='candidate']
        trace_ax.plot(np.arange(1,len(events)+1),[e['best_score'] for e in events],label=name)
    trace_ax.set_xlabel('Candidate evaluations');trace_ax.set_ylabel('Best search score (18-angle grid)')
    trace_ax.legend(fontsize=8);trace_ax.grid(alpha=.2);trace_fig.tight_layout();trace_fig.savefig(root/'search_progress.png',dpi=170);plt.close(trace_fig)
    lines=['# 灯泡滤光片 improve 实验结果','',
           '评价目标沿用上一轮：UV R=1、可见T=1、IR R=1；权重1:4:3。',
           '下表全部在原90角度、271波长上复核；搜索结果又在限定迭代预算内进行了密集网格联合厚度优化。',
           '材料数据、15–400 nm单层厚度、45层上限和相邻材料约束不变。','',
           f'纯HPPO最好设计得分：{baseline["score"]:.8f}。',
           f'本次最好得分：{best["score"]:.8f}（{best["case"]}），差值{best["score"]-baseline["score"]:+.8f}。','',
           '| 阶段 | 得分 | 可见平均T | 可见平均A | IR平均R | 层数 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["case"]} | {r["score"]:.6f} | {r["visible_T"]:.2%} | {r["visible_A"]:.2%} | {r["IR_R"]:.2%} | {r["layers"]} |')
    lines+=['','提升来自已有HPPO候选的结构邻域搜索和TMM梯度厚度优化。未复训HPPO，未执行VAC模仿训练，也未使用论文结构作起点。',
            '不能将混合搜索结果归因于纯HPPO，也没有全局最优保证。',
            'best_design.json为最好完整设计；dense_validation保留逐项复核、优化记录与压缩光谱。',
            'comparison.png、spectra.png为密集验证图；search_progress.png只表示18角度搜索轨迹。',
            'CHANGELOG.md、before/、improvement_tests.log记录必要修复与7项通过的回归检查。',
            f'报告时间：{datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")}。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(best=best,baseline=rows[0],report=str(root/'RESULTS.md')),indent=2),flush=True)


if __name__=='__main__':
    main()
