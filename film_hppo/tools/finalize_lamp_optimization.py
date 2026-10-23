"""Validate native-HPPO lamp candidates on common grids and write final artifacts.

Deletion is deliberately separate: inspect PROCESS.md and final dependencies,
then remove only the experiment's exact trial directories using PowerShell.
"""
from copy import deepcopy
import csv
import hashlib
import json
import pickle
import math
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from film_rl.config import load_config
from film_rl.design import compact_design
from film_rl.improve import layers_of
from film_rl.lamp_metrics import lamp_metrics, lamp_key, dominates, IR_BANDS
from film_rl.optics import FilmOptics
from evaluate_baseline import calculate_spectra
from tools.optimize_lamp_hppo import read, write


def deployment_key(report):
    m = report['application_metrics']
    outband_guard = m['UV_R'] >= .70 and m['IR_R'] >= .85 and m['IR_min_subband_R'] >= .70
    return (m['application_targets_met'], outband_guard, *lamp_key(m)[1:])


def independent_spotcheck(optics, report):
    """Independent scalar Fresnel recursion for the air/stack/air geometry."""
    errors=[]
    thickness=np.asarray([row['thickness_nm'] for row in report['layers']])
    ids=[row['material_id'] for row in report['layers']]
    for wave in [400.,480.,550.,700.,900.,1500.,2800.]:
        w=int(np.argmin(abs(optics.wavelengths_nm-wave)))
        n=np.array([1.+0j]+[optics.material_indices[i,w] for i in ids]+[1.+0j])
        for angle in [0,30,60,80,89]:
            cosine=np.sqrt(1-(np.sin(np.deg2rad(angle))/n)**2)
            values=[]
            for pol in ['s','p']:
                if pol=='s':
                    denom=n[:-1]*cosine[:-1]+n[1:]*cosine[1:]
                    r=(n[:-1]*cosine[:-1]-n[1:]*cosine[1:])/denom
                    t=2*n[:-1]*cosine[:-1]/denom
                else:
                    denom=n[1:]*cosine[:-1]+n[:-1]*cosine[1:]
                    r=(n[1:]*cosine[:-1]-n[:-1]*cosine[1:])/denom
                    t=2*n[:-1]*cosine[:-1]/denom
                reflection,transmission=r[-1],t[-1]
                for interface in range(len(r)-2,-1,-1):
                    phase=np.exp(2j*np.pi*n[interface+1]*cosine[interface+1]*thickness[interface]/wave)
                    den=1+r[interface]*reflection*phase**2
                    transmission=t[interface]*transmission*phase/den
                    reflection=(r[interface]+reflection*phase**2)/den
                values.append([abs(reflection)**2,abs(transmission)**2])
            expected=np.mean(values,axis=0)
            actual=np.array([report['R'][angle][w],report['T'][angle][w]])
            errors.append(float(abs(expected-actual).max()))
    if max(errors)>1e-7:
        raise AssertionError(('Independent Fresnel recursion mismatch',max(errors)))
    return dict(method='independent scalar Fresnel recursion, s/p average',points=len(errors),
                max_absolute_RT_error=max(errors),passed=True)


def evaluate(optics, data):
    spectra = calculate_spectra(optics, layers_of(data), optics.angles_deg, 64)
    report = {k: deepcopy(v) for k, v in data.items() if k not in ('design_format', 'sampling', 'target_errors')}
    objective = optics.objective.evaluate_spectra(spectra)
    report.update(objective)
    report.update(reward=objective['score'], wavelengths_nm=optics.wavelengths_nm.tolist(),
                  angles_deg=optics.angles_deg.tolist(), **{q: x.tolist() for q, x in spectra.items()})
    epsilon=float(optics.cfg.get('reward_epsilon',1e-6))
    report['log_reward']=-math.log10(max(objective['mae'],epsilon))/-math.log10(epsilon)
    for q, word in [('R', 'reflection'), ('T', 'transmission'), ('A', 'absorption')]:
        report['mean_'+word] = float(spectra[q].mean())
        report['min_'+word] = float(spectra[q].min())
    report['application_metrics'] = lamp_metrics(report)
    report['selection_method'] = 'physical-metric Pareto comparison; common90 angles and2nm mesh'
    return report


def main():
    torch.set_num_threads(1)
    root = PROJECT/'outputs'/sys.argv[1]
    plan = read(root/'progress.json')
    if plan['status'] != 'training_complete_requires_dense_validation':
        raise RuntimeError('Training has not finished')
    final = root/'final'
    for name in ['production','production_refined','production_fine']:
        if (root/name/'training_config.json').exists():final=root/name
    cfg = load_config(final/'training_config.json')
    cfg['angles_deg'] = list(range(90))
    common = FilmOptics(cfg)
    candidates = []
    seen = set()
    stage_inputs = []
    for stage in plan['stages']:
        records = [(read(path), str(path)) for path in Path(stage['folder']).glob('candidate_*.json')]
        for path in (Path(stage['run'])/'designs').glob('*.json'):
            values = read(path)
            records.extend((row, str(path)) for row in (values if isinstance(values,list) else [values]))
        records.sort(key=lambda pair:deployment_key(pair[0]),reverse=True)
        stage_inputs += [(data,path,stage) for data,path in records[:10]]
    if final.name.startswith('production'):
        records=[]
        for path in (root/'trials/physical_refinement').glob('seed*/best_design.json'):
            records.append((read(path),str(path)))
        if (root/'trials/late_refinement/best_design.json').exists():
            records.append((read(root/'trials/late_refinement/best_design.json'),str(root/'trials/late_refinement/best_design.json')))
        if (root/'trials/fine_refinement/best_design.json').exists():
            records.append((read(root/'trials/fine_refinement/best_design.json'),str(root/'trials/fine_refinement/best_design.json')))
        for path in [final/'refined_design.json',final/'policy_deterministic_design.json',final/'final_policy_deterministic_design.json']:
            records.append((read(path),str(path)))
        for path in (PROJECT/'runs'/root.name/final.name/'designs').glob('*.json'):
            values=read(path)
            records.extend((row,str(path)) for row in (values if isinstance(values,list) else [values]))
        records.sort(key=lambda pair:deployment_key(pair[0]),reverse=True)
        stage_inputs += [(data,path,{'name':'hybrid_production','sigma':cfg['training']['initial_sigma'],
                         'lr':cfg['training']['learning_rate'],'split':True}) for data,path in records[:15]]
    for data,path,stage in stage_inputs:
            signature = tuple((i, round(d, 5)) for i, d in layers_of(data))
            if signature in seen:
                continue
            seen.add(signature)
            report = evaluate(common, data)
            report['candidate_source'] = {'stage': stage['name'], 'path_before_cleanup': path,
                'sigma': stage['sigma'], 'lr': stage['lr'], 'split': stage['split']}
            candidates.append(report)
    # Retain fair common-grid comparison with earlier completed hybrid networks.
    for old_name in ['production','production_refined']:
        if old_name == final.name or not (root/old_name/'status.json').exists():continue
        records=[]
        for path in (PROJECT/'runs'/root.name/old_name/'designs').glob('*.json'):
            values=read(path)
            records.extend((row,str(path)) for row in (values if isinstance(values,list) else [values]))
        records.sort(key=lambda pair:deployment_key(pair[0]),reverse=True)
        for data,path in records[:5]:
            signature=tuple((i,round(d,5)) for i,d in layers_of(data))
            if signature in seen:continue
            seen.add(signature)
            report=evaluate(common,data)
            report['candidate_source']={'stage':old_name,'path_before_cleanup':path,
                'sigma':None,'lr':None,'split':True}
            candidates.append(report)
    candidates.sort(key=deployment_key, reverse=True)
    # Include physically different Pareto alternatives, not just a score top-k.
    front = [row for row in candidates if not any(dominates(other['application_metrics'], row['application_metrics'])
             for other in candidates if other is not row)]
    shortlist = candidates[:5]
    for extreme in [max(front, key=lambda r:r['application_metrics']['visible_T']),
                    max(front, key=lambda r:r['application_metrics']['IR_min_subband_R'])]:
        if not any(extreme is x for x in shortlist):
            shortlist.append(extreme)
    dense_cfg = deepcopy(cfg)
    dense_cfg['wavelengths_nm'] = list(np.arange(300., 3001., 2.))
    dense = FilmOptics(dense_cfg)
    write(final/'evaluation_config.json', dense_cfg)
    validated = []
    for number, data in enumerate(shortlist):
        report = evaluate(dense, data)
        report['candidate_source'] = data['candidate_source']
        validated.append(report)
        print(json.dumps({'dense_candidate': number, 'metrics': report['application_metrics']}, ensure_ascii=False), flush=True)
    validated.sort(key=deployment_key, reverse=True)
    best = validated[0]
    independent_check=independent_spotcheck(dense,best)
    baseline_path = PROJECT/'outputs/lamp_four_targets/20261021_195347/best_design.json'
    baseline = evaluate(dense, read(baseline_path))
    final_policy = (evaluate(dense,read(final/'final_policy_deterministic_design.json'))
                    if (final/'final_policy_deterministic_design.json').exists() else None)
    best_compact = compact_design(best)
    best_compact['candidate_source'] = best['candidate_source']
    write(final/'best_design.json', best_compact)
    metrics = best['application_metrics']
    comparison = {'baseline': baseline['application_metrics'], 'optimized': metrics,
                  'baseline_source': str(baseline_path), 'selected_source': best['candidate_source'],
                  'common_angles': list(range(90)), 'wavelength_step_nm': 2.,
                  'all_application_targets_met': metrics['application_targets_met'],
                  'optimized_dominates_baseline': dominates(metrics, baseline['application_metrics']),
                  'baseline_dominates_optimized': dominates(baseline['application_metrics'], metrics),
                  'dense_candidate_metrics': [r['application_metrics'] for r in validated]}
    comparison['independent_optical_check']=independent_check
    if final_policy is not None:
        comparison['final_network_deterministic_dense'] = final_policy['application_metrics']
        write(final/'final_policy_deterministic_dense.json',compact_design(final_policy))
    write(final/'metrics.json', comparison)
    np.savez_compressed(final/'spectra.npz', wavelengths_nm=best['wavelengths_nm'],
        angles_deg=best['angles_deg'], **{q:np.asarray(best[q]) for q in 'RTA'},
        **{'baseline_'+q:np.asarray(baseline[q]) for q in 'RTA'})
    wave = np.asarray(best['wavelengths_nm'])
    with (final/'spectra.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['wavelength_nm','baseline_R','baseline_T','baseline_A','optimized_R','optimized_T','optimized_A'])
        writer.writerows(zip(wave, *[np.asarray(r[q]).mean(0) for r in (baseline,best) for q in 'RTA']))
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, layout='constrained')
    for ax, q in zip(axes, 'TRA'):
        for row, label in [(baseline,'Previous HPPO'),(best,'Optimized HPPO')]:
            ax.plot(wave, np.asarray(row[q]).mean(0), label=label)
        ax.axvspan(480,700,color='gold',alpha=.15)
        ax.set_ylabel(q); ax.set_ylim(0,1); ax.grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('Wavelength (nm)')
    fig.suptitle('Uniform mean over 90 angles; 2 nm validation')
    fig.savefig(final/'spectra_comparison.png', dpi=160);plt.close(fig)
    # Retained initialization is a complete final-stage asset; it has no runtime
    # dependency on the temporary source checkpoint that created it.
    source_dir = final/'source'
    source_dir.mkdir(exist_ok=True)
    for path in (PROJECT/'film_rl').glob('*.py'):
        shutil.copy2(path, source_dir/path.name)
    for name in ('optimize_lamp_hppo.py','finalize_lamp_optimization.py','refine_lamp_physical.py','run_lamp_production.py'):
        shutil.copy2(PROJECT/'tools'/name, source_dir/name)
    reusable = deepcopy(read(final/'training_config.json'))
    reusable['training']['exp_name'] = 'runs/'+root.name+'/production'
    write(PROJECT/'configs/incandescent_optimized.json', reusable)
    lines = ['# 白炽灯 HPPO 优化过程与最终结果','',
        '## Material Passport','', '- Origin Skill: ARS experiment-agent', '- Origin Mode: run / validate',
        '- Origin Date: 2026-10-21', '- Verification Status: ANALYZED; common-grid optical recalculation completed',
        '- Version Label: lamp_physical_v1','',
        '用户明确允许加入厚度优化。最终采用HPPO候选材序、TMM梯度厚度精调、精英轨迹监督初始化及原生PPO继续训练的混合流程；不能称为纯HPPO从零达到。没有论文结构初始化或材料替换。每阶段重置优化器和随机数状态，不是完整断点恢复。',
        '材料、空气边界、1–80层、15–400nm层厚及相邻材料限制沿用原设置。可见窗口沿用480–700nm；这不是对全部400–780nm均设T=1。',
        '对照及早期训练用18角度、10nm；最后精调及保留训练采用18角度、5nm。最终固定0–89°的90角度、2nm复核。下表均为等权角度/波长均值，不表示每个点达标。','',
        '## 参数试验','', '| 阶段 | 新增交互步预算 | sigma | lr | 红外细分 | 可见T | UV R | IR R | 最弱IR段R |',
        '|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for stage in plan['stages']:
        m=stage['best']; lines.append(f"| {stage['name']} | {stage['budget']} | {stage['sigma']} | {stage['lr']} | {stage['split']} | {m['visible_T']:.5f} | {m['UV_R']:.5f} | {m['IR_R']:.5f} | {m['IR_min_subband_R']:.5f} |")
    if final.name.startswith('production'):
        lines += ['', '用户授权混合优化后增加的厚度精调：', '',
            '```json',json.dumps(read(root/'trials/physical_refinement/summary.json'),ensure_ascii=False,indent=2),'```',
            '梯度先以实际TMM中心差分检验。优化目标是提高可见T并惩罚UV/IR反射及可见吸收约束的违例，保存可行中间解，不仅接受优化器末次输出。',
            '网络监督初始化2400步，随后继续PPO60000交互步，sigma=.012、lr=3e-6、离散熵=.0001、连续熵=0。',
            '```json',json.dumps(read(root/'trials/physical_refinement/gradient_check.json'),ensure_ascii=False,indent=2),'```']
    if final.name == 'production_refined':
        lines += ['', '首版混合模型完成60000步PPO后，采用后续45层精调结果重新做2400步轨迹初始化；最终保留阶段另外完成24000步PPO，学习率1e-6、sigma=.01。不将两个阶段混为一次从零训练。']
    if final.name == 'production_fine':
        lines += ['', '先后完成首版混合模型60000步、45层模型24000步PPO；随后针对10nm与2nm复核差异，增加5nm精调与最终5nm网格12000步PPO。最终学习率1e-6、sigma=.008、离散熵=.0001、连续熵=0。各阶段分别初始化，不将模型初始化或额外预算混为从零纯PPO的性能。']
    if (root/'trials/late_refinement/summary.json').exists():
        lines += ['', '后续从长程HPPO中按物理门槛选出新材序，再进行一轮厚度精调：',
            '```json',json.dumps(read(root/'trials/late_refinement/summary.json'),ensure_ascii=False,indent=2),'```']
    if (root/'trials/fine_refinement/summary.json').exists():
        lines += ['', '最后5nm网格精调，用于降低10nm采样的光谱偏差：',
            '```json',json.dumps(read(root/'trials/fine_refinement/summary.json'),ensure_ascii=False,indent=2),'```']
    if (root/'trials/ar_screening.json').exists():
        screening=read(root/'trials/ar_screening.json')
        lines += ['', f'另外检查了{len(screening)}套前后匹配层配置；最佳粗网格可见T={max(r["metrics"]["visible_T"] for r in screening):.6f}，未出现足以替代主方案的突破。']
    if (root/'trials/structured_screening.json').exists():
        screening=read(root/'trials/structured_screening.json')
        lines += ['', f'另外粗测了{len(screening)}套交替高/低折射率、厚度渐变的结构化起点；未同时满足带外门槛，未作为最终模型初始化。']
    lines += ['', '阶段表是训练18角度的筛选指标；下表才是统一90角度/2nm复核，不能直接混用。',
        '筛选保留物理指标的有界Pareto候选，并用调和均值帮助排序；训练本身仍使用加权光谱误差奖励。没有把score作为唯一评价。',
        '最终部署选择优先满足UV R>=.70、IR均值>=.85、最弱IR段>=.70的带外门槛，再比较调和指标；避免选择带外性能不足的高透射候选。',
        '红外细分触发原因：'+plan.get('split_reason','短程对照出现实质改善，未触发分段。'),'',
        '## 最终同条件对比','', '| 指标 | 旧最佳HPPO | 本次保留设计 |','|---|---:|---:|']
    for key in ['visible_T','visible_R','visible_A','visible_T_p10','visible_T_0_60','UV_R','IR_R','IR_min_subband_R']:
        lines.append(f"| {key} | {comparison['baseline'][key]:.6f} | {metrics[key]:.6f} |")
    for number, bounds in enumerate(IR_BANDS):
        lines.append(f"| R {bounds[0]:g}–{bounds[1]:g}nm | {comparison['baseline']['IR_subband_R'][number]:.6f} | {metrics['IR_subband_R'][number]:.6f} |")
    if final_policy is not None:
        lines += ['', '最终网络确定性输出也独立在90角度/2nm复核：', '```json',
                  json.dumps(final_policy['application_metrics'],ensure_ascii=False,indent=2),'```']
    actual=[]
    for run in sorted((PROJECT/'runs'/root.name).iterdir()):
        path=run/'result.pkl'
        if path.is_file():
            with path.open('rb') as stream:result=pickle.load(stream)
            actual.append({'stage':run.name,'env_steps':int(result['env_step']),'optimizer_iterations':int(result['train_iter'])})
    lines += ['', '全部已完成阶段的实际训练计数（非计划预算）：','```json',
              json.dumps(actual,ensure_ascii=False,indent=2),'```',
              '材料数据SHA256：','```json',json.dumps(plan['material_hashes'],ensure_ascii=False,indent=2),'```']
    lines += ['', '所有验收目标同时满足：'+str(metrics['application_targets_met'])+'。',
        '本次设计在全部报告的Pareto方向支配旧设计：'+str(comparison['optimized_dominates_baseline'])+'；反向支配：'+str(comparison['baseline_dominates_optimized'])+'。',
        '未达标时必须如实报告剩余差距；若不同波段有得失，应称权衡而不能称所有方面都提高。',
        '不将反射率版chi用于实际透射照明的唯一评价。','',
        '## 最终训练设置与目标','', '```json',json.dumps(reusable,ensure_ascii=False,indent=2),'```','',
        '最终候选来源：'+json.dumps(best['candidate_source'],ensure_ascii=False),
        '最终训练初始化位于'+reusable['training']['initial_model']+'；保留文件包含完整权重，删除临时来源不影响复用。',
        '## 代码与检查','',
        '新增lamp_metrics.py：T/R/A、UV、五段IR、角度/分位数诊断及非score候选保存；新增initial_sigma配置，确保在原生PPO初始化后生效。默认旧任务保持原路径。',
        '回归测试：初次96项通过；新增sigma生效与低score物理改善仍保存的测试。首个新增保存测试使用系统临时目录遇到沙盒写入限制，改为项目内临时目录后5项通过；最后完整98项测试全部通过。',
        '独立Fresnel递推抽查（7波长、5角度，各自s/p平均）与TMM-fast的误差：'+json.dumps(independent_check,ensure_ascii=False),
        '## 清理记录','',
        '清理范围仅限本次root/trials和runs同组未被选为最终保留的阶段；最终保留目录名为'+final.name+'。历史实验、材料数据库和其他项目不属于删除范围。',
        '中间配置及指标已汇入本文；临时权重、日志和JSON不再逐一保留。删除会损失中间试验逐步重放能力，这是按用户要求选择的保留策略。',
        '清理状态：待执行。','',
        '## 最终文件','', '- configs/incandescent_optimized.json：可重复启动的最终参数。',
        '- outputs/'+root.name+'/'+final.name+'/：训练配置、完整初始化权重、最优膜系、密网格指标/光谱/图和源码快照。',
        '- runs/'+root.name+'/'+final.name+'/：实际最终训练的checkpoint、日志与设计。','',
        '## 解释边界','',
        '参数筛选不是独立测试；所有生成候选均参与选择。不同阶段有模型继承关系，不能当作独立seed统计。',
        '尚未获得多seed重复的显著性或全局最优证明；本次结论限于实际预算及当前nk数据。']
    (root/'PROCESS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plan.update(status='validated_pending_cleanup', selected_source=best['candidate_source'], comparison=comparison)
    write(root/'progress.json',plan)
    print(json.dumps(comparison,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
