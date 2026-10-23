"""Assemble the completed, configuration-only experiment evidence."""
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

root=Path(sys.argv[1]).resolve()
project=Path('E:/lithography_machine_lens/film_hppo')
plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
latest=json.loads((root/'latest_validation.json').read_text(encoding='utf-8'))
assert latest['all_complete']
validation=Path(latest['directory'])
comparison=json.loads((validation/'comparison.json').read_text(encoding='utf-8'))
policies=json.loads((root/'policy_evaluation/summary.json').read_text(encoding='utf-8'))
trial_rows=[r for r in comparison['results'] if r['experiment'].startswith('e')]
assert len(trial_rows)==12 and len(policies)==12
integrity=[]
baseline_sources=plan['batches'][0]['manifest']['source_hashes']
for batch in plan['batches']:
    manifest=batch['manifest']
    assert manifest['source_hashes']==baseline_sources
    for name,digest in manifest['source_hashes'].items():
        assert hashlib.sha256((project/name).read_bytes()).hexdigest()==digest,name
    for name,digest in manifest['material_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    for entry in manifest['runs']:
        state=json.loads(Path(entry['record']).read_text(encoding='utf-8'))
        assert state['status']=='complete',entry['record']
        text=Path(entry['train_log']).read_text(encoding='utf-8',errors='replace')
        collector=(Path(entry['run_dir'])/'log/collector/collector_logger.txt').read_text(encoding='utf-8')
        episodes=re.findall(r'total_episode_count: (\d+)',collector)
        with Path(entry['plot']).with_suffix('.csv').open(encoding='utf-8-sig',newline='') as stream:
            scalar_rows=list(csv.DictReader(stream))
        assert all(math.isfinite(float(r['value'])) for r in scalar_rows)
        integrity.append(dict(group=batch.get('label','e'+str(batch['epochs'])),seed=entry['seed'],
                              source_unchanged=True,materials_unchanged=True,
                              config_unchanged=hashlib.sha256(Path(entry['config']).read_bytes()).hexdigest()==entry['config_sha256'],
                              env_steps=state['result']['env_step'],updates=state['result']['train_iter'],
                              training_seconds=state['training_seconds'],
                              finite_logged_scalar_count=len(scalar_rows),
                              collector_completed_episodes=int(episodes[-1]),
                              rollout_KL_stops=text.count('rollout reuse stopped:'),
                              tracebacks=text.count('Traceback (most recent call last):')))
assert all(r['config_unchanged'] and not r['tracebacks'] for r in integrity)
(root/'integrity_report.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
for name in ['comparison.png','spectra_comparison.png','comparison.csv','group_statistics.json']:
    shutil.copy2(validation/name,root/name)
best=max(trial_rows,key=lambda r:r['score'])
shutil.copy2(best['dense_design'],root/'best_training_design.json')
best_sample=max(policies,key=lambda r:r['score_max'])
shutil.copy2(best_sample['best_sampled_design'],root/'best_inference_design.json')
cfg=json.loads(Path(best['training_config']).read_text(encoding='utf-8'))
cfg['training']['exp_name']='runs/incandescent_T_selected/seed'+str(cfg['training']['seed'])
cfg['_experiment_selected_from']=best['training_config']
(root/'best_trial_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
old=latest['best_previous_under_new_objective']
best_mean=max(policies,key=lambda r:r['score_mean'])
lines=['# 灯泡滤光片 HPPO 配置实验结果','',
       f'完成时间：{datetime.now().isoformat(timespec="seconds")}。共12次训练，全部正常结束。',
       f'共采集 {sum(r["env_steps"] for r in integrity):,} 个环境步，每次约60000步，在采样轮边界停止。','',
       '## 主要结论','',
       f'- 本轮训练保存候选在密集网格上的最高得分：**{best["score"]:.8f}**（{best["experiment"]}）。',
       f'- 旧候选按同一新目标重新计算的最好得分：**{old["score"]:.8f}**。差值：{best["score"]-old["score"]:+.8f}。',
       f'- 最好设计的可见光平均透射率 **{best["visible_T"]:.2%}**，红外平均反射率 **{best["IR_R"]:.2%}**，共 **{best["layers"]}层**。',
       f'- 独立64回合评估中的最高策略均值：**{best_mean["score_mean"]:.5f}**（{best_mean["experiment"]}，采样标准误{best_mean["score_standard_error"]:.5f}）。',
       f'- 独立推理中生成的最好样本得分：**{best_sample["score_max"]:.8f}**（{best_sample["experiment"]}），与训练候选分开保存。',
       '- 没有把增加层数、更换训练权重后的分数上涨，或一次偶然高分，作为达到滤光片目标的证据。',
       '- 原配置 configs/incandescent_task.json 保留；未将试验参数写回默认配置。','',
       '## 评价口径','',
       '波段仍为300–470 nm、480–700 nm、710–3000 nm。可见光目标由R=0改为T=1，避免吸收满足低反射。',
       '统一复核得分：F=(UV平均R + 4×可见光平均T + 3×IR平均R)/8。',
       '训练采用18个角度分箱中心与原271波长；本表每个候选都重新使用0–89°共90角度、271波长、s/p平均计算。',
       'IR权重6组只改变训练权重，最终评分仍采用1:4:3。min2/min4组限制训练时可选的最短结构，仍然在原始最大45层可行空间内。',
       '第一轮为两组各3个seed；其余6项为单seed诊断。不能据此证明跨seed稳定优越，也未覆盖更长训练预算。','',
       '## 全部训练结果（最好保存设计）','',
       '| 试验 | 密集得分 | 可见T | 可见A | IR R | 层数 | 独立策略均值 |',
       '|---|---:|---:|---:|---:|---:|---:|']
for row in trial_rows:
    policy=next(p for p in policies if p['experiment']==row['experiment'])
    lines.append(f'| {row["experiment"]} | {row["score"]:.6f} | {row["visible_T"]:.2%} | {row["visible_A"]:.2%} | '
                 f'{row["IR_R"]:.2%} | {row["layers"]} | {policy["score_mean"]:.5f} |')
lines += ['', '## 保存内容','',
          '- `comparison.png`：统一密集复核后的设计表现；`spectra_comparison.png`：实际R/T/A谱线。',
          '- `comparison.csv`：完整数值、配置路径、设计路径、实际采样与更新次数。',
          '- `best_training_design.json`：本轮训练期间保存的最佳候选；并不表示突破了旧候选。',
          '- `best_inference_design.json`：训练后独立64回合评估中生成的最佳样本，与训练候选分开保存。',
          '- `best_trial_config.json`：最高训练候选所属的试验配置，输出名已改成独立目录；并非自动推荐的新默认值。',
          '- `policy_evaluation/summary.json`及`policy_quality.png`：原生ckpt_best的独立64回合结果。',
          '- `integrity_report.json`：源文件、材料与运行配置散列，实际步数、耗时、KL保护次数。',
          '- `policy_diagnostics.json`：最后检查点在指定状态下的动作概率和厚度分布，仅作诊断。',
          '- `plan.json`与`EXPERIMENT_LOG.md`：可追踪每项设置、日志、检查点与曲线。','',
          '没有新增强化学习算法、网络模块、厚度优化器、结构邻域搜索或模仿学习。没有训练数据来源于论文结构。']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (root/'EXPERIMENT_LOG.md').open('a',encoding='utf-8') as stream:
    stream.write(f'\n\n## 全部结束\n\n{datetime.now().isoformat(timespec="seconds")}：12/12训练正常完成，源代码、材料与每次运行配置散列检查通过。最终复核目录：{validation}。完整结论见 RESULTS.md。\n')
print(json.dumps(dict(best_training=best,old_best=old,best_policy=best_mean,
                      env_steps=sum(r['env_steps'] for r in integrity),report=str(root/'RESULTS.md')),indent=2))
