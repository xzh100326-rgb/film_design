"""Repair result navigation after the explicit, hash-checked cleanup plan."""
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CLEAN=ROOT/'outputs/maintenance/cleanup_20261021'
MAIN=ROOT/'outputs/hppo_spectral_search/20261021_145752'


def main():
    if (CLEAN/'validation.json').exists():raise FileExistsError('Cleanup has already been finalized; see validation.json')
    plan=json.loads((CLEAN/'plan.json').read_text(encoding='utf-8'))
    assert (CLEAN/'filesystem_completed.json').exists()
    variants=[]
    for row in plan['moves']:
        a,b=ROOT/row['source'],ROOT/row['destination']
        variants.extend([(str(a),str(b)),(a.as_posix(),b.as_posix()),
                         (row['source'],row['destination']),
                         (row['source'].replace('/','\\'),row['destination'].replace('/','\\'))])
    variants=sorted(set(variants),key=lambda x:len(x[0]),reverse=True)
    def remap(value):
        if isinstance(value,str):
            for a,b in variants:
                if value==a or value.startswith(a+'/') or value.startswith(a+'\\'):
                    return b+value[len(a):]
        elif isinstance(value,list):return [remap(x) for x in value]
        elif isinstance(value,dict):return {remap(k):remap(v) for k,v in value.items()}
        return value
    names=set(plan['rewrite_metadata_names'])
    changed=[]
    for base in ('outputs','runs'):
        for p in (ROOT/base).rglob('*'):
            if not p.is_file() or p.suffix not in ('.json','.csv','.md'):continue
            rel=p.relative_to(ROOT)
            if CLEAN in p.parents or 'before' in rel.parts or 'source' in rel.parts or 'configs' in rel.parts:continue
            if p.name not in names and p.parent.name!='status' and p.suffix!='.md':continue
            raw=p.read_text(encoding='utf-8-sig')
            if p.suffix=='.json':
                old=json.loads(raw);new=remap(old)
                if new==old:continue
                updated=json.dumps(new,ensure_ascii=False,indent=2)+'\n'
            elif p.suffix=='.csv':
                reader=csv.DictReader(io.StringIO(raw));fields=reader.fieldnames
                rows=list(reader);new=[]
                for row in rows:
                    new.append({k:remap(v) for k,v in row.items()})
                if new==rows:continue
                stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields)
                writer.writeheader();writer.writerows(new);updated='\ufeff'+stream.getvalue()
            else:
                updated=raw
                for a,b in variants:updated=updated.replace(a,b)
                if updated==raw:continue
            p.write_text(updated,encoding='utf-8');changed.append(rel.as_posix())

    def link(label,relative):return f'[{label}]({(ROOT/relative).as_posix()})'
    def doc(relative,text):
        p=ROOT/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')
    best='outputs/hppo_spectral_search/20261021_145752'
    trial='outputs/incandescent_hppo_experiments/20261021_004517'
    doc('outputs/README.md', '# 实验结果入口\n\n'
        '正式结果按任务/实验组保留；每组的参数副本、图、数值表放在同一批次中。\n\n'
        '## 当前灯泡滤光片\n\n'
        f'- {link("当前最佳膜系：80层，MAE=0.26755",best+"/best_design.json")}\n'
        f'- {link("纯HPPO结果、曲线和过程",best+"/RESULTS.md")}；{link("增强因子差距诊断",best+"/ENHANCEMENT_GAP.md")}\n'
        f'- GAE对照与继续训练的批量输出统一放在 {link("batches",best+"/batches")}；对应原生训练数据位于runs/lamp_R_hppo_*。\n'
        f'- {link("历史T目标参数试验",trial+"/RESULTS.md")}：12次测试的配置、最佳模型、曲线和最终验证保留，批量输出已收进该目录的batches/。\n'
        f'- {link("历史混合搜索对照","outputs/incandescent_improve/20261021_094617/RESULTS.md")}，与纯HPPO分开。\n'
        f'- {link("增强因子baseline","outputs/lamp_baseline/README.md")}。\n\n'
        '## DUV 193 nm 与吸收任务\n\n'
        '| 任务 | 结果位置 |\n|---|---|\n'
        f'| DUV旧nk、10 seeds | {link("nk_v1","outputs/duv193__hppo__nk_v1__a00-45")} |\n'
        f'| DUV当前nk、10 seeds | {link("nk_v2","outputs/duv193__hppo__nk_v2__a00-45")} |\n'
        f'| 400–2000 nm吸收HPPO基线 | {link("baseline","outputs/abs400-2000__hppo__baseline")} |\n'
        f'| 吸收多seed曲线与汇总 | {link("multiseed","outputs/abs400-2000__hppo__multiseed")} |\n'
        f'| 吸收混合搜索 | {link("pilot","outputs/abs400-2000__hybrid__pilot")} / {link("search","outputs/abs400-2000__hybrid__search")} |\n\n'
        f'{link("上周展示汇总","outputs/week_20261014_20261020/README.md")}；{link("材料和论文来源","outputs/references")}。参考资料位置保留，避免影响材料转换和baseline。\n\n'
        '## 清理及维护\n\n'
        f'{link("本次清理清单和测试过程README","outputs/maintenance/cleanup_20261021/README.md")}；{link("代码变更验证记录","outputs/maintenance/README.md")}。\n'
        '新增正式实验继续使用“任务__方法__版本/批次时间/seedN”；临时检查使用outputs/maintenance下独立目录，结束后将结论写进README。\n')
    doc('runs/README.md','# 训练原始数据入口\n\n'
        '正式实验保留designs、模型checkpoint、result.pkl和TensorBoard日志。图表和报告从outputs/README.md进入。\n\n'
        '| 目录 | 内容 |\n|---|---|\n'
        '| duv193__hppo__nk_v1__a00-45 | 旧nk的10个seed |\n'
        '| duv193__hppo__nk_v2__a00-45 | 当前nk的10个seed |\n'
        '| abs400-2000__hppo__baseline | 吸收任务代表性基线 |\n'
        '| incandescent_task | 早期灯泡批量训练 |\n'
        '| incandescent_task2__hppo__terminal | 原R目标训练基线 |\n'
        '| lamp_R_hppo_l095 | 当前实验lambda=0.95的2个seed |\n'
        '| lamp_R_hppo_l100 | 当前实验lambda=1.0的2个seed |\n'
        '| lamp_R_hppo_l100_continue_s0 | 纯HPPO模型权重继续训练 |\n'
        '| history/incandescent_T_trials_20261021 | 旧T目标12次参数测试 |\n\n'
        f'{link("当前最好设计",best+"/best_design.json")}；{link("所有结果索引","outputs/README.md")}。\n\n'
        '历史T测试保留ckpt_best.pth.tar，初始化/中间/末次checkpoint已清理；当前R目标、DUV和吸收正式实验的checkpoint保持原样。更早完整归档仍在archive/results_before_organization_20261020。\n')
    doc('runs/history/README.md','# 历史参数测试\n\n'
        'incandescent_T_trials_20261021集中保存旧灯泡T目标的12次训练，按原组名/时间/seed排列。\n'
        '每次的最佳模型、designs、TensorBoard、训练结果和配置快照保留；初始化与其他迭代模型已删。\n'
        '原生配置快照与模型字节保持原样，其中exp_name反映训练时旧地址；当前文件位置以整理后的manifest/status与清理path_mapping为准。\n'
        f'{link("测试参数、过程和结论",trial+"/EXPERIMENT_LOG.md")}；{link("测试结果",trial+"/RESULTS.md")}。\n')
    doc('outputs/incandescent_hppo_experiments/README.md','# 旧灯泡T目标HPPO试验\n\n'
        f'{link("12次试验结果",trial+"/RESULTS.md")}；{link("完整过程",trial+"/EXPERIMENT_LOG.md")}。\n'
        '批量输出统一在20261021_004517/batches/；训练模型在runs/history/incandescent_T_trials_20261021/。\n'
        '最终验证保留validation/20261021_013033；较早未完成时的光谱输出已清理，其README留存在清理记录中。\n'
        '可见光目标为T=1，不能与当前R=0的score直接混排。\n')
    doc('outputs/maintenance/README.md','# 维护与验证记录\n\n'
        f'- {link("结果清理及测试过程","outputs/maintenance/cleanup_20261021/README.md")}\n'
        f'- {link("统一S光谱奖励的变更记录","outputs/maintenance/reward_S_update_20261021/CHANGES.md")}\n'
        '- cleanup_20261021/retained_legacy_designs保留旧完整JSON备份的精简结构与指标。\n'
        '- preserved_readmes保存已清理测试目录中的原始Markdown，navigation_metadata_before.json.gz保存移动前的索引。\n')
    doc(best+'/README.md','# 当前纯HPPO光谱误差实验\n\n'
        f'{link("最佳设计",best+"/best_design.json")}；{link("完整结果",best+"/RESULTS.md")}；{link("训练过程",best+"/EXPERIMENT_LOG.md")}；{link("增强因子诊断",best+"/ENHANCEMENT_GAP.md")}。\n\n'
        'batches/l095、batches/l100、batches/l100_continue_s0集中保存批量配置、状态、原生训练控制台日志、曲线和数值表；plan.json已同步新地址。runs中的正式训练组位置保持不变。\n'
        'best_design.json、best_spectra.npz、baseline_stage2完整光谱、模型及材料散列保留。推理仅保留结构、episode表和统计；重复光谱可重新计算。\n'
        'dense_validation/candidate_history.jsonl.gz无损合并先前缓存；当前复核结果仍为各案例JSON与comparison.csv。\n'
        'baseline_stage1的旧基线/论文光谱与stage2完全一致，统一使用baseline_stage2/spectra_02.*和spectra_03.*；stage1/spectra_01.*仍保留。\n'
        '顶层重复console日志已清理，正式训练日志保存在batches/*/*/logs/，过程说明见EXPERIMENT_LOG.md。\n')
    doc('outputs/lamp_baseline/README.md','# 灯泡增强因子baseline索引\n\n'
        f'- {link("正式历史对照（comparison03）","outputs/lamp_baseline/20261021_comparison03/README.md")}\n'
        f'- {link("当前最佳设计的计算",best+"/baseline_stage2/README.md")}\n'
        f'- {link("分辨率收敛表","outputs/lamp_baseline/resolution_comparison.json")}\n'
        '- comparison01/02已被后续计算替代，原设置、指标表和README保存在history/，重复原始光谱和图已删除。\n'
        '- resolution_check保留设置和计算结果；1nm/128角度的大数组已清理，结论是与2nm/64角度的chi相对差小于0.02%。\n')
    doc('outputs/incandescent_improve/incomplete_20261021_152429/README.md',
        '# 未完成的improve试跑\n\n原outputs/improvement_20261021_152429。日志仅有initial记录，没有完成候选搜索结果。\n'
        '保留输入配置、初始设计、起点清单和源代码快照，作为测试过程记录；不纳入当前最好设计或纯HPPO成绩。\n')
    for rel in (best+'/baseline_stage1/README.md','outputs/lamp_baseline/20261021_resolution_check/README.md'):
        p=ROOT/rel
        with p.open('a',encoding='utf-8') as stream:
            stream.write('\n\n## 文件整理说明（2026-10-21）\n\n重复/分辨率测试大数组已清理，设置、指标与本README保留。当前完整增强因子光谱见outputs/hppo_spectral_search/20261021_145752/baseline_stage2；分辨率收敛表见outputs/lamp_baseline/resolution_comparison.json。\n')
    # Validation protects the current winning design and relocated manifests.
    best_hash=hashlib.sha256((MAIN/'best_design.json').read_bytes()).hexdigest()
    assert best_hash==plan['best_design_sha256']
    study=json.loads((MAIN/'plan.json').read_text(encoding='utf-8'))
    checked=[]
    for batch in study['batches']:
        assert Path(batch['batch']).is_dir()
        for entry in batch['manifest']['runs']:
            for key in ('config','run_dir','record','train_log','plot'):
                assert Path(entry[key]).exists(),(key,entry[key])
            assert hashlib.sha256(Path(entry['config']).read_bytes()).hexdigest()==entry['config_sha256']
            assert (Path(entry['run_dir'])/'ckpt/ckpt_best.pth.tar').is_file()
            checked.append(entry['run_dir'])
    for p in (ROOT/'runs/history/incandescent_T_trials_20261021').rglob('result.pkl'):
        assert (p.parent/'ckpt/ckpt_best.pth.tar').is_file()
    after=[p for base in ('runs','outputs') for p in (ROOT/base).rglob('*') if p.is_file()]
    report={'completed_at':datetime.now().astimezone().isoformat(),'metadata_files_relocated':changed,
            'best_design_unchanged':True,'current_study_runs_checked':checked,
            'after_files':len(after),'after_bytes':sum(p.stat().st_size for p in after),
            'removed_files':plan['delete_files'],'removed_bytes':plan['delete_bytes'],
            'note':'Frozen native/config/model/design snapshots preserved. Navigation changes only; current winning JSON checksum unchanged.'}
    (CLEAN/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (CLEAN/'path_mapping.json').write_text(json.dumps([{k:m[k] for k in ('source','destination')} for m in plan['moves']],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (CLEAN/'README.md').open('a',encoding='utf-8') as stream:
        stream.write(f'\n## 完成结果\n\n已移动{len(plan["moves"])}个目录，删除{plan["delete_files"]}个文件，直接删除{plan["delete_bytes"]/1024**2:.2f} MiB。最佳设计JSON散列不变；当前5个训练阶段的模型、日志、配置和曲线入口检查通过。路径表见path_mapping.json，完整验证见validation.json。\n')
    print(json.dumps({'metadata_files_updated':len(changed),'best_design_unchanged':True,'runs_checked':len(checked),'removed_MiB':round(plan['delete_bytes']/1024**2,2)},indent=2))


if __name__=='__main__':main()
