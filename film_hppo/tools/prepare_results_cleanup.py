"""Prepare a bounded cleanup of completed result artifacts; no deletion here."""
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from film_rl.design import compact_design

STAMP='20261021'
OUT=ROOT/'outputs/maintenance'/f'cleanup_{STAMP}'
MAIN='outputs/hppo_spectral_search/20261021_145752'
TRIAL='outputs/incandescent_hppo_experiments/20261021_004517'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    p=ROOT/p if not isinstance(p,Path) else p
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def md(p,s):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
def files(p):return sorted(x for x in p.rglob('*') if x.is_file()) if p.is_dir() else [p]


def main():
    if OUT.exists():raise FileExistsError(OUT)
    original=[p for base in ('runs','outputs') for p in (ROOT/base).rglob('*') if p.is_file()]
    inventory=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in original]
    OUT.mkdir(parents=True)
    write(OUT/'inventory_before.json',inventory)
    moves=[];deletes=[];captured={}
    def move(a,b):
        p=ROOT/a
        if p.exists():moves.append({'source':a,'destination':b,'files':[{'path':q.relative_to(p).as_posix() if p.is_dir() else '', 'bytes':q.stat().st_size,'sha256':sha(q)} for q in files(p)]})
    for p in sorted((ROOT/'runs').glob('incandescent_T_*')):
        move(p.relative_to(ROOT).as_posix(),f'runs/history/incandescent_T_trials_{STAMP}/{p.name}')
    for p in sorted((ROOT/'outputs').glob('incandescent_T_*')):
        move(p.relative_to(ROOT).as_posix(),f'{TRIAL}/batches/{p.name}')
    for name,label in [('lamp_R_hppo_l095','l095'),('lamp_R_hppo_l100','l100'),('lamp_R_hppo_l100_continue_s0','l100_continue_s0')]:
        move('outputs/'+name,f'{MAIN}/batches/{label}')
    move('outputs/reward_S_update_20261021','outputs/maintenance/reward_S_update_20261021')
    move('outputs/improvement_20261021_152429','outputs/incandescent_improve/incomplete_20261021_152429')
    def mapped(s):
        for m in sorted(moves,key=lambda x:len(x['source']),reverse=True):
            if s==m['source'] or s.startswith(m['source']+'/'):return m['destination']+s[len(m['source']):]
        return s
    def remove(a,reason):
        p=ROOT/a
        if not p.exists():return
        if any(a==x['original'] or a.startswith(x['original']+'/') for x in deletes):return
        # Every pre-existing README/MD under a discarded tree survives separately.
        for q in files(p):
            if q.suffix.lower()=='.md':
                rel=q.relative_to(ROOT).as_posix()
                md('outputs/maintenance/cleanup_20261021/preserved_readmes/'+rel,q.read_text(encoding='utf-8-sig'))
        deletes.append({'original':a,'path':mapped(a),'reason':reason,'is_directory':p.is_dir(),
                        'files':[{'path':q.relative_to(p).as_posix() if p.is_dir() else '', 'bytes':q.stat().st_size,'sha256':sha(q)} for q in files(p)]})

    # Retain each historical test's best model, designs, TensorBoard, result,
    # configs and curves. Only its initialization/intermediate checkpoints go.
    for p in sorted((ROOT/'runs').glob('incandescent_T_*/**/ckpt/iteration_*.pth.tar')):
        assert (p.parent/'ckpt_best.pth.tar').is_file()
        remove(p.relative_to(ROOT).as_posix(),'旧T目标参数测试：保留ckpt_best，删除随机初始化和中间/末次检查点')
    for rel,reason in [
        ('outputs/check_runtime','一次性运行检查日志'),
        ('outputs/design_export_validation','完整/精简JSON导出检查的临时输出'),
        ('outputs/incandescent_task2_check.json','终局奖励smoke check的完整光谱临时输出'),
        ('outputs/incandescent_hppo_experiments/20261021_004502','正式试验前的早期基准快照；正式报告已保留'),
        (TRIAL+'/validation/20261021_004754','未完成时的中间验证；保留最终013033验证'),
        (TRIAL+'/validation/20261021_011211','未完成时的中间验证；保留最终013033验证'),
        (MAIN+'/runtime_audit_partial.json','最终training_audit.json已包含完整审计'),
    ]:remove(rel,reason)

    # Preserve compact copies of unique pre-formatting designs before dropping
    # giant raw-grid backups; equality checks protect layers and score.
    for manifest in (ROOT/'outputs/design_format_backups').glob('*/manifest.json'):
        records=json.loads(manifest.read_text(encoding='utf-8'))
        write(OUT/'design_format_conversion_manifest.json',records)
        converted=[]
        for record in records:
            p=Path(record['backup']);raw=json.loads(p.read_text(encoding='utf-8-sig'))
            d=compact_design(raw)
            assert d['layers']==raw['layers'] and d['score']==raw['score']
            target=OUT/'retained_legacy_designs'/p.name
            write(target,d)
            converted.append({'original':str(p),'original_sha256':sha(p),'compact':str(target),'score':d['score']})
        write(OUT/'design_format_conversion_verified.json',converted)
    remove('outputs/design_format_backups','旧完整光谱备份已精简保留layers/score/指标，并验证相等')

    for name in ('20261021_comparison01','20261021_comparison02'):
        p=ROOT/'outputs/lamp_baseline'/name
        for leaf in ('baseline.json','baseline_settings.json','comparison.csv','README.md'):
            q=p/leaf
            if q.exists():
                target=ROOT/'outputs/lamp_baseline/history'/name/leaf
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(q.read_bytes())
        remove(p.relative_to(ROOT).as_posix(),'已被comparison03替代的baseline光谱、图及临时文件；原指标/设置/README已保留')
    resolution=ROOT/'outputs/lamp_baseline/20261021_resolution_check'
    for p in files(resolution):
        if p.suffix in ('.npz','.png') or p.name.startswith('spectra_'):
            remove(p.relative_to(ROOT).as_posix(),'分辨率收敛检查的派生光谱/图；保留结果表、设置和README')
    for p in sorted((ROOT/'outputs').glob('*.console.log')):remove(p.relative_to(ROOT).as_posix(),'重复终端输出；对应结果表与README保留')
    remove('outputs/baseline_resolution01.json','分辨率设置已保存在resolution_check/baseline_settings.json')

    # Drop derived inference spectra; corresponding best designs and episode
    # tables survive. Main best_spectra and final baseline spectra are retained.
    for base in (ROOT/TRIAL/'policy_evaluation',ROOT/MAIN/'policy_evaluation'):
        for p in base.rglob('best_sampled_spectra.npz'):
            assert (p.parent/'best_sampled_design.json').is_file()
            remove(p.relative_to(ROOT).as_posix(),'逐模型推理派生光谱可由已保存结构重算；保留统计与逐episode CSV')
    for number in (2,3):
        for suffix in ('csv','npz'):
            old=ROOT/MAIN/f'baseline_stage1/spectra_{number:02d}.{suffix}'
            new=ROOT/MAIN/f'baseline_stage2/spectra_{number:02d}.{suffix}'
            if old.exists() and new.exists():
                if suffix=='npz':
                    import numpy as np
                    with np.load(old) as a,np.load(new) as b:
                        assert set(a.files)==set(b.files)
                        assert all(np.array_equal(a[k],b[k]) for k in a.files)
                else:assert old.read_bytes()==new.read_bytes()
                remove(old.relative_to(ROOT).as_posix(),'与baseline_stage2数值完全相同的旧基准光谱副本')
    for p in (ROOT/MAIN).glob('*.log'):
        if p.name!='gae_tests.log':remove(p.relative_to(ROOT).as_posix(),'重复实验控制台输出；正式训练日志和EXPERIMENT_LOG/RESULTS保留')
    cache=ROOT/MAIN/'dense_validation/cache'
    if cache.exists():
        history=[{'cache_name':p.name,'design':json.loads(p.read_text(encoding='utf-8'))} for p in sorted(cache.glob('*.json'))]
        target=ROOT/MAIN/'dense_validation/candidate_history.jsonl.gz'
        with gzip.open(target,'wt',encoding='utf-8') as stream:
            for row in history:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        with gzip.open(target,'rt',encoding='utf-8') as stream:assert [json.loads(x) for x in stream]==history
        remove(cache.relative_to(ROOT).as_posix(),'分散缓存已无损合并为candidate_history.jsonl.gz')

    # Small, exact metadata backup before navigation paths are changed. Frozen
    # configs, native run snapshots, model/design files are never rewritten.
    metadata_names={'plan.json','manifest.json','summary.json','best_result.json','final_summary.json','latest_validation.json','results.csv','comparison.csv','README.md','RESULTS.md','EXPERIMENT_LOG.md'}
    for p in original:
        if p.name in metadata_names or '/status/' in p.as_posix():
            if p.suffix in ('.json','.csv','.md') and p.stat().st_size<2_000_000:
                captured[p.relative_to(ROOT).as_posix()]={'sha256':sha(p),'text':p.read_text(encoding='utf-8-sig')}
    with gzip.open(OUT/'navigation_metadata_before.json.gz','wt',encoding='utf-8') as stream:json.dump(captured,stream,ensure_ascii=False)
    plan={'project_root':str(ROOT),'created_at':datetime.now().astimezone().isoformat(),
          'moves':moves,'deletes':deletes,'delete_files':sum(len(x['files']) for x in deletes),
          'delete_bytes':sum(f['bytes'] for x in deletes for f in x['files']),
          'original_files':len(inventory),'original_bytes':sum(x['bytes'] for x in inventory),
          'best_design_sha256':sha(ROOT/MAIN/'best_design.json'),'rewrite_metadata_names':sorted(metadata_names)}
    write(OUT/'plan.json',plan)
    md('outputs/maintenance/cleanup_20261021/README.md',
       '# 2026-10-21 结果清理与测试过程\n\n'
       '目的：正式结果与临时检查分开；保留当前最佳设计地址、正式训练模型、各次测试的最佳模型/设计/曲线和测试说明。\n\n'
       '## 保留的测试过程\n\n'
       '- T目标12次HPPO参数试验：epoch=5/10各3 seed；熵系数0.003/0.01、IR权重6、最少2/4层、学习率3e-4各1次。仍保留原EXPERIMENT_LOG.md、RESULTS.md、配置和逐次best checkpoint。\n'
       '- R目标GAE试验：lambda=0.95/1.0各2 seed，另继续训练seed0；971512环境步。最佳80层、MAE=0.26755333、chi(f=1)=9.32641459。正式checkpoint、TensorBoard和原生训练日志保留。\n'
       '- baseline分辨率：2nm/64角度与1nm/128角度，原有comparison表保留；对应chi相对差小于0.02%。\n'
       '- JSON导出：精简格式保留layers与score；旧完整光谱备份转换为retained_legacy_designs，并逐个验证相等。\n'
       '- 临时runtime/smoke检查、未完成的中间评估和被替代baseline：记录原路径/散列/清理理由；原有Markdown保存在preserved_readmes。\n\n'
       '## 清理证据\n\n'
       'plan.json为精确移动/删除清单；inventory_before.json为清理前文件散列；navigation_metadata_before.json.gz保存被整理的原始索引文字。\n'
       '删除的派生光谱可由保留的具体膜系、材料数据和配置重新计算；candidate_history.jsonl.gz无损保存原来的全部候选缓存。\n'
       '旧T试验仅保留ckpt_best，删除初始化及中间/末次checkpoint；当前R目标实验不裁剪checkpoint。\n')
    print(json.dumps({'moves':len(moves),'delete_files':plan['delete_files'],'delete_MB':round(plan['delete_bytes']/1024**2,2),'plan':str(OUT/'plan.json')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
