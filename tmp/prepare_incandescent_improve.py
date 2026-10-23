from copy import deepcopy
import json
from pathlib import Path
import sys

project=Path('E:/lithography_machine_lens/film_hppo')
sys.path.insert(0,str(project))
from film_rl.config import load_config

root=project/'outputs/incandescent_improve/20261021_094617'
cfg=load_config(project/'outputs/incandescent_hppo_experiments/20261021_004517/evaluation_config.json')
(root/'evaluation_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
for seed in (0,1):
    trial=deepcopy(cfg)
    trial['angles_deg']=list(range(2,90,5))
    trial['training']['seed']=seed
    trial['improvement']=dict(seed=seed,target_score=.9,seed_limit=8,structure_rounds=4,
                              candidates_per_round=12,beam_width=3,max_thickness_iterations=40,
                              warm_start_steps=0)
    (root/f'config_search{seed}.json').write_text(json.dumps(trial,indent=2),encoding='utf-8')
(root/'CHANGELOG.md').write_text('''# 灯泡滤光片 improve 对照实验

目标沿用上一轮：UV R=1、480–700 nm T=1、IR R=1，波段权重1:4:3。
材料数据、边界介质、最大45层、单层15–400 nm、无总厚度上限、相邻材料不重复均保持不变。

## 必要修复（原文件保存在 before/）

- 修复 discover_seeds 在 max_total_thickness_nm=null 时对 None 进行加法，导致有效HPPO设计被跳过的问题。
- 按当前目标重新评价候选后再排序截取，不用旧目标得分排名。
- 优先选取不同材料顺序，避免8个起点都是同一单层结构的厚度微小变化。
- 关闭VAC模仿时，方法记录不再声称执行了模仿；额外保存seed_manifest.json以追踪HPPO起点。
- 新增候选发现的回归检查。HPPO网络与训练损失未修改。

## 实验设置

两次独立结构搜索（搜索seed0、seed1），各8个HPPO起点、4轮、每轮12个候选、beam=3、厚度优化最多40次迭代。
训练模式关闭：warm_start_steps=0，不复训HPPO，也不训练VAC模仿模型。
搜索使用18个角度分箱中心，波长仍271点；最后用原90角度、271波长复核并进行密集网格厚度优化。
没有把论文42层结构用作起点。已有HPPO设计均在当前材料和目标下重新评价。

此方法是HPPO候选 + 离散结构邻域搜索 + TMM梯度厚度优化，不能表述为纯HPPO训练达到了结果。
上一轮纯HPPO最好保存设计的统一得分约0.62676255，作为对照基线。
''',encoding='utf-8')
print(root)
