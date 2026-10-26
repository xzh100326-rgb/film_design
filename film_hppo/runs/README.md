# 训练原始数据入口

正式实验保留designs、模型checkpoint、result.pkl和TensorBoard日志。图表和报告从outputs/README.md进入。

| 目录 | 内容 |
|---|---|
| duv193__hppo__nk_v1__a00-45 | 旧nk的10个seed |
| duv193__hppo__nk_v2__a00-45 | 当前nk的10个seed |
| abs400-2000__hppo__baseline | 吸收任务代表性基线 |
| incandescent_task | 早期灯泡批量训练 |
| incandescent_task2__hppo__terminal | 原R目标训练基线 |
| lamp_R_hppo_l095 | 当前实验lambda=0.95的2个seed |
| lamp_R_hppo_l100 | 当前实验lambda=1.0的2个seed |
| lamp_R_hppo_l100_continue_s0 | 纯HPPO模型权重继续训练 |
| history/incandescent_T_trials_20261021 | 旧T目标12次参数测试 |

[当前最好设计](E:/lithography_machine_lens/film_hppo/outputs/hppo_spectral_search/20261021_145752/best_design.json)；[所有结果索引](E:/lithography_machine_lens/film_hppo/outputs/README.md)。

历史T测试保留ckpt_best.pth.tar，初始化/中间/末次checkpoint已清理；当前R目标、DUV和吸收正式实验的checkpoint保持原样。更早完整归档仍在archive/results_before_organization_20261020。
