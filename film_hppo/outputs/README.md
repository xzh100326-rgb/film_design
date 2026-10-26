# 实验结果入口

正式结果按任务/实验组保留；每组的参数副本、图、数值表放在同一批次中。

## 当前灯泡滤光片

- [四目标多seed HPPO及论文结构对照](E:/lithography_machine_lens/film_hppo/outputs/lamp_four_targets/20261021_195347/README.md)：本次训练组，包含分开的熵参数与显式记录的迁移分支。
- [加入可见透射率目标后的最终训练诊断](E:/lithography_machine_lens/film_hppo/outputs/incandescent_hppo_gae1/diagnosis_20261021_193508/README.md)：180480环境步、7200次更新；含初始/最佳/最终检查点对照及探索收缩图。
- [当前 HPPO 最优设计的 improve 对照](E:/lithography_machine_lens/film_hppo/outputs/incandescent_improve/20261021_172001/README.md)：独立保存结构/厚度优化结果及前后光谱比较。
- [当前最佳膜系：80层，MAE=0.26755](E:/lithography_machine_lens/film_hppo/outputs/hppo_spectral_search/20261021_145752/best_design.json)
- [纯HPPO结果、曲线和过程](E:/lithography_machine_lens/film_hppo/outputs/hppo_spectral_search/20261021_145752/RESULTS.md)；[增强因子差距诊断](E:/lithography_machine_lens/film_hppo/outputs/hppo_spectral_search/20261021_145752/ENHANCEMENT_GAP.md)
- GAE对照与继续训练的批量输出统一放在 [batches](E:/lithography_machine_lens/film_hppo/outputs/hppo_spectral_search/20261021_145752/batches)；对应原生训练数据位于runs/lamp_R_hppo_*。
- [历史T目标参数试验](E:/lithography_machine_lens/film_hppo/outputs/incandescent_hppo_experiments/20261021_004517/RESULTS.md)：12次测试的配置、最佳模型、曲线和最终验证保留，批量输出已收进该目录的batches/。
- [历史混合搜索对照](E:/lithography_machine_lens/film_hppo/outputs/incandescent_improve/20261021_094617/RESULTS.md)，与纯HPPO分开。
- [增强因子baseline](E:/lithography_machine_lens/film_hppo/outputs/lamp_baseline/README.md)。

## DUV 193 nm 与吸收任务

| 任务 | 结果位置 |
|---|---|
| DUV旧nk、10 seeds | [nk_v1](E:/lithography_machine_lens/film_hppo/outputs/duv193__hppo__nk_v1__a00-45) |
| DUV当前nk、10 seeds | [nk_v2](E:/lithography_machine_lens/film_hppo/outputs/duv193__hppo__nk_v2__a00-45) |
| 400–2000 nm吸收HPPO基线 | [baseline](E:/lithography_machine_lens/film_hppo/outputs/abs400-2000__hppo__baseline) |
| 吸收多seed曲线与汇总 | [multiseed](E:/lithography_machine_lens/film_hppo/outputs/abs400-2000__hppo__multiseed) |
| 吸收混合搜索 | [pilot](E:/lithography_machine_lens/film_hppo/outputs/abs400-2000__hybrid__pilot) / [search](E:/lithography_machine_lens/film_hppo/outputs/abs400-2000__hybrid__search) |

[上周展示汇总](E:/lithography_machine_lens/film_hppo/outputs/week_20261014_20261020/README.md)；[材料和论文来源](E:/lithography_machine_lens/film_hppo/outputs/references)。参考资料位置保留，避免影响材料转换和baseline。

## 清理及维护

[本次清理清单和测试过程README](E:/lithography_machine_lens/film_hppo/outputs/maintenance/cleanup_20261021/README.md)；[代码变更验证记录](E:/lithography_machine_lens/film_hppo/outputs/maintenance/README.md)。
新增正式实验继续使用“任务__方法__版本/批次时间/seedN”；临时检查使用outputs/maintenance下独立目录，结束后将结论写进README。
