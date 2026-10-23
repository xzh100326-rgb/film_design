# 400–2000 nm 吸收率改进日志

日期：2026-10-20。项目：`E:/lithography_machine_lens/film_hppo`。

## 已验证的结果

| 项目 | 400–2000 nm 平均吸收率 |
|---|---:|
| 本次自动选出的原 HPPO 设计（按当前基底重算） | 93.1043% |
| 联合搜索后的最佳结构 | **98.2846%** |
| 预训练 VAC 自主确定性输出 | **98.2800%** |
| 原生 PPO 后续更新后，官方 evaluator 的 8 次平均 | **98.2563%** |
| 更新后模型，另设采样种子的 64 次随机输出平均 | **98.2611%** |
| 上述 64 次随机输出中的最低值 | **98.2105%** |

最佳结构在 1 nm 间隔、1601 点上重算的平均吸收率为 **98.3483%**。161 点 TMM-fast 光谱与独立 Fresnel 递推的最大差异小于 1e-15。达到的是全波段**平均**吸收率目标；最佳结构密网格最低点约 82.53%，并非每个波长均达到 95%。

这些结果来自**结构搜索 + TMM 厚度梯度优化 + 精英轨迹监督预训练 + 原生 PPO 后续更新**。不能写成“纯 HPPO 从随机初始化独立搜索出 98.28%”。没有将论文结构硬编码为初始膜系，也没有修改材料 n/k、吸收率公式或波段来提高分数。仍使用此前从论文图中近似提取的光学常数；结果为仿真值，不代表实际加工后的吸收率。

## 保持的物理条件与约束

- 材料空间和编号：0 MgF2、1 TiO2、2 Si、3 Ge、4 Cr；允许选择这五种材料，结构无需全部使用。
- 五份 YAML 文件内容及 SHA256 不变，继续支持逐波长 n/k。
- 400–2000 nm，间隔 10 nm，共 161 点；正入射、非偏振。
- 固定基底：n=1.45、k=0、500 nm 玻璃，背面空气，沿用此前已修改好的原文代码设置。
- 每层 10–200 nm、镀膜总厚度不超过 600 nm、最多 30 层、至少 1 层、相邻材料不同。
- 评分为全波段平均吸收率，训练奖励为最终一步的线性 `1-MAE`，`reward_alpha=0`。

## 修改的文件和逻辑

1. 新增 `film_rl/improve.py`。
   - 自动发现 `runs/**/best_*.json` 中材料和约束兼容的历史 HPPO 设计，按**当前**基底和数据重新计算，避免使用旧光学条件下的评分。
   - 从这些设计出发进行材料替换、插入、删除，保留多种结构作为候选；未放宽材料或厚度空间。
   - 使用 TMM-fast 的 PyTorch 自动微分计算所有层厚度的梯度；SLSQP 同时处理单层边界及总厚度不等式约束。
   - 优化变量归一化到 0–1，解决以 nm 直接优化时尺度造成的慢收敛；最终仍以 nm 输出。
   - 对预算进行盒约束/总厚度联合投影；优化失败或变差时保留原候选；每个接受结果通过普通 `FilmOptics.evaluate` 重算。
   - 让 VAC 监督学习优良轨迹，按环境剩余厚度预算反解连续动作；验证真实逐步生成的膜系，而非只验证教师动作。
   - 预训练后将独立 Gaussian 的 log-sigma 设置为 -4，使厚度在优良设计附近继续探索。原材料头、厚度头、critic 和动作接口保留。
   - 输出配置、源码、材料哈希、逐候选 `search.jsonl`、最佳结构、模型及策略验证结果；目录已存在时拒绝覆盖。
   - 后续新执行 `improve` 会生成 `warm_start_config.json`，可直接用于 `train`。
2. 修改 `film_rl/__main__.py`。
   - 增加 `improve` 子命令及可选 `--design` 初始设计。
   - `train` 和 `check` 支持 `training.initial_model`；加载时校验材料、光学数据、基底、目标、约束指纹，并严格校验模型参数形状。
   - 只初始化模型权重；不会复用旧策略采样数据或旧优化器状态。
3. 修改 `film_rl/hppo.py`。
   - 有预训练模型时设置原生 `learn.ppo_param_init=False`，防止 DI-engine 初始化覆盖加载的权重。冷启动仍保留原生初始化。
4. 修改 `configs/400-2000absob.json` 的训练部分，并新增 `configs/400-2000absob_cold_start.json` 保留修改前训练设置。

| 参数 | 修改前 | 现在 | 原因 |
|---|---:|---:|---|
| initial_model | 无 | experiment2/warm_start_model.pt | 从已验证的优良策略开始 |
| learning_rate | 3e-4 | 3e-6 | 窄厚度分布对均值漂移敏感，减小后续 PPO 更新 |
| entropy_weight | 0.001 | 0.0001 | 避免预训练后立即扩大厚度随机性 |
| epoch_per_collect | 10 | 3 | 降低同一批数据上的累计漂移 |
| eval_freq | 100 | 20 | 更频繁核查是否保持目标 |
| stop_value | 0.9999 | 0.95 | 对应本次提出的平均吸收率门槛 |
| exp_name | runs/spectral_rta_example | runs/absorption_400_2000_improved | 与旧实验区分 |

`max_train_iter=10000`、`max_env_step=1000000` 仍保留。实际验证因达到评估门槛而提前停止，不是用完这些上限。`improvement` 是新增独立搜索预算配置，不改变环境奖励。

5. 修改 `run_seed_batch.py`：批量准备/执行时额外记录并校验预训练模型文件哈希，保留原代码和材料哈希校验。不同 seed 的热启动实验共享同一个预训练模型，不能当作不同随机初始化的纯 PPO 实验。
6. `pyproject.toml` 明确声明已在环境中可用的 SciPy 优化依赖。
7. 更新 README 使用说明，移除已删除 `spectral_rta_example.json` 的过期引用；对应测试改为读取现有吸收示例配置。
8. 新增 `tests/test_improvement.py`：验证厚度梯度与有限差分一致、约束投影、候选不退化、模型指纹，以及预训练权重经过原生 PPO 初始化后保持不变。全部 70 项测试通过，见 `all_tests.log`。

本次没有改写 `policy.py` 的所选材料连续 PPO 损失。改变训练初始化和增加联合搜索，与更换 PPO 概率项是不同实验因素。

## 实验过程和保留的中间结果

- `before/`：修改前代码和配置备份，包含此前的奖励、基底和训练参数。
- `experiment1/`：第一版联合搜索，厚度变量直接使用 nm，33 个候选、2601 次梯度调用，约 104.75 秒，最佳平均吸收率 96.2687%。保留源码快照。
- `experiment2/`：厚度变量归一化后，35 个候选、1342 次梯度调用，约 46.19 秒（含 1800 次监督预训练），最佳平均吸收率 98.2846%。保留源码快照和配置。
- `native_ppo.console.log` / `runs/absorption_400_2000_improved`：首次热启动接入验证发现 DI-engine 会重新初始化权重，已中止，不是最终实验。该目录未完成，不应作为改进结果。
- `native_ppo_verified.console.log`：关闭重复初始化后的正式验证，学习率 3e-5，8 次评估平均 98.1377%。小 sigma 下漂移保护频繁触发，因此随后将学习率减小到 3e-6。
- `native_ppo_final.console.log` / `runs/absorption_400_2000_improved_261020_141349`：最终配置的原生训练，共 9504 个环境步、27 次优化器更新，最终 8 次评估平均 98.2563%，达到 stop_value 后正常停止。
- `final_policy_validation.json`：对 `iteration_27.pth.tar` 的更新后模型，另外生成 64 个膜系验证，未再更新参数。
- `verification.json`：材料/约束未改变、1 nm 密网格和独立求解器交叉验证。
- `improvement_results.png`、`spectra.csv`：改前/改后/网络输出光谱、搜索进展和层厚。
- `native_ppo_training.png/csv`：正式 PPO 过程曲线；本次达到门槛后很快停止，曲线不代表长期收敛研究。
- `final_manifest.json`：交付时文件哈希、软件版本、最终指标与正式实验路径。

## 运行方法

在项目目录、yolov11 环境中：

```powershell
# 读取已经准备好的优良初始模型，继续原生 PPO
python -m film_rl train --config configs/400-2000absob.json

# 只重新计算最佳设计光谱
python -m film_rl evaluate --config configs/400-2000absob.json --design outputs/absorption_improvement_20261020/experiment2/best_design.json --output outputs/improved_design_evaluation.json

# 用新的目录重新执行联合改进，并可通过 --design 指定历史 HPPO 设计
python -m film_rl improve --config configs/400-2000absob_cold_start.json --output outputs/another_absorption_improvement
python -m film_rl train --config outputs/another_absorption_improvement/warm_start_config.json

# 原始冷启动 PPO 对照
python -m film_rl train --config configs/400-2000absob_cold_start.json
```

这里已经实际生成并验证了超过 95% 的设计和模型。有限预算的随机搜索不能保证任意新配置、新 seed 或后续任意更新均达到相同结果；达到门槛后应保留已验证模型及最佳设计。
