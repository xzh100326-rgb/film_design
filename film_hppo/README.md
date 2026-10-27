# 薄膜设计：DI-engine HPPO + TMM-fast

结果导航：[outputs/README.md](E:/lithography_machine_lens/film_hppo/outputs/README.md)；[runs/README.md](E:/lithography_machine_lens/film_hppo/runs/README.md)；[清理和测试过程](E:/lithography_machine_lens/film_hppo/outputs/maintenance/cleanup_20261021/README.md)。

项目逐层选择材料和厚度，用 TMM-fast 计算完整膜系的 R/T/A 光谱，再按目标光谱评分。训练基于 **DI-engine 0.5.3 的 `PPOPolicy` 和 hybrid `VAC`**；`FilmPPOPolicy` 使用官方 PPO 损失函数和优化器，并让连续策略概率项只对应实际选中材料的厚度，同时处理动作形状与数值保护。

## 当前默认配置

入口为 `configs/duv193.json`，本次清理未修改它。

| 条件 | 当前值 |
| --- | --- |
| 波长 | 193 nm |
| 入射介质 / 基底 | 空气 / `CaF2 (110-800nm, RIT).sub` |
| 角度 / 偏振 | 0–45°，10 点 / 非偏振 |
| 材料 | MgF2、LaF3、GdF3，按配置顺序编号 |
| 目标 | 全角度透射率 1.0 |
| 膜层数 | 1–20 层 |
| 单层 / 总厚度 | 5–100 nm / 不超过 400 nm |
| 相邻材料 | 禁止相同材料相邻 |
| 奖励时机 | 完成整个膜系时计算一次；中间步骤为 0 |

## 安装与运行

在已有的 yolov11 环境中使用：

```powershell
conda activate yolov11
Set-Location 'E:\lithography_machine_lens\film_hppo'
# 新环境安装依赖时使用；已有环境无需重复安装：
python -m pip install -e .

# 选择并检查材料；不会训练
python -m film_rl materials --config configs/duv193.json

# 检查网络、环境、TMM、采集器和 GAE；不更新参数
python -m film_rl check --config configs/duv193.json --output outputs/smoke_check.json

# 运行全部回归测试
python -m unittest discover -s tests -v

# 独立评估一个设计
python -m film_rl evaluate --config configs/duv193.json --design examples/design.json --output outputs/example_spectrum.json

# 只有以下命令启动训练
python -m film_rl train --config configs/duv193.json
```

`check` 验证接口连通和采样数据的有限性，不执行反向传播，也不证明长时间训练一定收敛。

## 材料数据

镀膜材料从本地 refractiveindex.info 数据库读取：

```json
"material_source": "refractiveindex",
"refractiveindex_dir": "../../refractiveindex.info-database",
"materials": [
  {"name": "MgF2", "file": "data/main/MgF2/nk/Rodriguez-de Marcos.yml"},
  {"name": "LaF3", "file": "data/main/LaF3/nk/Rodriguez-de Marcos.yml"},
  {"name": "GdF3", "file": "data/main/GdF3/nk/Franta.yml"}
]
```

`refractiveindex_dir` 可以指向仓库根目录或 `database` 目录；相对目录以配置文件所在目录为基准。`materials` 至少包含两种材料，其顺序决定动作编号。`file` 支持数据库相对路径、材料目录内的文件名及外部测量 YAML 的绝对路径。

不指定 `file` 时，程序按材料文件夹名查找 YAML。只有一个版本时自动读取；多个版本时在终端显示编号、工艺注释及当前波段是否可用，由用户选择。全部选择成功后，只将材料的文件选择写回配置，保留其他设置。删除某条 `file` 可重新选择。所有候选都不可用时直接报错。

`materials` 命令只读取镀膜数据，不运行 TMM；基底的实际数据兼容性由 `check`、`evaluate` 或训练前置检查确认。`--non-interactive` 禁止交互，适合脚本和子进程；未固定的多版本材料会报错。

支持 `tabulated nk`、分别制表的 n/k 和色散公式 1–9。数据库波长单位 μm，项目使用 nm；表格 n/k 分别线性插值，公式按原式求值。**所有材料和基底都必须覆盖目标波长，不允许外推。** 只有 n 的数据必须显式设置 `"assume_zero_k": true` 才可令 k=0，输出会记录该假设。缺失材料不会被自动删去或替换，例如本地缺少 AlF3 时应补充可靠 YAML。

基底仍使用 `material_dir` 下的 `.sub` 表格，保留波长、n、k 的插值和有效性检查。复折射率约定为 `n+i*k`。

单波长固定折射率对照模式仍可使用 `configs/duv193_fixed193.json`，仅适用于 193 nm；其中的数值是历史对照常数，不能全部视为所引文献的原表数据，k=0 是无吸收假设。

**已移除 `.lm` 镀膜读取、`configs/duv193_lm.json`、对应示例和一次性历史诊断脚本。** 旧 `.lm` 配置会明确提示迁移，不会自动套用其他材料。历史训练结果保留；需要旧读取代码时可从项目旁的 `film_hppo_cleanup_backup_*.zip` 恢复，其他项目副本未修改。

## 多波长、多角度与目标光谱

波长、角度均支持递增列表或范围，例如：

```json
"wavelengths_nm": {"start": 400.0, "stop": 2000.0, "num": 161},
"angles_deg": {"start": 0.0, "stop": 45.0, "num": 10},
"polarization": "unpolarized",
"spectral_targets": [
  {"quantity": "R", "wavelength_range_nm": [400.0, 700.0], "target": 1.0, "weight": 1.0},
  {"quantity": "A", "wavelength_range_nm": [800.0, 1200.0], "target": 1.0, "weight": 2.0},
  {"quantity": "T", "wavelength_range_nm": [1500.0, 2000.0], "target": 1.0, "weight": 1.0}
]
```

每个目标先在其波段及全部角度上计算 MAE，再按权重归一化求和。权重不会因波段采样点数较多而自动变大；未指定目标的间隔不计分。波段是闭区间，相接端点会同时属于两个波段，能量守恒矛盾的目标会被拒绝。

统一用 `S_b` 表示第b个目标项的实际光谱，由 `quantity` 选择 `R`、`T` 或 `A`；`S` 是数学符号，JSON 中不要写 `"quantity": "S"`。通用入口为 `SpectralObjective.evaluate_spectra({"R": R, "T": T, "A": A})`，单一响应任务也可只提供所需通道。

```text
MAE_b = mean_over_configured_angles_and_band_wavelengths(abs(S_b - S_target_b))
score = 1 - sum_b(normalized_weight_b * MAE_b)
```

先逐点取绝对误差，再求平均；不是先求波段均值再比较。对于覆盖且不重叠的波段，若要等价于全波段所有采样点统一平均，可将每项weight设为该项采样点数量；否则保留按应用指定的波段权重。TMM完整光谱和目标输入保持不变，design输出保存 `score_formula`、`reward_formula` 和 `spectral_aggregation`，不展开各点数组。

`quantity` 为 R/T/A，`target` 可为标量、该波段的逐波长数组，或 `[角度, 波长]` 二维数组，值域均为 0–1。不写 `wavelength_range_nm` 表示全波段；单独设置 `{"quantity":"A","target":1}` 即优化平均吸收。

有 `spectral_targets` 时优先使用它；否则继续使用 `target_transmission`，支持相同的标量/曲线/二维数组语法。目标是每次训练的固定条件，当前网络没有将目标谱作为输入，修改目标后应重新训练。

评分及训练奖励：

```text
MAE = sum(normalized_weight[j] * mean(abs(actual[j] - target[j])))
score = 1 - MAE
log_reward = -log10(max(MAE, reward_epsilon)) / -log10(reward_epsilon)
reward = (1 - reward_alpha) * score + reward_alpha * log_reward
```

`reward_alpha` 在 [0,1] 内，`reward_epsilon` 在 (0,1) 内。未设置光谱上下限时，原始 score 用于最佳设计、评估和 `stop_value`，混合 reward 用于学习。在纯 T=1 目标下 score 等于目标角度和波长上的平均透射率；混合 R/T/A 目标下它是综合匹配评分。

使用上述纯S光谱误差奖励且不设置上下限时设 `"reward_alpha": 0.0`，于是 `reward=score`。波段权重以本次配置快照为准。其他历史配置中的对数混合选项保留。奖励仍仅在episode结束时产生；不自动运行improve或增强因子计算，HPPO/VAC和PPO更新规则不变。

### 可选的光谱上下限与越界惩罚

每个 `spectral_targets` 项可添加 `bounds`；未添加、空对象 `{}` 或 min/max 均为 null 时不启用惩罚，继续使用原来的奖励、设计选择和原生训练入口。现有配置不需要修改。

```json
"spectral_targets": [
  {"quantity": "R", "wavelength_range_nm": [300, 470], "target": 1.0, "weight": 1.0,
   "bounds": {"min": 0.70, "scope": "band_mean", "penalty_weight": 10.0}},
  {"quantity": "T", "wavelength_range_nm": [480, 700], "target": 1.0, "weight": 1.0,
   "bounds": {"min": 0.70, "max": 1.0, "scope": "band_mean", "penalty_weight": 10.0}},
  {"quantity": "R", "wavelength_range_nm": [710, 3000], "target": 1.0, "weight": 1.0,
   "bounds": {"min": 0.85, "scope": "band_mean", "penalty_weight": 10.0}}
]
```

数值为演示门槛，必须按应用要求设置。`min`、`max` 均可单独省略；值为有限的 0–1 数字，且 min<=max。`scope` 默认为 `band_mean`，先对该波段及配置中的全部角度求均值，再检查上下限；此模式上下限必须为标量。`pointwise` 则在每个波长/角度计算越界平方后求平均，上下限支持标量、该波段逐波长数组或 `[角度, 波长]` 数组。目标必须处于上下限内（band_mean 检查目标均值），矛盾配置会在启动前报错。可接受范围只影响惩罚，原来的 target MAE 仍逐点计算。

```text
base_reward = (1 - reward_alpha) * score + reward_alpha * log_reward
P_i = penalty_weight_i * mean(max(min_i - S_i, 0)^2 + max(S_i - max_i, 0)^2)
reward = base_reward - sum(P_i)
```

band_mean 模式先将 S_i 替换为均值；省略的一侧不产生惩罚。`penalty_weight` 默认 10，必须正且有限，与原来的目标 `weight` 独立，不自动归一化。增加约束项会增加总惩罚，应按需要分配系数。奖励允许为负，不截断到 0。满足范围时罚项为 0，原目标继续引导改善；平方软惩罚不能保证一定找到合格设计。

HPPO、improve 的梯度优化、目录设计筛选和批量导出均支持此机制。启用时优先选择满足全部边界的设计，再按扣罚后的 reward 比较；均未满足则按 reward 比较并明确标注未达标。improve 保存并复核优化过程中遇到的最佳厚度，防止末次迭代丢掉更好的可行解。原始 `score` 始终保留，不能把它当作扣罚后的 reward。

训练使用原生 DI-engine 采集器/学习器和 PPO 损失，边界评估器要求本次评估的**所有 episode**均满足边界，并且平均原始 score 达到 `stop_value`，才判定提前完成；预算耗尽仍会正常结束。improve 的 `target_score` 同样要求设计满足全部边界。满足判断使用 1e-8 数值容差，仅针对配置的采样网格和 scope，最终设计仍需密网格复核。

设计 JSON 增加 `base_reward`、`constraint_penalty`、`constraints_satisfied`、`constraint_violations`；每项记录上下限、实际均值/极值、越界量和惩罚。批量 CSV 增加扣罚 reward、总惩罚、是否满足边界。启用边界的评估曲线标注 `penalized_reward`，不能解释成透射率百分比。添加/修改 bounds 会改变模型问题指纹；旧目标的预训练模型不能直接作为相同任务加载。

R、T 是功率比，非偏振取 s/p 结果的平均。不设置 `substrate_thickness_nm`（或设为 null）时，基底为半无限介质，`A=1-R-T` 只表示有限镀膜层的吸收。设置正数厚度时，基底作为固定有限层进行相干 TMM 计算，背面连接 `exit_n`（默认 1.0）；此时 T 是透过基底背面的功率比，A 包含镀膜及有限基底的吸收。基底不计入可设计膜层数或镀膜总厚度上限。输出包含完整 R/T/A、基底参数、分波段误差及材料来源、哈希和光学常数。

`configs/400-2000absob.json` 使用 OML-PPO 作者代码中的理想玻璃设置：n=1.45、k=0、厚度 500 nm、背面空气。来源：[任务参数](https://github.com/hammer-wang/oml-ppo/blob/master/rl-multilayer/RLMultilayer/taskenvs/tasks.py)及[光学计算](https://github.com/hammer-wang/oml-ppo/blob/master/rl-multilayer/RLMultilayer/utils.py)。这是作者代码的相干计算模型，不是实际厚玻璃的色散或非相干模型。DUV 配置仍使用原来的半无限基底设置。

上面的 JSON 片段可用于设置混合目标。原来的 110–800 nm CaF2 数据不能外推到 2000 nm。

## 400–2000 nm 吸收率改进流程（2026-10-20）

新增 `improve`：从 `runs/**/best_*.json` 中寻找材料编号和约束兼容的 HPPO 历史设计，按当前材料和基底重新评估；通过材料替换、插入、删除搜索结构，使用 TMM-fast 的厚度梯度与 SLSQP 联合优化所有层厚度，严格保留单层和总厚度约束。没有内置论文结构。接受结果均由普通 `FilmOptics.evaluate` 重新验证。

随后训练 VAC 模仿找到的优良轨迹，保存模型并测试它自主输出的膜系。**这是结构搜索、连续优化、监督预训练与 PPO 的联合流程，不是纯 PPO 从零训练达到相同结果的证明。** 预训练模型携带材料、基底、目标与约束的指纹；更改问题后会拒绝直接加载。它只初始化模型权重，不恢复旧 PPO 优化器或经验数据。

```powershell
# 已准备好的改进模型：继续原生 PPO 训练，平均评估吸收率达 95% 后停止
python -m film_rl train --config configs/400-2000absob.json

# 保留的纯 PPO 冷启动对照，不读取改进模型
python -m film_rl train --config configs/400-2000absob_cold_start.json

# 重新进行联合搜索；输出目录必须是新的，也可用 --design 指定一个历史设计
python -m film_rl improve --config configs/400-2000absob_cold_start.json --output outputs/my_absorption_improvement

# 使用该次搜索新生成的模型继续 PPO
python -m film_rl train --config outputs/my_absorption_improvement/warm_start_config.json
```

`improve` 的设置位于可选 `improvement` 对象；搜索有明确预算，达到目标会提前结束，预算耗尽会如实记录未达到目标。`warm_start_steps=0` 可只搜索而不训练模型。重复实验需固定输入设计、随机种子、材料和源码快照。

指定配置与设计目录，重新评分并优化前 5 个设计：

```powershell
python -m film_rl improve --config configs/incandescent_task.json --design-dir runs/incandescent_task/seed0_a1/designs --top-k 5 --non-interactive
```

`--design-dir` 递归读取指定目录中的 JSON（包括设计列表/Pareto 档案），只使用该目录的候选；与单文件 `--design` 互斥。材料编号/名称、层数、相邻材料和厚度必须满足当前配置。所有兼容设计先按当前材料数据及目标重新计算 score，去掉完全相同的设计，再按 score 降序选取前 K 个；相同材料序列的不同厚度仍可入选。目录模式默认 K=5，覆盖配置中的 `improvement.seed_limit`，可用 `--top-k` 更改。不足 K 个时优化全部兼容候选；没有兼容设计则报错，不转为随机起点，也不扫描其他 runs。

若当前配置启用了光谱 bounds，上述排名改为“满足全部边界优先，再按扣罚 reward 排序”，不会沿用历史文件的旧 score/惩罚。

输出目录自动带时间戳，也可通过 `--output` 指定新的目录。`seed_scan.json` 记录扫描、去重和跳过原因；`seed_manifest.json` 记录入选排名、源文件/列表索引、重新计算的 score 与各目标波段 R/T/A。每个入选设计均先独立进行厚度优化，结果保存在 `refined_seeds/seed1.json` 等文件中；`summary.json` 的 `seed_refinements` 提供优化前后 score 和结果路径。之后是否继续材料替换/插层/删层由 `improvement.structure_rounds` 控制；仅优化这些设计的厚度可将它设为 0。最终综合最优设计为 `best_design.json`，是否符合需求还需同时查看各波段物理指标。

当前五材料、单层 10–200 nm、总厚度 600 nm、正入射条件下，联合搜索得到 11 层膜系，161 点平均吸收率 **98.2846%**；网络直接输出 **98.2800%**。1 nm 间隔的密网格及独立 Fresnel 递推均已核对。原生 PPO 后续 8 次评估平均 **98.2563%**。这些是当前近似光学常数下的仿真平均值，不表示每个波长均超过 95%。

具体结果和修改日志：`outputs/abs400-2000__hybrid__pilot/20261020_140821/CHANGELOG.md`。该目录包含改前文件、两次搜索的源代码与配置快照、逐候选日志、模型、验证报告和图。

## 环境与 HPPO

M 种材料对应 M+1 个离散动作，前 M 个选材料，最后一个为 EOS；连续分支输出 M 个高斯潜变量。本步只取所选材料对应的潜变量 u，经过 `(tanh(u)+1)/2` 映射到允许的厚度区间。PPO 保存原始潜变量用于概率计算。

连续 PPO 策略项使用选中材料对应的一维 Gaussian 概率比；未选中的厚度参数不进入该概率比，EOS 的连续策略项为零。按整个 minibatch 平均，EOS 行不会改变其他行的归一化分母；所有行仍参与离散策略和价值函数学习。连续 KL 和 log-ratio 保护同步采用所选参数，EOS 对连续漂移贡献为零。

本次只调整连续策略概率项及其保护检查：仍保留原有全向量 Gaussian 熵正则、统一 entropy_weight 和每种材料的全局 sigma。因此未选中参数的 sigma 仍可能受到熵正则更新，不能将该改动解释成整个网络的所有未选中参数完全不更新。共享编码器及优化器动量也可能改变未来输出。网络形状、动作接口和 checkpoint 参数形状兼容原版。

状态为 `max_layers*(M+1)+1` 维向量，每层记录材料 one-hot 和归一化厚度，末尾记录层数比例。当前三材料、最多 20 层时为 81 维；状态不是光谱。

`FilmVAC` 在采样、评估和学习时使用一致的 EOS/相邻材料掩码。环境同时验证动作，拒绝非法输入且保持原状态。`min_layers=0` 可允许裸基底；`forbid_adjacent_same_material=false` 可取消相邻限制。

总厚度由 `max_total_thickness_nm` 控制，设为 null 或省略即可禁用。添加膜层时厚度映射同时满足单层限制、剩余预算和最少层数的预留厚度。EOS、达到最大层数或预算不足再添加一层时终止，计算一次完整膜系的奖励；非偏振的一次评估包含 s/p 两次底层求解。

训练使用原生 `serial_pipeline_onpolicy` 和同步 `BaseEnvManager`。`training.cuda` 控制网络设备，`tmm_device` 控制光学求解设备。采样后固定 GAE，默认 gamma=1、GAE lambda=0.95。每批 1024 步，按 128 条组成小批量，最多复用 10 遍；KL 或对数概率比超过阈值时提前结束本批复用。

`policy.py` 保留 categorical 动作 `[B]` 形状修正、有限值和梯度检查。离散 KL 为完整 categorical KL；连续 KL 为采样所选 Gaussian 的解析 KL，按批次平均，不是完整混合策略的精确 KL。`hppo.py` 保留高斯尺度限制及注册环境/策略的导入。

## 结果目录和可视化

单次训练写入 `training.exp_name` 指定的项目内相对路径，保存日志、配置、checkpoint 和 `designs/best_<env-id>.json`。各环境保存历史最高 score，在这些设计中按 score 取最大者即可找到最佳已保存候选。

目录已存在时添加时间后缀，日志和 designs 使用同一新目录，启动时打印实际路径。配置原值不变；重复训练不会覆盖旧运行。`Measure-Command` 会重新执行训练并将耗时输出到终端，本身不保存耗时文件。

```powershell
python -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6006
python plot_training.py --run runs/seed0_test1 --output outputs/seed0_test1.png
```

`--run` 应填写实际生成的目录。绘图导出 PNG 和 CSV，默认从设计元数据识别评价指标；只有纯 T=1 目标才使用 `--reward-as-transmission`，也可用 `--evaluation-metric score|transmission|reflection|absorption`。历史缺少元数据的运行需自行选择正确标签。

`plot_reward_comparison.py` 保留为纯 T=1 条件下的奖励函数对比工具，不代表混合目标的透射率。

## 批量 seed

```powershell
# 默认 seed0–seed9，最多同时 3 个训练进程
python run_seed_batch.py --workers 3
# 自定义组名或种子
python run_seed_batch.py --group comparison_A --seeds 0 2 5 --workers 1
# 只准备，不训练；使用打印的目录启动该批次
python run_seed_batch.py --prepare-only
python run_seed_batch.py --batch-dir "outputs/实验组/批次时间" --workers 3
```

脚本按配置自动生成实验组名，每次建立新的北京时间批次，不覆盖旧数据。组名只控制标签；光学、奖励和训练参数仍在 JSON 中设置。材料版本在父进程选择一次，子进程使用固定路径。

```text
runs/<组名>/<批次时间>/seedN/       日志、checkpoint、designs
outputs/<组名>/<同一批次时间>/
  base_config.json                 基础配置快照
  configs/seedN.json                每个 seed 的配置
  manifest.json                    路径及代码/材料/配置哈希
  logs/seedN.train.log              训练输出
  status/seedN.json                 状态、PID、耗时和结果
  designs/seedN.json                每个 seed 选出的单个代表设计（简化 JSON）
  seedN.png / seedN.csv             训练曲线
  results.csv / summary.json        汇总
```

快照只改变 seed 和实验路径，并解析材料数据目录。`--batch-dir` 执行未开始的批次，不能与 `--config`、`--group`、`--seeds` 同用；代码或材料在准备后改变会拒绝启动，应重新准备。它不提供 checkpoint 续训。

批量训练沿用配置中的 `initial_model`、`initial_sigma` 以及独立的离散/连续熵权重。
设置 `design_selection: "lamp_pareto"` 时，兼容读取单个设计和 Pareto 候选列表，按与环境一致的应用指标规则选择代表设计；其余配置按 score 选择。
`results.csv` 同时列出可见光 T/R/A、UV/IR 反射率、五个红外子段反射率及均值目标是否达标。非白炽灯任务这些列为空。
设计 JSON 省略逐角度、逐波长光谱及光学常数数组，保留准确层序、厚度和分波段摘要；训练日志和 checkpoint 仍完整保存。
训练曲线仍显示配置对应的 score/reward 等训练标量，不是所有物理指标随时间的曲线。批量入口不自动执行厚度精调。

```powershell
python summarize_seed_batch.py "outputs/实验组/批次时间"
Get-Content "outputs/实验组/批次时间/logs/seed0.train.log" -Tail 40 -Wait
```

## 文件分工

灯泡滤光片的独立 baseline（反射率曲线与增强因子）见 [BASELINE.md](BASELINE.md)。只输入设计地址计算：`python calculate_enhancement.py --design "设计.json"`，自动寻找原训练配置，默认只在终端输出；加 `--output "outputs/enhancement.json"` 保存JSON。完整图表报告运行 `python evaluate_baseline.py --help` 查看命令；不修改训练奖励。

| 文件 | 职责 |
| --- | --- |
| `film_rl/config.py` | 配置、网格与材料模式验证 |
| `film_rl/refractiveindex.py` | YAML 读取、色散公式、版本选择和材料常数 |
| `film_rl/optics.py` | `.sub` 基底、TMM、物理检查和混合奖励 |
| `film_rl/targets.py` | 多波段 R/T/A 目标、加权评分与冲突检查 |
| `film_rl/env.py` | 状态、动作、终止规则、奖励和最佳设计保存 |
| `film_rl/hppo.py` | hybrid VAC、动作掩码和原生训练配置 |
| `film_rl/policy.py` | 官方 PPO 损失与优化器、所选厚度概率项、形状适配及数值保护 |
| `film_rl/improve.py` | 材料结构搜索、TMM 厚度梯度、精英轨迹预训练与模型指纹 |
| `film_rl/__main__.py` | materials/check/evaluate/train 入口 |
| `run_seed_batch.py` | 批量准备、训练、绘图和汇总 |
| `summarize_seed_batch.py` | 新旧批次的进度及结果汇总 |
| `plot_training.py` | TensorBoard 标量导出 |
| `tests/` | 独立合成 YAML、环境、策略保护、目标光谱和绘图回归测试 |

材料改变、编号改变或状态维度改变后，不应直接套用旧 checkpoint。历史结果及其他项目副本保留，本项目不修改原始材料数据库。

## 可选增强因子奖励（2026-10-26）

默认不变：不写 `reward_objective`，或设为 `"spectral"`，继续使用原光谱 MAE / 对数混合奖励，再减现有 `bounds` 惩罚。

设为 `"reward_objective": "enhancement"` 时：

```text
base_reward = enhancement_factor (chi)
reward = chi - constraint_penalty
```

这里的惩罚仅为已有 `spectral_targets[].bounds` 的越界平方惩罚，不包括目标光谱 MAE。不设置 bounds 时 P=0。`reward_alpha` / `reward_epsilon` 在此模式不影响奖励；原始 `score` 和 `mae` 继续表示光谱匹配程度，另外导出 `enhancement_factor`、`enhancement`（温度等）、`reward_objective` 和公式。奖励可超过 1，保持终止时给奖励，中间步骤为 0。

可选热参数（以下也是默认值）：

```json
{
  "reward_objective": "enhancement",
  "enhancement_reward": {
    "view_factor": 0.95,
    "power_w": 100.0,
    "area_mm2": 20.0,
    "radiating_sides": 2.0,
    "outside_reflectance": 0.0,
    "angle_order": 64
  }
}
```

此模式复用当前波长网格，一次 TMM 使用配置角度与半球 Gauss-Legendre 角度的并集：原配置角度仍用于目标谱 MAE/边界检查；半球权重用于 chi。必须覆盖 400–780 nm，非偏振、空气/膜系/空气且无有限基底。这里使用项目已有 reflection-only 黑体回收模型，并未增加膜吸收/再发射模型；有吸收时不能把结果解释为已验证的真实灯泡效率。波段外反射率与双面面积仍是显式模型假设。最终候选应使用独立脚本加密波长和角度重新验证。

`training.stop_value`、`improvement.target_score` 在增强模式均以扣罚后的 reward 为单位（例如 12，而非 0.99）。有边界时，依然要求所有评估 episode 满足边界才能提前停止；最佳设计依然优先可行，再按 reward 排序。无边界时直接按 reward 排序。`design_selection` 保持 `score`，内部会根据奖励模式选择实际排序指标，不能使用 `lamp_pareto` 覆盖 chi 排序。奖励模式及热参数进入预训练模型的问题指纹，不能直接加载原光谱任务的预训练快照。

现成独立配置：`configs/incandescent_enhancement.json`，保留原 29 层设计对应的物理条件和光谱边界，仅切换奖励，停止目标示例为 12。原配置及历史设计均未切换。运行（在 film_hppo 目录、film 环境）：

```powershell
# 单设计评估；不训练
python -m film_rl evaluate --config configs/incandescent_enhancement.json --design outputs/incandescent_improvement/best_design.json --output outputs/enhancement_reward_evaluation.json

# 显式启动训练
python -m film_rl train --config configs/incandescent_enhancement.json
```

`improve` 的增强模式使用 SLSQP 数值差分梯度，每次扰动都会重新计算 chi，明显慢于原光谱目标的 TMM 自动微分；原光谱模式仍走原自动微分路径。绘制增强模式评估曲线时不要使用 `--reward-as-transmission`，reward 不是透射率百分比。

## 批量运行的 DLL 启动检查与补绘图

批处理启动前会在同一 Python 环境检查 torch、matplotlib、TensorBoard 是否可导入；失败则在启动训练前停止。`complete_plot_failed` 表示训练已完成、只有绘图失败，模型和设计保留；`training_status` / `plot_status` 单独记录。增强因子曲线支持 `enhancement_minus_penalty`，不按百分比绘制。

只补绘已完成训练（不会重训，也不会启动尚未运行的 seed）：

```powershell
python run_seed_batch.py --recover-plots outputs/incandescent_enhancement/20261026_204554
```

原绘图错误日志保留，补绘日志为 `logs/seedN.recovery.plot.log`。若 Windows Smart App Control 正在拦截 DLL，补绘前也会检查并报错，须先处理系统拦截。应用程序控制策略不同于 Python 奖励或模型错误，不要通过修改 DLL 名称、清空安全日志或关闭杀毒软件来掩盖问题。
