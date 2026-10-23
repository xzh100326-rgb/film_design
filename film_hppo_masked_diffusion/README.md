# HPPO + 随机掩码填充：193 nm 膜层设计

本目录是新的独立实验版本。原有 `film_hppo`、`film_hppo_step_reward` 和 `film_hppo_step_score` 未修改，也没有复制历史 runs 或恢复旧模型。材料直接读取本项目 `material_example`，默认最多 20 层、193 nm、空气入射、CaF2 基底、0–45° 共 10 个角度、非偏振。

本版实现“全 MASK 开始 → 随机抽取若干空位 → HPPO 填充材料/厚度 → TMM 评分”的逐步生成过程。它借用掩码生成的思想，没有扩散前向加噪、去噪似然损失、监督预训练、重新掩码或修改已填位置；不能据此称为完整的扩散模型，也没有证据表明它一定比顺序 HPPO 更好。掩码并行生成可参考 [MaskGIT 原论文](https://arxiv.org/abs/2202.04200)，本文实现的训练目标依然是强化学习。

## 状态、动作与 EOS

固定数组按入射侧到基底排列。MASK 与 EOS 是不同状态：

- MASK：尚未填充；计算 TMM 时跳过，不代表一层空气或零厚度膜。
- 材料：所选 `.lm` 的编号及连续物理厚度。
- EOS：材料类别之外的一个离散选择；物理膜系截止于最左边的 EOS，EOS 本身没有厚度。

每步环境从第一个 EOS 之前的空位中，不放回随机抽取若干位置。没有 EOS 时，从全部空位中抽取。抽取数量均匀取自 `fill_min..fill_max`；尾部空位不足时取全部剩余空位。位置排序后写入观测，策略看到具体位置再输出动作。同一步的所有选中位置都会写入；新 EOS 后面的写入保留在数组中，但不进入 TMM。

例如：

```text
[MASK, MgF2(35), MASK, SiO2(55), EOS, HfO2(70), MASK, ...]
                         ↓ TMM
空气 → MgF2(35) → SiO2(55) → CaF2
```

每个有效 step 都对当前压缩后的物理膜系评分一次，包括无有效膜层时的裸基底。如果 EOS 前仍有 MASK，回合继续填充前面的空位；后续也可能生成更靠前的 EOS，截断位置随之移动。第一个 EOS 前的空位全部填完后终止；没有 EOS 时，在全部槽位填完后终止。每步至少填一处，所以回合最多 `max_layers` 步。

`min_layers=1` 屏蔽在索引 0 放 EOS；一般情况下，索引小于 `min_layers` 的位置都不能放 EOS，保证终止设计满足最小层数。中间压缩膜系可能少于最小层数，仍计算奖励，但不作为最佳有效设计保存。设为 0 可以把裸基底纳入设计空间。

## 保留原生 HPPO 的方式

沿用 DI-engine 0.5.3 的 hybrid `VAC`、`PPOPolicy`、原生离散/连续 PPO 损失、GAE 和训练流水线。材料组合编码成一个标量分类动作，连续分支输出每个抽中位置的厚度潜变量。默认材料数 M=5，每步最多 K=3：

```text
EOS = M = 5
action_type = token_0 + 6 * token_1 + 36 * token_2
action_args = [u_0, u_1, u_2]
d_j = d_min + (d_max - d_min) * (tanh(u_j) + 1) / 2
```

对应 216 个联合分类动作和 3 个连续参数，原生分类损失仍只接收 `[batch]` 标量动作，避免多分类维度误广播。这里的厚度头按所选位置输出，不是原版本的每材料一个厚度参数；材料与厚度两个分布均由当前状态预测，没有新增材料条件厚度网络。

不足 K 个位置时，未使用的分类位固定为 0，连续参数固定使用 N(0,1)，在新旧策略间概率比为 1，且这些参数没有梯度。EOS 在过早位置的联合分类动作被屏蔽。采样、评估和学习使用相同的掩码。抽取数量和位置是环境的随机上下文，已包含在观测里；不由策略学习。

同一步的 EOS 及其后方选中位置的厚度在物理计算中被忽略；这些已选位置的潜变量仍进入原生连续分支损失。联合材料动作也保留所有选中位置的组合概率。这是保留原生 HPPO 的实现选择，可能带来无物理作用的采样方差。没有另行实现 EOS 条件下的连续损失裁剪。

观测包含每个位置的 M 个材料 one-hot、EOS、MASK、归一化厚度，以及 K×max_layers 个选中位置标记和 `step_count/max_layers`。默认维度是 `20×(5+3)+3×20+1=221`。旧顺序版本 checkpoint 的维度不兼容，应重新训练。

联合分类规模为 `(M+1)^K`，本版限制不超过 4096；大幅增大每步填充数量需要改用因子化或自回归策略，不能直接无上限增大 `fill_max`。

## 奖励和最佳设计

```text
MAE = mean(abs(T - target))
MSE = mean((T - target)^2)
step_reward = 1 - MAE        # reward_metric="mae"，默认
step_reward = 1 - MSE        # reward_metric="mse"
training_return = sum(step_reward)
```

角度和波长网格等权。默认目标全部为 1，MAE 奖励恰好等于当前平均功率透射率，直接对应最大化 193 nm 的角度平均透射率。MSE 奖励侧重减小低透射角度的平方误差；多角度时不等价于最大化平均透射率。目标、入射角和波长固定在每次训练配置中，不是在单次训练中任意变化的条件输入。

按要求保留每步绝对评分及 gamma=1，PPO 实际优化的是各步评分之和，可能偏好较长的填充轨迹；它与“最终透射率最大”并不严格等价。按历史最佳物理评分导出设计能够保留训练中遇到的好膜系，但不能消除奖励目标的这一差别。严格优化最终评分可另做终止奖励或差分奖励对照实验。

每步对满足 `min_layers` 的当前膜系更新历史最佳记录，包括非终止状态。中间状态中的 MASK 被压缩掉后即可作为独立有效物理膜系使用：

- `designs/best_design.json`：当前运行的全局最佳已遇设计；按单步物理评分比较。
- `designs/best_<env-id>.json`：各环境的历史最佳设计。
- 每个文件同时记录可直接重算的 `layers`、完整 `slots`、参与计算的 `active_positions`、EOS 位置和 R/T/A。

训练中持续评估并更新这些文件，评估环境也参与最佳设计搜索；全局文件的更新基于当前同步 `BaseEnvManager`，不要直接改成多进程共享写入。原生周期性 evaluator 的 `eval_episode_return`、停止阈值及最佳策略 checkpoint 仍使用最终状态评分；这与历史最佳设计文件是不同指标。原生 DI-engine 0.5.3 的 hybrid evaluator 也进行随机采样；评估结果不是每次相同的确定性贪心设计。

## 配置

修改 `configs/duv193.json`：

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `material_dir` | `../../material_example` | 相对配置文件所在目录解析 |
| `material_files` | 当前全部 5 个 `.lm` | 可选择其中非空子集；列表顺序决定材料编号 |
| `substrate_file` | `CaF2 (110-800nm, RIT).sub` | 可替换成目录内有相容光学数据的 `.sub` 或 `.lm` |
| `wavelengths_nm` | `[193.0]` | 优化波长 |
| `angles_deg` | 0–45°，10 点 | 可用范围对象或升序角度列表 |
| `target_transmission` | `1.0` | 标量、波长数组或 `[角度,波长]` 数组 |
| `polarization` | `unpolarized` | `(Ts+Tp)/2`，也可为 s 或 p |
| `max_layers` / `min_layers` | 20 / 1 | 固定槽位数和终止设计最小层数 |
| `fill_min` / `fill_max` | 1 / 3 | 每步随机抽取空位数量的范围 |
| `thickness_min_nm` / `thickness_max_nm` | 10 / 140 | 每层物理厚度界限 |
| `reward_metric` | `mae` | 可切换为 mse |

角度范围示例：`"angles_deg": {"start": 0, "stop": 30, "num": 7}`；单角度示例：`"angles_deg": [0]`。允许角度为 `[0,90)`。修改材料顺序、槽位数或动作维度后需要重新训练。

本版不启用旧项目的 400 nm 总膜厚限制或相邻同材料禁用规则。`max_total_thickness_nm` 必须为 null 或省略，`forbid_adjacent_same_material` 必须为 false 或省略；设置为其他值会明确报错。原因是随机插入会改变压缩膜系的相邻关系和预算分配，需要另外设计可行性掩码，而非复用顺序添加规则。当前每层厚度仍严格受限。

`.lm`/`.sub` 按现有 JSON 的 `wavelength/n/k` 读取，线性插值，采用 `n+i k`，不外推或抹掉吸收。TMM 继承原有实现及单位换算。默认 T 是穿过涂层后进入**半无限 CaF2 基底**的功率比例，不包含有限厚度透镜内传播及背面透射。

## 使用

在已安装项目依赖的 yolov11 环境中：

```powershell
conda activate yolov11
Set-Location 'E:\lithography_machine_lens\film_hppo_masked_diffusion'

# 无参数更新：验证材料、TMM、策略、环境管理器与实际采样/GAE
python -m film_rl check --output outputs/check_masked.json

# 行为、光学和临时网络更新回归测试
python -m unittest discover -s tests -v

# 有界训练检查，使用小规模独立 smoke 配置
python -m film_rl train --config configs/smoke.json

# 正式训练
python -m film_rl train --config configs/duv193.json

# 重算 layers 或带 MASK/EOS 的 slots 数组
python -m film_rl evaluate --design examples/design.json --output outputs/example.json
python -m film_rl evaluate --design examples/slots.json --output outputs/slots.json
```

默认正式训练上限为 100 万个交互步、10000 次小批量更新。实际停止检查在采集/学习批次之间进行，可能超过配置上限，幅度取决于批大小及回合长度。结果位于启动时打印的独立 `runs/duv193_masked_hppo_seed1`；目录存在时自动增加时间后缀。TensorBoard：`python -m tensorboard.main --logdir runs`。也可用 `plot_training.py --run <实际运行目录> --output outputs/training.png --reward-as-transmission`；百分数换算仅适用于 target=1 且 metric=mae 的最终物理评分，不能用于累计奖励或 MSE 分数。

数值保护继承原项目：固定 rollout GAE/价值目标、分类动作规范为标量、log(sigma) 限制、更新前 KL 与 log-ratio 检查、梯度与参数有限值检查。触发 KL 提前停止表示停止重复使用当前 rollout，随后继续采集；不保证长时间训练一定收敛。

本机无需依赖 conda 激活命令，也可直接使用已验证的解释器启动正式训练：

```powershell
Set-Location 'E:\lithography_machine_lens\film_hppo_masked_diffusion'
& 'C:\Users\xxxxx\.conda\envs\yolov11\python.exe' -m film_rl train --config configs/duv193.json
```

验证结果：24 项测试通过；完整原生训练流水线已完成 80 个采集交互步及 8 次优化器更新；导出的全局最佳膜系通过独立 TMM 重算，透射率数组误差小于 1e-12。具体检查见 `outputs/tests.log`、`outputs/check_masked.json`、`outputs/validation_summary.json`。`runs/smoke_masked_hppo` 是有界接口验证运行，不是正式优化结果；默认正式配置尚未长时间训练。

核心实现：`film_rl/env.py`（槽位状态和历史最佳）、`actions.py`（联合动作）、`hppo.py`（原生 VAC 与一致掩码）、`optics.py`（TMM 和两种奖励）、`policy.py`（原生 HPPO 数值保护）。接口依据：[DI-engine PPO 源码](https://github.com/opendilab/DI-engine/blob/main/ding/policy/ppo.py)、[TMM-fast 官方仓库](https://github.com/MLResearchAtOSRAM/tmm_fast)。
