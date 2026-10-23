# 193 nm DUV 膜层设计：每步直接评分版 film_hppo_step_score

本目录完整复制自 `E:/lithography_machine_lens/film_hppo`，每个有效 step 都重新计算当前整个膜系的 TMM，直接返回 `step_reward = 1 - MAE`，包括 EOS。没有增量差值、裸基底基准或额外膜层惩罚。源项目和此前的增量奖励分支 `film_hppo_step_reward` 未修改。`configs/duv193.json` 与复制时的源文件完全一致：当前最多 20 层、0–45° 共 10 点、单层 10–140 nm、总厚度 400 nm、禁止相邻同材料、seed1、训练上限 10000 次。实验名也保留原值，实际输出位于本副本下。

复制的 `runs/`、旧 `outputs/` 和历史诊断结果属于原终止奖励实验，不代表本版已经训练。不要把复制的历史曲线当作本版结果；新的训练仍调用原有命令，不自动恢复旧 checkpoint。`diagnostics/audit_multiangle.py` 是原终止奖励版本的历史检查，不适用于本版逐步奖励；本版请使用 `check` 和测试。

本项目逐层构建膜系，基于 **DI-engine 0.5.3 的 `PPOPolicy` 与 hybrid `VAC`（源码中标注 HPPO）** 联合输出材料和厚度参数，每个有效 step 的当前膜系由 **TMM-fast 0.3.0** 评估。`FilmPPOPolicy` 继承原生策略并为小批量更新添加数值保护，仍调用原生离散/连续 PPO 损失及优化器。默认 `check` 不训练；更新回归测试仅使用临时模型，不生成训练 checkpoint。

## 当前设计条件

| 参数 | 配置 |
| --- | --- |
| 波长 | 193 nm |
| 入射介质 | 空气，n=1 |
| 基底 | 材料目录中的 `CaF2 (110-800nm, RIT).sub` |
| 入射角 | 当前 0–45°，10 个等间隔点 |
| 偏振 | 默认非偏振，功率透射率取 `(T_s + T_p)/2` |
| 目标透射率 | 默认 1.0，即 100%，可修改 |
| 膜层数量 | 当前至少 1 层、最多 20 层 |
| 单层厚度 | 当前配置 10–140 nm，连续值 |
| 总膜厚上限 | 400 nm，所有膜层物理厚度之和 |
| 相邻材料 | 默认禁止连续两层使用同一种材料 |

配置入口：`configs/duv193.json`。原始 `.lm` 和 `.sub` 文件从 `E:/AI+film design/material_example` 读取，不复制或修改。

## 动作、状态和奖励

材料顺序显式固定，新增 `.sub` 或其他文件不会改变动作编号：

| action_type | 含义 |
| --- | --- |
| 0 | Al2O3 |
| 1 | GdF（沿用原文件名） |
| 2 | HfO2 |
| 3 | MgF2 |
| 4 | SiO2 |
| 5 | EOS，结束当前膜系 |

DI-engine 接收如下混合动作：

```python
action = {
    "action_type": 3,
    "action_args": [u_Al2O3, u_GdF, u_HfO2, u_MgF2, u_SiO2],
}
```

五个连续参数为网络采样的高斯潜变量。只使用本次所选材料对应的参数：

```text
d_max_effective = min(thickness_max_nm, remaining_budget)
d_nm = thickness_min_nm + (d_max_effective - thickness_min_nm) * (tanh(u_selected) + 1) / 2
```

有限潜变量一般映射到有效厚度范围的开区间，浮点饱和时可取边界；剩余预算恰好等于最小厚度时允许固定厚度。环境向 PPO 保留原始潜变量用于概率计算，不把物理厚度替换回动作；EOS 忽略连续参数。该实现直接沿用 DI-engine hybrid PPO 的连续向量损失，未选中的参数和 EOS 时的参数仍会进入原生连续分支的概率/熵计算；没有额外实现“按所选材料裁剪连续损失”的算法变体。

状态长度为 `max_layers × 6 + 1`，当前 `20 × 6 + 1 = 121`，类型为 float32：每个位置包含 5 维材料 one-hot 和 1 维归一化厚度，末尾是 `当前层数 / 最大层数`。空位置全零。膜层顺序和已有厚度均保留，固定目标、入射条件和材料谱由配置提供。

`forbid_adjacent_same_material=true` 禁止相邻重复材料：例如 MgF2 → MgF2 不合法，MgF2 → SiO2 → MgF2 合法。`FilmVAC` 根据状态中最后一层的 one-hot，将该材料的离散 logit 屏蔽；采样、评估、学习和 KL 检查使用同一掩码。第一层可以选任意材料，EOS 仍按最小层数规则处理。环境拒绝手动传入的重复材料动作且不改变膜系，`random_action()` 也只采样合法动作。独立 `evaluate` 同样检查该限制；如需重算历史含相邻重复层的设计，请使用单独配置设置 `forbid_adjacent_same_material=false`，不必修改原始设计文件。网络输入维度、厚度参数和奖励公式保持不变；此约束应在新启动的训练中使用。

总膜厚通过 `max_total_thickness_nm=400.0` 设置；设为 `null` 或省略此字段可禁用。每次添加膜层时，将厚度潜变量映射到单层范围与剩余预算共同允许的区间；若 `min_layers>1`，还会为尚需添加的最小层数预留厚度。已有材料/厚度信息可以确定剩余预算；网络结构未改变。结果 JSON 的 `total_thickness_nm` 记录总厚度，独立 evaluate 也会拒绝超出上限的膜系。历史超厚设计如需重算，应使用禁用总厚度限制的独立配置。

每个有效 step 都对当前全部膜层计算一次 R/T/A，并返回当前绝对评分。添加第 1 层、第 2 层等步骤均有奖励；EOS 不添加膜层，但仍重新评价现有膜系并返回相同评分。EOS、达到 `max_layers` 或剩余预算不足添加最薄一层时终止。预算终止原因记录为 `thickness_budget`，不要求必须用满 400 nm。非偏振分别执行 s、p 两次底层 TMM 求解，再平均功率结果，属于一次膜系评估。未达到 `min_layers=1` 时屏蔽 EOS；非法动作不会计算奖励。reset 不计算裸基底基准。

```text
MAE = mean_over_angles_and_wavelengths(abs(T - T_target))
step_reward = 1 - MAE
training_return = sum(step_reward for every valid step, including EOS)
```

当目标全部为 1 时，每步 reward 等于当前膜系的角度/波长平均透射率。评分降低时返回降低后的正评分，不返回两次评分差。`discount_factor=1.0` 保持原值，PPO 优化各步评分之和，因此可能偏向较长生成过程；本版按要求不增加长度修正、增量奖励或其他惩罚。角度和波长采样点等权，其他训练超参数保持源配置。

训练奖励与最终设计评价明确区分：`timestep.reward` / `info.step_reward` 是当前评分；`info.training_return` 是累计训练奖励；终止时的 `eval_episode_return` 保持最终膜系评分，评估均值、`stop_value`、最佳模型选择及 `designs` 保存仍按最终评分，不按可能超过 1 的累计值。新增日志 `evaluator_iter/training_return_mean` 显示累计训练奖励，`evaluator_iter/final_score_mean` 显示最终评分。

## 光学计算约定

- 膜层按**入射侧 → 所列膜层 → CaF2 基底**排列，和沉积顺序可能相反。
- `.lm` / `.sub` 按 JSON 中的 `wavelength/n/k` 表读取，显式采用线性插值，复折射率使用 `N=n+i k`；不外推、不把 k 强制设零。193 nm 由相邻表格波长插值得到，精度受原始数据限制。
- 波长与膜厚在进入 TMM-fast 前统一由 nm 换算为 m，入射角由度换为弧度。
- 入射介质及基底厚度设为无穷大，基底使用 `.sub` 中的光学常数。本项目的 T 是**进入半无限 CaF2 基底的功率比例**，不是穿过整片有限厚度透镜及其背面的总透射率。R/T/A 均以入射功率归一化；此处 A 对应涂层区域的损耗，未传播到基底深处或计算背面。
- 数值异常直接报错，不静默返回高奖励。验证包括裸界面 Fresnel 解析结果、无损四分之一波膜的解析反射率及能量守恒。

## 在已配置的 yolov11 环境中使用

无需重新安装已配置的依赖。在 PowerShell 中：

```powershell
conda activate yolov11
Set-Location 'E:\lithography_machine_lens\film_hppo_step_score'

# 只验证配置、HPPO 前向、环境管理器、采样/GAE 和 TMM，不更新参数
python -m film_rl check --output outputs/smoke_check_step_score.json

# 行为、物理与边界测试，不更新网络
python -m unittest discover -s tests -p test_step_score.py -v

# 全部测试：额外包含临时模型的少量反向传播/更新和故障注入测试
python -m unittest discover -s tests -v

# 对一个指定膜系做光学评估
python -m film_rl evaluate --design examples/design.json --output outputs/example_spectrum.json
```

`check` 中构建策略并初始化 DI-engine 学习侧的共享状态，以便检查采样/GAE 接口，但不调用学习前向、反向传播或优化器更新，并检查模型参数没有变化。检查每个有效 step 的 TMM 次数和 `1-MAE` 奖励，输出 `step_trace`。随机膜系仅用于接口验证，不是优化设计。无更新的采样检查可能显示 `train_iter=0`，日志位于 `outputs/check_step_score_runtime`。

需要训练时，单独运行下列命令：

```powershell
python -m film_rl train --config configs/duv193.json
```

训练调用原生 `serial_pipeline_onpolicy`，采用同步 `BaseEnvManager`，避免 Windows 子进程启动问题。默认 CPU，可在配置中分别修改 `training.cuda`（网络）和 `tmm_device`（光学计算）。单个 193 nm 点的小膜系通常无需把 TMM 切换到 GPU。

训练输出位于本副本的 `training.exp_name`，当前为 `runs/seed1_test1_0_45angle`；若已存在，按原规则添加时间后缀。启动时打印实际目录。复制的其他 runs 仅为历史资料，不自动加载 checkpoint。`designs/best_<env-id>.json` 保存各环境遇到的历史最高最终评分膜系及 R/T/A；其中 reward 是最终物理评分，training_return 是该回合各步评分之和。配置实验名与源项目一致，但两个项目根目录不同，新的输出互不覆盖。

重复启动训练时，项目会在生成 `result_dir` 前选定新的实验目录：若配置中的目录已存在，添加 `_年月日_时分秒` 后缀，同秒重名时继续加数字。日志、checkpoint、配置和 `designs` 均使用同一个新目录；启动时会打印实验目录与最佳膜系目录。JSON 中的 `training.exp_name` 保持原值，`check` 不创建新实验目录。旧版自动改名的运行可能把 `designs` 写入无后缀的原目录，应查看当次 `total_config.py` 中的 `env.result_dir` 确认历史位置；此修复不会自动移动或混合历史设计。`Measure-Command` 仅负责执行并计时，不决定这些目录。

## PPO 更新的数值保护

旧运行日志在约 1900 次更新时出现约 `6e26` 的策略损失以及明显的 KL 偏移，随后分类 logits 变为 NaN。第一轮数值保护后仍曾出现约 `4e26` 的策略损失。后续复现定位到离散动作的形状问题：DI-engine hybrid 数据路径可能把 action_type 拼成 `[B,1]`，原生 Categorical.log_prob 所需形状为 `[B]`，错误形状导致 `[B,B]` 广播并交叉比较不对应的样本。只检查正确配对动作的 KL 无法覆盖这一错误；现在同时在 transition 与学习入口将单样本 action_type 规范为标量，确保检查和原生损失使用相同的 `[B]` 动作。回归测试复现了“小 KL，但错误形状导致损失溢出”，并验证修复后的原生损失形状与有限性。

无更新的 `check` 无法覆盖反向传播故障。`film_rl/policy.py` 还保留如下数值保护：

- 固定每批采样时计算的 GAE 和价值目标（`recompute_adv=False`），避免多次复用同一批数据时不断改变拟合目标。
- 每次原生小批量更新前，用 float64 计算离散、连续两个分布各自的精确 `KL(old || new)`。任一平均 KL 超过 `training.target_kl=0.02`，停止复用当前批数据并重新采样。
- 在原生 `exp(logp_new-logp_old)` 前检查采样动作的对数概率比。绝对值超过 `training.max_abs_log_ratio=10` 时同样停止复用；不把 NaN 改成零或伪造 reward。
- 独立高斯分支的 log(sigma) 在使用前限制到 `[-5, 2]`；采样、评估和学习使用一致的分布。
- 对训练数据、网络输出和梯度做有限值检查；非有限梯度在优化器更新前报错。若新采样数据尚未更新就超过 KL 限制，报错提示旧样本或模型不一致，而不是循环跳过。

`epoch_per_collect=10` 现在表示每批数据**最多**重复 10 遍。若未触发提前停止，1024 条样本、128 的 batch_size 仍对应 80 次更新；触发保护时会更少。`max_train_iter=10000` 继续按实际小批量更新次数计数，因此不再固定对应 125 个采集周期。`rollout reuse stopped` 是停止复用当前批数据的提示，不是整个训练结束。

相关思路参见 [OpenAI Spinning Up 的 PPO KL 提前停止说明](https://spinningup.openai.com/en/latest/algorithms/ppo.html)。这些保护和有限回归测试不等于对长时间训练收敛性的保证。

## 查看训练曲线

训练会自动将 TensorBoard 事件文件保存到各实验的 `log/serial/`。在 yolov11 的另一个终端运行：

```powershell
python -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6006
```

浏览器打开 `http://127.0.0.1:6006`，在 Scalars 中选择当前 `duv193_hppo_shapesafe` 实验。优先查看 `evaluator_iter/reward_mean`、`learner_iter/total_loss_avg`、`learner_iter/policy_loss_avg`、`learner_iter/value_loss_avg`。`*_iter` 横轴为参数更新次数，`*_step` 横轴为采集交互步数。原生日志中的 approximate KL 与数值保护计算的精确 KL 不是同一个统计量。

导出 PNG 和原始标量 CSV：

```powershell
python plot_training.py --run runs/duv193_hppo_shapesafe --output outputs/training_curves_shapesafe.png --reward-as-transmission
```

目标谱全为 1 时，评估仍记录最终评分，可以用 `--reward-as-transmission` 将评估曲线换算成最终透射率百分数；累计 training_return 不可这样换算。使用 `--run` 指定本版新训练实际打印的目录，避免读取复制的旧实验。PNG/CSV 是导出时的快照；TensorBoard 可查看新增的累计奖励日志。

## 调整入射角和目标

把配置中的角度改为 0–30°、共 7 个等间隔采样点：

```json
"angles_deg": {"start": 0.0, "stop": 30.0, "num": 7}
```

也可用 `"angles_deg": [0, 10, 20, 30]`。这会评估并平均所有角度，而不是每回合随机抽取一个角度。`polarization` 可设为 `s`、`p` 或 `unpolarized`。

`wavelengths_nm` 支持相同的列表/范围语法，当前 `[193.0]` 是单波长。`target_transmission` 支持标量、与波长数相同的一维数组，或 `[角度, 波长]` 二维数组，值域为 0–1。多角度单波长的逐角度目标需写成 `[[0.99], [0.98], ...]`。改变物理配置或材料顺序后，应重新训练相应策略。

## 文件分工

```text
configs/duv193.json      光学条件、材料路径、训练参数
film_rl/config.py       参数和网格验证
film_rl/optics.py       材料读取、插值、TMM、绝对光学评分
film_rl/env.py          DI-engine 环境、混合动作、膜系状态和导出
film_rl/hppo.py         原生 VAC 的 EOS 掩码及 HPPO 配置
film_rl/policy.py       混合动作形状适配及原生 PPO 更新的 KL/概率比/梯度检查
film_rl/__main__.py     check / evaluate / train 命令
examples/design.json   手动评估示例，按入射侧到基底排序
tests/test_project.py  行为和物理验证
tests/test_policy_stability.py  临时模型的更新和数值稳定性回归
```

接口依据：[DI-engine PPO 源码](https://github.com/opendilab/DI-engine/blob/main/ding/policy/ppo.py)、[VAC 源码](https://github.com/opendilab/DI-engine/blob/main/ding/model/template/vac.py)、[TMM-fast 官方仓库](https://github.com/MLResearchAtOSRAM/tmm_fast)。实现优先匹配本机 `yolov11` 内已安装的版本。
