# 193 nm DUV 膜层设计：逐步 TMM 与增量式混合奖励

本项目逐层构建膜系，基于 **DI-engine 0.5.3 的 `PPOPolicy` 与 hybrid `VAC`（源码中标注 HPPO）** 联合输出材料和厚度参数，每个有效 step 的当前完整膜系由 **TMM-fast 0.3.0** 评估，PPO 学习相邻两步的混合评分差值。`FilmPPOPolicy` 继承原生策略并为小批量更新添加数值保护，仍调用原生离散/连续 PPO 损失及优化器。本项目由 film_hppo 复制，程序、配置、文档和测试均独立保存。主项目历史 runs/outputs 未复制；复制来源和哈希见 outputs/source_copy_manifest.json。check 和本副本测试不执行训练。

## 当前设计条件

| 参数 | 配置 |
| --- | --- |
| 波长 | 193 nm |
| 入射介质 | 空气，n=1 |
| 基底 | 材料目录中的 `CaF2 (110-800nm, RIT).sub` |
| 入射角 | 当前 0–45°，10 个等间隔角度，可改为列表或范围 |
| 偏振 | 默认非偏振，功率透射率取 `(T_s + T_p)/2` |
| 目标透射率 | 默认 1.0，即 100%，可修改 |
| 膜层数量 | 当前至少 1 层、最多 20 层 |
| 单层厚度 | 当前配置 5–100 nm，连续值 |
| 总膜厚上限 | 400 nm，所有膜层物理厚度之和 |
| 相邻材料 | 默认禁止连续两层使用同一种材料 |
| 镀膜材料 | MgF2 / LaF3 / AlF3 / GdF3，193 nm 的固定 n；k=0 为无吸收假设 |

配置入口：`configs/duv193.json`。默认 `material_source="fixed_193nm"`，使用四种材料的 193 nm 固定光学常数。CaF2 基底仍从 `E:/AI+film design/material_example` 中的 `.sub` 读取和插值。原始材料文件不修改。

## 材料数据模式

默认 `fixed_materials_193nm` 的材料顺序与数值为：

| 材料 | n | k |
| --- | --- | --- |
| MgF2 | 1.440 | 0.0 |
| LaF3 | 1.640 | 0.0 |
| AlF3 | 1.394 | 0.0 |
| GdF3 | 1.676 | 0.0 |

数值原样继承复制时主项目的当前配置；MgF2=1.440、LaF3=1.640 为先前按论文 Figure 1(b) 估读的近似值，AlF3/GdF3 使用先前选择的文献数据。此配置中的 **k=0 是明确的无吸收建模假设，并非实测 k**，可自行在各材料条目中修改。结果 JSON 保存实际使用的材料名称、n/k 和来源说明。固定值仅允许 `wavelengths_nm=[193.0]`；改为其他波长会报错，不会把 193 nm 的数据外推。

原来的 `.lm` 波长插值模式完整保留：设置 `material_source="lm"` 后，使用 `material_files` 列表中的文件及其真实 n/k 表进行线性插值。未声明 `material_source` 的历史配置仍按 lm 模式解释。默认配置保留原五材料的 `material_files` 列表供切换；也可以替换成其他数量的 `.lm` 材料，动作与状态维度会随列表长度自动调整。

修改前的五材料配置保存在 `configs/duv193_lm.json`，可独立运行：

```powershell
python -m film_rl check --config configs/duv193_lm.json --output outputs/smoke_check_lm.json
python -m film_rl evaluate --config configs/duv193_lm.json --design examples/design_lm.json --output outputs/example_spectrum_lm.json
```

材料数为 M 时，离散动作数为 M+1（末尾为 EOS），连续参数为 M 个，状态长度为 `max_layers*(M+1)+1`。四材料配置与原五材料配置的网络维度及材料编号不同，需要分别重新训练，不能复用对方的 checkpoint；重算旧设计时请使用它原来的配置。

## 动作、状态和奖励

材料顺序显式固定，新增 `.sub` 或其他文件不会改变动作编号：

| action_type | 含义 |
| --- | --- |
| 0 | MgF2 |
| 1 | LaF3 |
| 2 | AlF3 |
| 3 | GdF3 |
| 4 | EOS，结束当前膜系 |

DI-engine 接收如下混合动作：

```python
action = {
    "action_type": 1,
    "action_args": [u_MgF2, u_LaF3, u_AlF3, u_GdF3],
}
```

当前四个连续参数为网络采样的高斯潜变量。只使用本次所选材料对应的参数（示例选择 LaF3）：

```text
d_max_effective = min(thickness_max_nm, remaining_budget)
d_nm = thickness_min_nm + (d_max_effective - thickness_min_nm) * (tanh(u_selected) + 1) / 2
```

有限潜变量一般映射到有效厚度范围的开区间，浮点饱和时可取边界；剩余预算恰好等于最小厚度时允许固定厚度。环境向 PPO 保留原始潜变量用于概率计算，不把物理厚度替换回动作；EOS 忽略连续参数。该实现直接沿用 DI-engine hybrid PPO 的连续向量损失，未选中的参数和 EOS 时的参数仍会进入原生连续分支的概率/熵计算；没有额外实现“按所选材料裁剪连续损失”的算法变体。

状态为固定长度 `max_layers × (M+1) + 1`（当前 `20 × 5 + 1 = 101`）的 float32 向量：每个位置包含 M 维材料 one-hot 和 1 维归一化厚度，末尾是 `当前层数 / 最大层数`。空位置全零。膜层顺序和已有厚度均保留，固定目标、入射条件和材料谱由配置提供。旧五材料配置仍为 121 维。

`forbid_adjacent_same_material=true` 禁止相邻重复材料：例如 MgF2 → MgF2 不合法，MgF2 → LaF3 → MgF2 合法。`FilmVAC` 根据状态中最后一层的 one-hot，将该材料的离散 logit 屏蔽；采样、评估、学习和 KL 检查使用同一掩码。第一层可以选任意材料，EOS 仍按最小层数规则处理。环境拒绝手动传入的重复材料动作且不改变膜系，`random_action()` 也只采样合法动作。独立 `evaluate` 同样检查该限制；如需重算历史含相邻重复层的设计，请使用单独配置设置 `forbid_adjacent_same_material=false`，不必修改原始设计文件。网络输入维度、厚度参数和奖励公式保持不变；此约束应在新启动的训练中使用。

总膜厚通过 `max_total_thickness_nm=400.0` 设置；设为 `null` 或省略此字段可禁用。每次添加膜层时，将厚度潜变量映射到单层范围与剩余预算共同允许的区间；若 `min_layers>1`，还会为尚需添加的最小层数预留厚度。已有材料/厚度信息可以确定剩余预算。结果 JSON 的 `total_thickness_nm` 记录总厚度，独立 evaluate 也会拒绝超出上限的膜系。历史超厚设计如需重算，应使用禁用总厚度限制的独立配置。

每个有效 step 都评估当前完整膜系，包括 EOS。首次 reset 时计算一次裸基底混合评分 g0，之后各回合复用这个固定基准；裸基底不是一次动作，不产生奖励。回合 reset 时重新以 g0 作为上一评分。

```text
MAE = mean_over_angles_and_wavelengths(abs(T - T_target))
score = 1 - MAE
log_reward = -log10(max(MAE, reward_epsilon)) / -log10(reward_epsilon)
mixed_score = (1 - reward_alpha) * score + reward_alpha * log_reward
step_reward = mixed_score_current - mixed_score_previous
```

先对误差取平均、计算非线性混合评分，再求相邻膜系的评分差；不是对可能为负的增量直接取对数。目标全为 1 时，评分提高对应平均透射率提高。改善产生正奖励，变差产生负奖励，相同评分为零。`reward_space=[-1,1]`。

- EOS 没有改变膜系，训练奖励严格为 0。仍执行 TMM 评估，返回完整结果。
- 达到 `max_layers`（当前 20 层）或总厚度预算终止时，最后添加的一层仍只获得这一次增量奖励；不额外叠加绝对终止奖励。
- `discount_factor=1.0` 保持不变，因此整回合训练回报约等于 `g(final)-g(bare)`（float32 只有微小舍入误差）。这种反馈不保证透射率逐层上升，也不保证找到更多层；回报仍反映最终膜系的表现。
- 相邻材料限制、EOS 最小层数掩码、总膜厚约束、四材料/文件插值模式、状态、VAC 网络、PPO 与数值保护均继承主项目。

在 `configs/duv193.json` 顶层调整 `reward_alpha`（当前 0.5）与 `reward_epsilon`（当前 1e-6）。alpha=0 时退化为原始 score 的增量；alpha=1 时使用纯对数评分的增量。epsilon 截断附近纯对数项可能饱和，相同混合评分的奖励为零。

例如平均透射率由 0.998 提高到 0.999，alpha=0.5、epsilon=1e-6 时：

```text
g(0.998) = 0.7239141670
g(0.999) = 0.7495000000
step_reward = +0.0255858330
```

反向变化返回 -0.0255858330；EOS 返回 0。

保存字段需要区分：

| 字段 | 含义 |
| --- | --- |
| `score` | 原始 1-MAE；目标为 1 时等于平均透射率 |
| `mean_transmission` | 真实平均功率透射率 |
| `reward` / `mixed_score` | 当前膜系的绝对混合评分 g，保留 reward 名称以兼容独立 evaluate 和设计读取 |
| `step_reward` | 实际给 PPO 的单步奖励，可以为负 |
| `previous_mixed_score` | 本步之前的混合评分 |
| `baseline_mixed_score` | 裸基底的混合评分 |
| `episode_training_return` | 本回合实际返回的 float32 step_reward 之和 |

`info.eval_episode_return` 继续使用 `score`，因此评估均值、最佳 checkpoint、`stop_value`、最佳设计均按物理评分比较。目标谱全为 1 时，可继续使用 `plot_training.py --reward-as-transmission`，不会把可能为负的累计训练回报误当透射率。

独立 `evaluate` 不产生动作序列，其 `reward` 是完整膜系的绝对混合评分；逐步增量只由环境 `step()` 产生。

## 光学计算约定

- 膜层按**入射侧 → 所列膜层 → CaF2 基底**排列，和沉积顺序可能相反。
- `material_source="fixed_193nm"` 时镀膜层使用配置中的固定 n/k，仅用于 193 nm；当前 k=0 为无吸收假设。`material_source="lm"` 时镀膜层读取 `.lm`，按 JSON 中的 `wavelength/n/k` 表线性插值，不外推、不把文件中的 k 强制设零。基底始终从 `.sub` 插值。复折射率使用 `N=n+i k`。
- 波长与膜厚在进入 TMM-fast 前统一由 nm 换算为 m，入射角由度换为弧度。
- 入射介质及基底厚度设为无穷大，基底使用 `.sub` 中的光学常数。本项目的 T 是**进入半无限 CaF2 基底的功率比例**，不是穿过整片有限厚度透镜及其背面的总透射率。R/T/A 均以入射功率归一化；此处 A 对应涂层区域的损耗，未传播到基底深处或计算背面。
- 数值异常直接报错，不静默返回高奖励。验证包括裸界面 Fresnel 解析结果、无损四分之一波膜的解析反射率及能量守恒。

## 在已配置的 yolov11 环境中使用

无需重新安装已配置的依赖。在 PowerShell 中：

```powershell
conda activate yolov11
Set-Location 'E:\lithography_machine_lens\film_hppo_step_reward'

# 只验证配置、HPPO 前向、环境管理器、采样/GAE 和 TMM，不更新参数
python -m film_rl check --output outputs/smoke_check.json

# 行为、物理与边界测试，不更新网络
python -m unittest discover -s tests -p test_step_rewards.py -v

# 全部测试：不执行训练或优化器更新
python -m unittest discover -s tests -v

# 对一个指定膜系做光学评估
python -m film_rl evaluate --design examples/design.json --output outputs/example_spectrum.json
```

`check` 中会构建策略并初始化 DI-engine 学习侧的共享状态，以便检查采样/GAE 接口，但不会调用学习前向、反向传播或优化器更新，并检查模型参数没有变化。其随机膜系结果仅用于接口验证，不是优化设计。运行无更新的采样检查时，DI-engine 可能显示 `train_iter=0` 并在 `outputs/check_runtime` 写入日志。

需要训练时，单独运行下列命令：

```powershell
python -m film_rl train --config configs/duv193.json
```

训练调用原生 `serial_pipeline_onpolicy`，采用同步 `BaseEnvManager`，避免 Windows 子进程启动问题。默认 CPU，可在配置中分别修改 `training.cuda`（网络）和 `tmm_device`（光学计算）。单个 193 nm 点的小膜系通常无需把 TMM 切换到 GPU。

训练输出在 `training.exp_name` 指定的项目内目录，包含 DI-engine 日志、配置和 checkpoint；实际路径以启动时打印为准。历史结果保留，不自动加载其 checkpoint。`designs/best_<env-id>.json` 保存各环境遇到的历史最高 `score` 膜系及 R/T/A；在这些文件中取 `score` 最大者就是所有已保存候选的最优值（旧文件没有 `score` 时使用旧 `reward`）。`training.exp_name` 使用项目内相对路径，以兼容 DI-engine 的 Windows 日志路径。后续可直接把保存的膜系 JSON 传给 `evaluate --design` 重算。

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

浏览器打开 `http://127.0.0.1:6006`，在 Scalars 中选择当前 `seed0_test1` 实验。优先查看 `evaluator_iter/reward_mean`、`learner_iter/total_loss_avg`、`learner_iter/policy_loss_avg`、`learner_iter/value_loss_avg`。`*_iter` 横轴为参数更新次数，`*_step` 横轴为采集交互步数。原生日志中的 approximate KL 与数值保护计算的精确 KL 不是同一个统计量。

导出 PNG 和原始标量 CSV：

```powershell
python plot_training.py --run runs/seed0_test1 --output outputs/training_curves_shapesafe.png --reward-as-transmission
```

当前混合奖励仍将原始 `score` 提供给 evaluator，因此目标谱全为 1 时，继续使用 `--reward-as-transmission` 可以正确绘制透射率百分数；绘图没有把混合奖励当成透射率。目标谱不是全 1 时应省略该参数。`evaluator_iter/training_return_mean` 是独立的增量累计训练回报指标；`evaluator_iter/final_mixed_score_mean` 是最终混合评分。PNG/CSV 是导出时的快照，重新运行可刷新；TensorBoard 用于查看持续写入的日志。PPO 的策略损失可以为负且不要求单调下降，应结合评估结果判断设计效果。

## 调整入射角和目标

把配置中的角度改为 0–30°、共 7 个等间隔采样点：

```json
"angles_deg": {"start": 0.0, "stop": 30.0, "num": 7}
```

也可用 `"angles_deg": [0, 10, 20, 30]`。这会评估并平均所有角度，而不是每回合随机抽取一个角度。`polarization` 可设为 `s`、`p` 或 `unpolarized`。

`wavelengths_nm` 支持相同的列表/范围语法，当前 `[193.0]` 是单波长。`target_transmission` 支持标量、与波长数相同的一维数组，或 `[角度, 波长]` 二维数组，值域为 0–1。多角度单波长的逐角度目标需写成 `[[0.99], [0.98], ...]`。改变物理配置或材料顺序后，应重新训练相应策略。

## 批量 seed 实验的命名与目录

在项目目录、yolov11 环境中运行：

```powershell
python run_seed_batch.py --workers 3
```

默认运行 seed0–seed9，共 10 次，最多同时训练 3 个独立进程。脚本从当前配置自动生成实验组名；当前配置为 `duv193_angle0-45_step_delta_mix_a0p5`。`alpha=0` 使用 `step_delta_linear`，`alpha=1` 使用 `step_delta_log`。名称描述波长、角度范围和奖励形式，完整条件仍以该批次的配置快照为准。

每次调用生成新的北京时间批次编号，例如 `20261018_150000`。同一秒发生重名时增加 `_01`、`_02`，不会移动、归档或覆盖旧实验。目录结构：

```text
runs/<实验组>/<批次时间>/
  seed0/                 DI-engine 日志、checkpoint、designs、result.pkl
  ...seed9/

outputs/<实验组>/<同一批次时间>/
  base_config.json       解析材料路径后的基础配置快照
  manifest.json          seed、训练目录、配置及日志路径、文件哈希
  configs/seed0.json     各 seed 的独立配置
  logs/seed0.train.log   训练进程的标准输出和错误输出
  logs/seed0.plot.log    绘图日志
  status/seed0.json      当前状态、PID、耗时、结果
  seed0.png              该 seed 完成后自动导出的曲线
  seed0.csv              原始曲线数据
  ...seed9.png / .csv
  results.csv            全部 seed 的结果对照表
  summary.json           详细进度及结果
```

原始 `configs/duv193.json` 不修改；各 seed 的配置只改变 `training.seed` 和 `training.exp_name`。训练与绘图都读取 manifest 中的实际路径，不再拼接旧的 `seedN_test0` 目录。完整批处理会自动刷新汇总（训练期间约每 30 秒，完成一个 seed 后也刷新），单个 seed 失败会记录错误并继续其他 seed。

自定义实验组、配置或种子列表：

```powershell
python run_seed_batch.py --group duv193_angle0-45_step_delta_mix_a0p5 --workers 3
python run_seed_batch.py --config configs/duv193.json --group thickness_comparison --seeds 0 2 5 --workers 1
```

`--group` 仅修改标签，不改变 alpha 等实验参数；参数仍在配置 JSON 中调整。组名使用英文字母、数字、下划线、短横线或点，不包含目录分隔符。波段采样点数、材料和厚度限制等条件变化时，可用组名标出当前比较变量。

只生成配置、预览路径，不启动训练：

```powershell
python run_seed_batch.py --prepare-only
```

随后要运行这一份准备好的批次，使用它打印的实际输出目录：

```powershell
python run_seed_batch.py --batch-dir "outputs/实验组/批次时间" --workers 3
```

`--batch-dir` 用于执行尚未开始的配置快照，不是修改输出目录，也不是 checkpoint 续训；不能与 `--config`、`--group`、`--seeds` 同用。代码、材料或准备好的 seed 配置被修改时会拒绝启动，需要重新准备批次。

查看汇总或实时查看一个 seed 的终端日志：

```powershell
python summarize_seed_batch.py "outputs/实验组/批次时间"
Get-Content "outputs/实验组/批次时间/logs/seed0.train.log" -Tail 40 -Wait
```

汇总表分别记录最佳设计的真实透射率、原始评分、最终混合评分、累计训练回报、最后一步奖励、最佳/最后评估均值、膜层数、总厚度、训练耗时和目录。旧批次的 manifest/status 布局仍可由汇总脚本读取；历史结果留在源项目；本副本的 runs/outputs 独立保存新实验。

## 文件分工

```text
configs/duv193.json      光学条件、材料路径、训练参数
film_rl/config.py       参数和网格验证
film_rl/optics.py       材料读取、插值、TMM、绝对混合评分
film_rl/env.py          DI-engine 环境、混合动作、膜系状态和导出
film_rl/hppo.py         原生 VAC 的 EOS 掩码及 HPPO 配置
film_rl/policy.py       混合动作形状适配及原生 PPO 更新的 KL/概率比/梯度检查
film_rl/__main__.py     check / evaluate / train 命令
run_seed_batch.py       按实验组/时间批次/seed 批量训练、绘图和汇总
summarize_seed_batch.py 新旧批次的进度及结果 CSV 导出
examples/design.json   手动评估示例，按入射侧到基底排序
tests/test_step_rewards.py  混合评分增量、终止、回报与采样日志验证
tests/test_material_modes.py  材料模式、TMM、网络维度与批量准备验证
```

接口依据：[DI-engine PPO 源码](https://github.com/opendilab/DI-engine/blob/main/ding/policy/ppo.py)、[VAC 源码](https://github.com/opendilab/DI-engine/blob/main/ding/model/template/vac.py)、[TMM-fast 官方仓库](https://github.com/MLResearchAtOSRAM/tmm_fast)。实现优先匹配本机 `yolov11` 内已安装的版本。
