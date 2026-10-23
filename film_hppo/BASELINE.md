# 灯泡滤光片评价：反射率曲线与增强因子

入口为 `evaluate_baseline.py`，计算核心为 `film_rl/enhancement.py`。独立评价已有设计，不改变奖励、不启动训练或厚度优化。

## 只输入设计地址计算增强因子

简洁入口为 `calculate_enhancement.py`，默认只输出到终端，不生成图表或结果文件：

```powershell
python calculate_enhancement.py --design "E:\lithography_machine_lens\film_hppo\runs\incandescent_task\seed0_limit\designs\best_66ded14df42548adba28e9bdbe6a79fa.json"

# 如需保存计算结果，增加 --output；已有文件不会覆盖
python calculate_enhancement.py --design "你的设计.json" --output "outputs\enhancement_result.json"

# 设计单独复制出来、找不到训练快照时，显式指定原配置
python calculate_enhancement.py --design "你的设计.json" --config "configs\incandescent_task.json"
```

程序沿设计的父目录查找最近的 `config_snapshot.json` 或 `total_config.py`，自动沿用原训练/优化的材料顺序及光谱范围。只解析训练快照的字面量配置，不执行其中的Python代码；有材料校验信息时核对材料文件哈希。手动指定 `--config` 优先。没有快照则要求指定原配置，不会偷偷使用现在修改过的配置。

默认计算 f=1 和0.95，100 W、两面各20 mm²，400–780 nm人眼加权，光谱窗口外R=0；CPU重算2 nm波长间隔及64点半球角积分。这些沿用下文的物理模型约定。参数可用 `--view-factors 1 0.95`、`--wavelength-step-nm 1`、`--angle-order 96`、`--power-w`、`--area-mm2`、`--radiating-sides`、`--outside-reflectance` 调整。结果JSON记录假设、原文件哈希、材料、结构、温度和功率残差；其中480–700 nm半球平均R/T/A只是额外诊断，不是训练中均匀角度平均的指标。

## 生成完整图表报告

```powershell
# 评价一套设计；把路径替换为实际 design 路径
python evaluate_baseline.py --config configs/incandescent_task.json --design runs/你的实验/designs/best_xxx.json --output outputs/baseline_run01

# 多套设计在相同条件下比较，并重新计算论文 Table 3 的42层结构
python evaluate_baseline.py --config configs/incandescent_task.json --design outputs/design_A.json outputs/design_B.json --labels A B --include-paper-reference --output outputs/baseline_compare01
```

输出目录必须是新目录，已有实验不会覆盖。命令可从 film_hppo 根目录运行；材料 ID 和名称必须与所给 config 一致。论文 SiN 显式映射为当前 Si3N4 数据；参考结构结果标记为 `Paper42_current_nk`，不是作者原始光谱。

输出：

- `reflection.png`：上图为0–89°每1°均匀平均R，下图为半球加权R；设计曲线和论文目标曲线同时显示。
- `baseline.json`：各设计在 f=1、0.95 下的 chi、温度、功率残差、材料来源、输入文件校验以及模型假设。
- `comparison.csv`、`README.md`：精简比较表。
- `spectra_XX.csv`：波长、均匀平均和半球平均 R/T/A；`npz`保留角度光谱。
- `emission.png`：论文有效发射模型的等功率出射谱。

默认通过 `paper_curve_csv` 叠加此前从论文Figure 7(a)提取的曲线（仅上图），这是图像曲线估计值，不是作者原始数据，且不参与chi计算。若没有该提取文件，可在设置中将 `paper_curve_csv` 设为null。同时可用 `--include-paper-reference` 显示论文结构按当前nk重算的结果，二者标签不同。

参数集中在 `configs/incandescent_baseline.json`，通过 `--settings` 可指定另一份。修改参数后，不同设计需重新使用同一份设置比较。

## 文献模型与实现约定

[OML-PPO 附录 B](https://arxiv.org/html/2006.11940#A2) 定义：

`epsilon_eff(lambda,theta)=1-f^2 R(lambda,theta)`。

用 `2 sin(theta) cos(theta)` 进行半球积分，然后在固定输入功率下求温度，再用人眼响应加权的发射功率比计算 chi。默认可见积分400–780 nm，不是训练目标区间480–700 nm；训练权重不参与 chi。

默认假设100 W、20 mm²，f=1及0.95。黑体出射度使用 `M_lambda=pi*B_lambda`（W/m²/nm）。采用两面、每面20 mm²，裸黑体温度2576.81 K，接近文中2578 K。**两面归一化是根据文中基准温度作的明确推断，不是附录明确提供的几何条件；没有拟合增强因子。** 可用 `radiating_sides` 调整归一化约定，所有设计须保持一致。

默认保留光学 config 的波长上下限，独立以2 nm间隔重算TMM；不修改训练配置。半球积分采用64点Gauss–Legendre，令 `u=sin(theta)^2`，避免90°奇点。额外计算0–89°曲线均匀平均；计算按波长分块以控制内存。

热平衡包含全波长黑体辐射，积分总量使用 Stefan–Boltzmann 定律，再减去已知窗口内回收部分。**窗口外没有材料数据，不外推 nk。默认窗口外R=0；同时给出窗口外R=1的敏感性上界。** 原文未公开其具体波段外处理，主值及界限不能宣称严格复现16.60，窗口外界限也不是实际性能。

默认人眼响应由 [Sharpe et al. 2005](https://doi.org/10.1167/5.11.3) Eq.5 重建：

`V_energy=(1.624340*l_energy+m_energy)/2.525598`。

使用开源 Colour v0.4.6 中的 Stockman–Sharpe 2°锥体响应数据，保留其BSD许可证与校验值，并与论文Table 3数值核对；没有用CIE 1924或2008曲线冒充2005数据。响应表可离线使用。`eye_csv`可显式替换为两列CSV，表头为 `wavelength_nm,V_energy`，数值必须是线性能量响应，不能直接传入对数值或光子响应。

该模型仅通过反射描述回收，未独立求薄膜吸收、温升、再发射；吸收不能直接当作可见透射。报告另保留480–700 nm半球平均A诊断。温度是理想黑体模型温度，不代表钨丝实际可运行温度。不能据此直接判断可加工性或寿命。

## 作者仓库检查

首先检查了 [hammer-wang/oml-ppo](https://github.com/hammer-wang/oml-ppo)，固定版本 `ef63afe85e9c09cb3ccf0df01218c6264535ac9c`。检查22个Python、notebook和说明文件（notebook只读取源码单元、不执行），没有找到增强因子、普朗克谱或热平衡参考实现。目录及源码哈希记录在 `outputs/references/oml_enhancement_source/audit.json`。因此本程序依据附录B实现，不能标为作者开源计算程序。

文献报告 chi=16.60 (f=1)、10.67 (f=0.95)。只有在材料数据、插值、角度积分、光谱范围及其外推、几何和人眼响应约定一致时，数值才可严格比较。本程序优先用于同条件比较自己的设计；同时显示论文结构在当前数据下的重算结果。

验证命令：

```powershell
python -m unittest discover -s tests -p test_enhancement.py -v
```
