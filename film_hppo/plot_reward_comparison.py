"""Plot linear and mixed rewards for target transmission = 1; no training."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def mixed_reward(transmission, alpha, epsilon):
    error = 1.0 - np.asarray(transmission, dtype=float)
    log_reward = -np.log10(np.maximum(error, epsilon)) / -np.log10(epsilon)
    return (1.0 - alpha) * transmission + alpha * log_reward


def main():
    project = Path(__file__).resolve().parent
    cfg = json.loads((project / "configs/duv193.json").read_text(encoding="utf-8-sig"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=cfg.get("reward_alpha", 0.0))
    parser.add_argument("--epsilon", type=float, default=cfg.get("reward_epsilon", 1e-6))
    parser.add_argument("--output", type=Path, default=project / "outputs/reward_curve_comparison.png")
    args = parser.parse_args()
    if not np.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error("alpha must be finite and within [0, 1]")
    if not np.isfinite(args.epsilon) or not 0 < args.epsilon < 1:
        parser.error("epsilon must be finite and within (0, 1)")
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "path"})
    transmission = np.unique(np.concatenate((np.linspace(0., .99, 1200),
                                             1. - np.logspace(-2, -9, 1800), [1.])))
    new = mixed_reward(transmission, args.alpha, args.epsilon)
    assert np.isfinite(new).all() and np.all(np.diff(new) >= -1e-12)
    assert np.isclose(new[0], 0) and np.isclose(new[-1], 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    colors = ("#2467A3", "#D16A16")
    for ax in axes:
        ax.plot(100*transmission, transmission, color=colors[0], linewidth=2.3, label="原奖励：r = T")
        ax.plot(100*transmission, new, color=colors[1], linewidth=2.3, label="新奖励：线性项与对数项混合")
        ax.set_xlabel("平均透射率 T（%）")
        ax.set_ylabel("Reward")
        ax.grid(alpha=.22)
    axes[0].set(xlim=(0, 100), ylim=(0, 1.035), title="完整范围")
    axes[0].legend(loc="upper left", frameon=False, fontsize=10)
    axes[1].set(xlim=(99., 99.99), ylim=(.65, 1.015), title="接近目标时的变化（局部放大）")
    values = np.array([.998, .999])
    new_values = mixed_reward(values, args.alpha, args.epsilon)
    for ax in axes:
        ax.scatter(100*values, values, color=colors[0], s=25, zorder=4)
        ax.scatter(100*values, new_values, color=colors[1], s=25, zorder=4)
    for value, reward, offset in zip(values, new_values, [(-28, -26), (-18, 28)]):
        axes[1].annotate(f"{100*value:.1f}% → {reward:.6f}", (100*value, reward),
                         xytext=offset, textcoords="offset points", fontsize=9,
                         ha="right" if offset[0] < 0 else "left",
                         arrowprops={"arrowstyle": "-", "color": colors[1]}, color=colors[1])
    old_gain = float(values[1]-values[0])
    new_gain = float(new_values[1]-new_values[0])
    axes[1].text(.04, .72,
                 f"99.8% → 99.9%\n原奖励增量：{old_gain:.6f}\n新奖励增量：{new_gain:.6f}\n增量比：{new_gain/old_gain:.2f} 倍",
                 transform=axes[1].transAxes, fontsize=10, va="top")
    fig.suptitle(f"奖励随透射率变化：α = {args.alpha:g}，ε = {args.epsilon:g}", fontsize=16, y=.97)
    fig.text(.5, .035,
             "目标透射率为 100%；此时 MAE = 1 − T。先平均误差，再计算对数项。\n"
             "图示为函数映射；奖励差异放大不等同于 PPO 更新幅度同比增加。",
             ha="center", fontsize=10, color="#444444", linespacing=1.6)
    fig.subplots_adjust(left=.07, right=.98, top=.83, bottom=.22, wspace=.27)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(json.dumps({"alpha": args.alpha, "epsilon": args.epsilon,
                      "transmission": values.tolist(), "old_reward": values.tolist(),
                      "new_reward": new_values.tolist(), "old_gain": old_gain,
                      "new_gain": new_gain, "gain_ratio": new_gain/old_gain,
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
