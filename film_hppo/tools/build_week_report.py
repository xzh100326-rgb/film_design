"""Build a concise presentation index from curated, existing experiments only."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from film_rl.config import load_config
from film_rl.optics import FilmOptics

WEEK = PROJECT/"outputs/week_20261014_20261020"
DUV_GROUPS = [("duv193__hppo__nk_v1__a00-45", "20261018_160311", "nk_v1"),
              ("duv193__hppo__nk_v2__a00-45", "20261019_143507", "nk_v2")]
ABS_RUN = PROJECT/"runs/abs400-2000__hppo__baseline/20261020_004725/seed1"
ABS_OUTPUT = PROJECT/"outputs/abs400-2000__hppo__baseline/20261020_004725"
HYBRID = PROJECT/"outputs/abs400-2000__hybrid__search/20261020_155601"
DUV_HYBRID = PROJECT/"outputs/duv193__hybrid__nk_v2__a00-45/20261020_164544"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def link(label, path):
    return f"[{label}](/{Path(path).as_posix()})"


def read_best(run):
    items = []
    for p in (run/"designs").glob("best_*.json"):
        d = json.loads(p.read_text(encoding="utf-8-sig"))
        score = d.get("score", d.get("mean_transmission", d.get("mean_absorption", -1)))
        items.append((float(score), p, d))
    return max(items, key=lambda row: row[0])


def training_rows():
    rows, highlights = [], {}
    for group, stamp, variant in DUV_GROUPS:
        base = PROJECT/"runs"/group/stamp
        output = PROJECT/"outputs"/group/stamp
        best_item = None
        for run in sorted(base.glob("seed*")):
            result = clean(pickle.loads((run/"result.pkl").read_bytes()))
            score, source, design = read_best(run)
            write_json(output/"designs"/f"{run.name}_best.json", design)
            rows.append({"task": "duv193", "method": "hppo", "variant": variant,
                         "experiment": group, "timestamp": stamp, "seed": int(run.name[4:]),
                         "metric": "mean_transmission", "best_design_score": score,
                         "last_evaluation_mean": result["eval_value"], "layer_count": design["layer_count"],
                         "optimizer_updates": result["train_iter"], "env_steps": result["env_step"],
                         "run_dir": str(run), "output_dir": str(output), "raw_run_available": True,
                         "best_design": str(output/"designs"/f"{run.name}_best.json")})
            if best_item is None or score > best_item[0]: best_item = (score, source, design)
        write_json(output/"best_design.json", best_item[2])
        write_json(output/"CURATED_ORIGIN.json", {"raw_source": str(best_item[1]),
                    "metric": "transmission", "variant": variant, "selected_for_week": True,
                    "note": "Raw original files retained in archive; navigation paths on this batch copy were updated."})
        highlights[variant] = {"score": best_item[0], "design": best_item[2], "directory": output}
    score, source, design = read_best(ABS_RUN)
    result = clean(pickle.loads((ABS_RUN/"result.pkl").read_bytes()))
    write_json(ABS_OUTPUT/"best_design.json", design)
    write_json(ABS_OUTPUT/"run_result.json", result)
    rows.append({"task": "abs400-2000", "method": "hppo", "variant": "baseline",
                 "experiment": "abs400-2000__hppo__baseline", "timestamp": "20261020_004725", "seed": 1,
                 "metric": "mean_absorption", "best_design_score": score,
                 "last_evaluation_mean": result["eval_value"], "layer_count": design["layer_count"],
                 "optimizer_updates": result["train_iter"], "env_steps": result["env_step"],
                 "run_dir": str(ABS_RUN), "output_dir": str(ABS_OUTPUT), "raw_run_available": True,
                 "best_design": str(ABS_OUTPUT/"best_design.json")})
    highlights["abs_baseline"] = {"score": score, "design": design, "directory": ABS_OUTPUT}
    batch = PROJECT/"outputs/abs400-2000__hppo__multiseed/20261020_010902"
    summary = json.loads((batch/"summary.json").read_text(encoding="utf-8-sig"))
    for entry in summary["runs"]:
        result = entry.get("result", {})
        rows.append({"task": "abs400-2000", "method": "hppo", "variant": "multiseed_reports_only",
                     "experiment": batch.parent.name, "timestamp": batch.name, "seed": entry["seed"],
                     "metric": "mean_absorption", "best_design_score": entry.get("best_saved_score"),
                     "last_evaluation_mean": result.get("eval_value"), "layer_count": entry.get("best_layer_count"),
                     "optimizer_updates": result.get("train_iter"), "env_steps": result.get("env_step"),
                     "run_dir": entry.get("run_dir"), "output_dir": str(batch), "raw_run_available": False,
                     "best_design": entry.get("best_design"),
                     "note": "Original run directories/checkpoints were already absent before organization; reports only."})
    for folder, task, variant in [(HYBRID, "abs400-2000", "structure_and_thickness_search"),
                                  (DUV_HYBRID, "duv193", "nk_v2_recomputed_two_layer")]:
        design = json.loads((folder/"best_design.json").read_text(encoding="utf-8-sig"))
        summary = json.loads((folder/"summary.json").read_text(encoding="utf-8-sig"))
        rows.append({"task": task, "method": "hybrid", "variant": variant,
                     "experiment": folder.parent.name, "timestamp": folder.name, "seed": None,
                     "metric": "mean_absorption" if task.startswith("abs") else "mean_transmission",
                     "best_design_score": design["score"], "last_evaluation_mean": None,
                     "layer_count": design["layer_count"], "optimizer_updates": None, "env_steps": None,
                     "run_dir": None, "output_dir": str(folder), "raw_run_available": True,
                     "best_design": str(folder/"best_design.json"),
                     "candidates": summary["candidates"], "seconds": summary["elapsed_seconds"]})
        highlights[task+"_hybrid"] = {"score": design["score"], "design": design, "directory": folder}
    return rows, highlights


def plots(rows, highlights):
    figures = WEEK/"figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for variant, axis in zip(("nk_v1", "nk_v2"), axes):
        data = sorted([r for r in rows if r["task"] == "duv193" and r["method"] == "hppo" and r["variant"] == variant], key=lambda r: r["seed"])
        axis.plot([r["seed"] for r in data], [100*r["best_design_score"] for r in data], "o-", label="Best saved design")
        axis.plot([r["seed"] for r in data], [100*r["last_evaluation_mean"] for r in data], "s--", label="Last evaluator mean")
        axis.set(xlabel="Seed", ylabel="Mean transmittance (%)", title=f"DUV 193 nm, 0-45 deg, {variant}")
        axis.set_xticks(range(10)); axis.grid(alpha=.25); axis.legend(fontsize=8)
    fig.savefig(figures/"duv_multiseed.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for variant, label in [("nk_v1", "Historical n(LaF3)=1.696"), ("nk_v2", "Current n(LaF3)=1.640")]:
        d = highlights[variant]["design"]
        axes[0].plot(d["angles_deg"], np.asarray(d["T"])[:, 0]*100, "o-", label=label)
    d = highlights["duv193_hybrid"]["design"]
    axes[0].plot(d["angles_deg"], np.asarray(d["T"])[:, 0]*100, "--", label="Recomputed two-layer local result")
    axes[0].set(xlabel="Angle (deg)", ylabel="Transmittance (%)", title="DUV: optical-constant versions kept separate")
    # Compare absorption spectra using exactly the current finite-glass physics.
    cfg = load_config(PROJECT/"configs/400-2000absob_cold_start.json")
    optics = FilmOptics(cfg)
    baseline = highlights["abs_baseline"]["design"]
    layers = [(x["material_id"], x["thickness_nm"]) for x in baseline["layers"]]
    recomputed = optics.evaluate(layers)
    write_json(ABS_OUTPUT/"best_design_current_physics.json", recomputed)
    hybrid = highlights["abs400-2000_hybrid"]["design"]
    hybrid_layers = [(x["material_id"], x["thickness_nm"]) for x in hybrid["layers"]]
    verified_hybrid = optics.evaluate(hybrid_layers)
    assert abs(verified_hybrid["score"]-hybrid["score"]) < 1e-12
    write_json(HYBRID/"best_design_verified_current_physics.json", verified_hybrid)
    axes[1].plot(recomputed["wavelengths_nm"], np.asarray(recomputed["A"])[0]*100,
                 label=f"HPPO design, current physics: {recomputed['score']*100:.3f}%")
    axes[1].plot(hybrid["wavelengths_nm"], np.asarray(hybrid["A"])[0]*100,
                 label=f"Hybrid search: {hybrid['score']*100:.3f}%")
    axes[1].set(xlabel="Wavelength (nm)", ylabel="Absorptance (%)", title="Absorption: current glass substrate for both")
    for axis in axes:
        axis.grid(alpha=.25); axis.legend(fontsize=8)
    fig.savefig(figures/"spectra_comparison.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    records = [json.loads(line) for line in (HYBRID/"search.jsonl").read_text().splitlines()]
    candidates = [r for r in records if r["event"] == "candidate"]
    initial = next(r["score"] for r in records if r["event"] == "initial")
    axes[0].plot([0]+list(range(1, len(candidates)+1)), [100*initial]+[100*r["best_score"] for r in candidates])
    axes[0].axhline(95, color="gray", linestyle="--", label="95% mean absorption")
    axes[0].set(xlabel="Candidate evaluated", ylabel="Best mean absorptance (%)", title="Hybrid-search progress")
    axes[0].legend(fontsize=8)
    layers = hybrid["layers"]
    colors = dict(zip(["MgF2", "TiO2", "Si", "Ge", "Cr"], plt.cm.tab10.colors))
    axes[1].bar(range(1, len(layers)+1), [r["thickness_nm"] for r in layers], color=[colors[r["material"]] for r in layers])
    axes[1].set_xticks(range(1, len(layers)+1), [r["material"] for r in layers], rotation=60)
    axes[1].set(ylabel="Thickness (nm)", title=f"Hybrid design: {len(layers)} layers, {hybrid['total_thickness_nm']:.1f} nm")
    for axis in axes: axis.grid(alpha=.2)
    fig.savefig(figures/"absorption_search_progress.png", dpi=180); plt.close(fig)
    return recomputed, verified_hybrid


def documents(rows, highlights, recomputed, verified_hybrid):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (WEEK/"experiments.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    write_json(WEEK/"experiments.json", rows)
    index = ["# 最近一周实验进展（2026-10-14—2026-10-20）", "",
             "本页是汇报入口。所有数值为仿真结果；平均光谱指标与单次评估平均、最好单设计分别记录。", "",
             "## 建议展示的四个结果", "", "| 实验 | 最好单设计 | 说明 |", "|---|---:|---|"]
    index += [f"| DUV，固定常数 v1 | {highlights['nk_v1']['score']*100:.4f}% 透射率 | MgF2=1.420、LaF3=1.696；0–45°，10 seeds |",
              f"| DUV，固定常数 v2 | {highlights['nk_v2']['score']*100:.4f}% 透射率 | MgF2=1.440、LaF3=1.640；0–45°，10 seeds |",
              f"| 400–2000 nm，HPPO 基线 | {highlights['abs_baseline']['score']*100:.4f}% 吸收率 | seed1；原生训练结果，原基底 n=1.5 半无限 |",
              f"| 400–2000 nm，联合搜索 | {verified_hybrid['score']*100:.4f}% 吸收率 | 结构变化＋厚度梯度优化；14 层，非纯 HPPO 搜索结果 |", "",
              "## 图表", "", link("DUV 多 seed：最好设计与最终评估均值", WEEK/"figures/duv_multiseed.png"), "",
              link("DUV 角度曲线及吸收光谱对比", WEEK/"figures/spectra_comparison.png"), "",
              link("吸收联合搜索进展及最终层厚", WEEK/"figures/absorption_search_progress.png"), "",
              link("逐实验 CSV", WEEK/"experiments.csv"), "", "## 快速打开结果", ""]
    for label, key in [("DUV v1 最佳设计", "nk_v1"), ("DUV v2 最佳设计", "nk_v2"),
                       ("吸收 HPPO 基线设计", "abs_baseline"), ("吸收联合搜索最佳设计", "abs400-2000_hybrid")]:
        index += ["- "+link(label, highlights[key]["directory"]/"best_design.json")]
    index += ["", "## 汇报时需要区分", "",
              "- DUV 的两个版本具有不同光学常数，不能把数值变化全部归因于算法。当前 v2 已有五层结构达到约 99.8231%；之前 improve 选择历史 v1 的两层结构并按 v2 重算，仅为 99.6681%，不代表 v2 的最优值。自动预筛选候选的逻辑未在本次目录整理中改变。",
              f"- 吸收基线按当前 n=1.45、500 nm 有限玻璃条件复算为 {recomputed['score']*100:.4f}%；光谱对比图已统一采用当前基底。材料数据仍为图像近似提取，不是新的实测常数。",
              "- 98.9035% 来自外部结构搜索和厚度优化。之后 VAC 模仿该结构，是预训练，不是纯 PPO 独立发现该结构的证据。",
              "- 吸收 10-seed 批次现存曲线和汇总，原 runs/checkpoint 在整理前已经缺失；CSV 标记 raw_run_available=False，未伪造补回。",
              "- 各目录内原始日志、UUID 设计文件及源码快照保留；单独导出 best_design.json 方便展示。", "",
              "## 命名与归档", "",
              "统一规则：`任务__方法__版本/yyyymmdd_hhmmss/seedN`。runs 保存训练状态；outputs 保存曲线、设计、验证和汇总。联合搜索没有原生 PPO run，因此只在 outputs 中出现。",
              "未选入展示的试跑、中断实验、旧材料批次已完整保存在 `archive/results_before_organization_20261020/raw/`。原文件经 SHA256 验证，删除数为 0。", "",
              link("目录迁移日志", WEEK/"organization_log.json"), "", link("旧路径与新路径对照", WEEK/"path_mapping.json"), "",
              "历史配置/源码快照中的路径保留当时值，便于追溯；当前主配置的模型路径和输出名字已更新。新复制的批次 manifest/status/CSV 中的操作路径已修复。"]
    (WEEK/"README.md").write_text("\n".join(index)+"\n", encoding="utf-8")
    (PROJECT/"RESULTS.md").write_text("# 实验结果入口\n\n"+link("最近一周进展与展示图", WEEK/"README.md")+"\n\n"+link("全部实验索引", WEEK/"experiments.csv")+"\n", encoding="utf-8")
    (PROJECT/"runs/README.md").write_text(
        "# 保留的训练结果\n\n统一采用：任务__方法__版本/时间戳/seedN。\n\n"
        "- duv193__hppo__nk_v1__a00-45：历史固定光学常数，10 seeds。\n"
        "- duv193__hppo__nk_v2__a00-45：当前固定光学常数，10 seeds。\n"
        "- abs400-2000__hppo__baseline：吸收任务的代表性 HPPO 基线。\n\n"
        "其他原运行目录已完整归档，见 archive/results_before_organization_20261020/raw/runs。\n\n"
        +link("本周结果索引", WEEK/"README.md")+"\n", encoding="utf-8")
    (PROJECT/"outputs/README.md").write_text(
        "# 实验输出\n\n"+link("最近一周展示入口", WEEK/"README.md")+"\n\n"
        "任务目录与 runs 同名；联合搜索、诊断和参考资料只保存在 outputs。\n"
        "references/yang2016_digitized_nk 保存材料曲线提取来源。\n"
        "旧原始文件完整归档于 archive/results_before_organization_20261020/raw/outputs。\n", encoding="utf-8")
    archive = PROJECT/"archive/results_before_organization_20261020"
    (archive/"README.md").write_text(
        "# 整理前原始结果归档\n\nraw/runs 和 raw/outputs 保留原目录层级及文件原字节，未删减。\n"
        "inventory_before.json 记录 1200 个原文件的大小、时间和 SHA256；completed.json 记录迁移。\n"
        "config_before 保存当前配置在改名之前的版本。本次只改输出路径和模型引用，没有改变训练/光学参数。\n"
        "如需恢复旧结构，请先查看 plan.json，避免覆盖整理后产生的新结果。\n", encoding="utf-8")
    write_json(WEEK/"presentation_summary.json", {"experiments": len(rows),
               "duv_v1_best": highlights["nk_v1"]["score"], "duv_v2_best": highlights["nk_v2"]["score"],
               "abs_hppo_best_original_physics": highlights["abs_baseline"]["score"],
               "abs_hppo_best_current_physics": recomputed["score"], "abs_hybrid_best": verified_hybrid["score"]})


if __name__ == "__main__":
    torch.set_num_threads(1)
    rows, highlights = training_rows()
    recomputed, verified = plots(rows, highlights)
    documents(rows, highlights, recomputed, verified)
    print(json.dumps(json.loads((WEEK/"presentation_summary.json").read_text()), indent=2))
