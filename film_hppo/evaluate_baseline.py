"""Offline lamp-filter baseline: reflectance curves and Appendix B enhancement.

Example: python evaluate_baseline.py --config configs/incandescent_task.json
    --design path/to/best_design.json --output outputs/lamp_baseline_01
"""
import argparse
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tmm_fast import coh_tmm

from film_rl.config import PROJECT_DIR, grid, load_config
from film_rl.enhancement import enhancement, hemisphere_quadrature, planck_exitance, eye_response
from film_rl.optics import FilmOptics


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read_layers(data, optics):
    layers = []
    for row in data["layers"]:
        i = row["material_id"]
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(optics.names):
            raise ValueError("Invalid material_id in design")
        if row.get("material", optics.names[i]) != optics.names[i]:
            raise ValueError("Design material ID/name differs from config; use its original material ordering")
        thickness = row["thickness_nm"]
        if isinstance(thickness, bool) or not isinstance(thickness, (float, int)) or not np.isfinite(thickness):
            raise ValueError("Each thickness must be a finite number in nm")
        layers.append((i, float(thickness)))
    # Reuse the production validation without running a large TMM here.
    cfg = optics.cfg
    if not cfg["min_layers"] <= len(layers) <= cfg["max_layers"]:
        raise ValueError("Layer count is outside config bounds")
    if any(not cfg["thickness_min_nm"] <= d <= cfg["thickness_max_nm"] for _, d in layers):
        raise ValueError("Thickness is outside config bounds")
    if cfg.get("forbid_adjacent_same_material", True) and any(a[0] == b[0] for a,b in zip(layers,layers[1:])):
        raise ValueError("Adjacent layers use the same material")
    budget = cfg.get("max_total_thickness_nm")
    if budget is not None and sum(d for _, d in layers) > budget+1e-9:
        raise ValueError("Design exceeds total thickness budget")
    return layers


def calculate_spectra(optics, layers, angles_deg, chunk_size):
    """Native TMM-fast, s/p averaged; wavelength chunks bound peak memory."""
    indices = np.stack([optics.incident_index]+[optics.material_indices[i] for i,_ in layers]
                       +[optics.substrate_index])[None]
    thickness = np.asarray([[np.inf]+[d for _,d in layers]+[np.inf]])*1e-9
    spectra = {q: np.empty((len(angles_deg),len(optics.wavelengths_nm))) for q in "RT"}
    with torch.no_grad():
        for start in range(0,len(optics.wavelengths_nm),chunk_size):
            sl = slice(start,start+chunk_size)
            polarizations = [coh_tmm(pol,indices[:,:,sl].copy(),thickness.copy(),
                                    np.deg2rad(angles_deg),optics.wavelengths_nm[sl]*1e-9,
                                    device=optics.cfg["tmm_device"]) for pol in ("s","p")]
            for q in "RT":
                values = [FilmOptics._array(result[q])[0] for result in polarizations]
                spectra[q][:,sl] = np.mean(values,axis=0)
    r,t = spectra["R"],spectra["T"]
    if (not all(np.isfinite(x).all() for x in (r,t)) or np.min(r)<-1e-7 or np.min(t)<-1e-7
            or np.max(r)>1+1e-7 or np.max(t)>1+1e-7 or np.max(r+t)>1+1e-7):
        raise RuntimeError("Nonfinite/nonphysical TMM spectrum")
    spectra = {q:np.clip(x,0,1) for q,x in spectra.items()}
    spectra["A"] = np.clip(1-spectra["R"]-spectra["T"],0,1)
    return spectra


def evaluate(cfg, designs, labels, settings, output, paper_reference=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    # Baseline always uses s/p average, including when training used one polarization.
    cfg = deepcopy(cfg)
    cfg["polarization"] = "unpolarized"
    training_grid = deepcopy(cfg["wavelengths_nm"])
    wave = grid(training_grid,"wavelengths_nm")
    step = settings.get("wavelength_step_nm")
    if step is not None:
        if isinstance(step,bool) or not isinstance(step,(int,float)) or not np.isfinite(step) or step<=0:
            raise ValueError("wavelength_step_nm must be positive and finite, or null to use config grid")
        dense = np.arange(wave[0],wave[-1],step)
        cfg["wavelengths_nm"] = np.unique(np.r_[dense,wave[-1]]).tolist()
    optics = FilmOptics(cfg)
    if (not np.allclose(optics.incident_index,1) or not np.allclose(optics.substrate_index,1)
            or optics.substrate_thickness_nm is not None):
        raise ValueError("Appendix B lamp model requires air on both sides and no finite substrate")
    if optics.wavelengths_nm[0] > 400 or optics.wavelengths_nm[-1] < 780:
        raise ValueError("Baseline requires the optical grid to cover at least 400-780 nm")
    gauss_angles, weights = hemisphere_quadrature(settings["angle_order"])
    # Plot the familiar uniform-angle mean AND the physically weighted hemisphere.
    uniform_angles = np.arange(90,dtype=float)
    angles, inverse = np.unique(np.r_[uniform_angles,gauss_angles],return_inverse=True)
    uniform_idx, gauss_idx = inverse[:90],inverse[90:]
    chunk = settings["wavelength_chunk_size"]
    if isinstance(chunk,bool) or not isinstance(chunk,int) or chunk<1:
        raise ValueError("wavelength_chunk_size must be a positive integer")
    cases = []
    for path,label in zip(designs,labels):
        path = Path(path).resolve()
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8-sig"))
        layers = read_layers(data,optics)
        cases.append((label,layers,{"path":str(path),"sha256":hashlib.sha256(raw).hexdigest()}))
    if paper_reference:
        from tools.evaluate_oml_task2_reference import TABLE
        mapping = {"SiN":"Si3N4"}
        rows = [{"material_id":optics.names.index(mapping.get(n,n)),"material":mapping.get(n,n),"thickness_nm":d}
                for n,d in TABLE]
        layers = read_layers({"layers":rows},optics)
        cases.append(("Paper42_current_nk",layers,{"paper": "Table 3, MLST 2 (2021) 025013",
                     "mapping":mapping,"note":"Published structure recalculated with CURRENT nk; not original chi=16.60"}))
    if not cases:
        raise ValueError("Supply at least one --design or --include-paper-reference")
    if len({x[0] for x in cases}) != len(cases):
        raise ValueError("Case labels must be distinct")
    paper_curve, paper_curve_source = None, None
    if settings.get("paper_curve_csv"):
        path=Path(settings["paper_curve_csv"])
        paper_curve=np.loadtxt(path,delimiter=",",skiprows=1,ndmin=2)
        if (paper_curve.shape[1]!=2 or len(paper_curve)<2 or not np.isfinite(paper_curve).all()
                or np.any(np.diff(paper_curve[:,0])<=0) or np.any(paper_curve[:,1]<0) or np.any(paper_curve[:,1]>1)):
            raise ValueError("paper_curve_csv requires ordered wavelength_nm,R columns in [0,1]")
        paper_curve_source={"file":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                            "note":"PDF Fig.7(a) f=1 vector extraction, not raw author data; plotted only, NOT used to compute chi"}
    # Validate thermal and eye settings before creating an experiment directory.
    kwargs = {key:settings[key] for key in ("power_w","area_mm2","radiating_sides","outside_reflectance","visible_range_nm","eye_csv")}
    eye_response(kwargs["eye_csv"])
    factors = settings["view_factors"]
    if not isinstance(factors,list) or not factors or len(factors)!=len(set(factors)):
        raise ValueError("view_factors must contain distinct factors")
    for factor in factors:
        enhancement(optics.wavelengths_nm,np.zeros(len(optics.wavelengths_nm)),view_factor=factor,**kwargs)
    output.mkdir(parents=True)
    write(output/"config_snapshot.json",cfg)
    write(output/"baseline_settings.json",settings)
    wave = optics.wavelengths_nm
    results,curves,emission_curves = [],[],[]
    rows = []
    for number,(label,layers,source) in enumerate(cases,1):
        spectra = calculate_spectra(optics,layers,angles,chunk)
        uniform = {q:spectra[q][uniform_idx].mean(0) for q in "RTA"}
        hemi = {q:weights@spectra[q][gauss_idx] for q in "RTA"}
        metrics = []
        for factor in factors:
            result = enhancement(wave,hemi["R"],view_factor=factor,**kwargs)
            if settings["tail_sensitivity"]:
                alternatives = [enhancement(wave,hemi["R"],view_factor=factor,
                                **{**kwargs,"outside_reflectance":value}) for value in (0.,1.)]
                result["tail_closure_chi_range"] = [x["chi"] for x in alternatives]
                result["tail_closure_temperature_range_k"] = [x["temperature_k"] for x in alternatives]
            metrics.append(result)
            rows.append({"design":label,"layers":len(layers),"view_factor":factor,
                         "chi":result["chi"],"temperature_k":result["temperature_k"],
                         "bare_temperature_k":result["bare_temperature_k"],
                         "chi_outside_R0":result.get("tail_closure_chi_range",[None,None])[0],
                         "chi_outside_R1":result.get("tail_closure_chi_range",[None,None])[1]})
            emission_curves.append((label,factor,result["temperature_k"],1-factor**2*hemi["R"]))
        visible = (wave>=480)&(wave<=700)
        record = {"label":label,"layer_count":len(layers),"total_thickness_nm":sum(d for _,d in layers),
                  "source":source,"enhancement":metrics,
                  "visible_480_700_hemispherical_mean_A":float(hemi["A"][visible].mean()),
                  "layers":[{"material_id":i,"material":optics.names[i],"thickness_nm":d} for i,d in layers]}
        results.append(record)
        curves.append((label,uniform["R"],hemi["R"]))
        with (output/f"spectra_{number:02d}.csv").open("w",newline="",encoding="utf-8-sig") as stream:
            writer=csv.writer(stream)
            writer.writerow(["wavelength_nm"]+[f"{mean}_{q}" for mean in ("uniform","hemisphere") for q in "RTA"])
            writer.writerows(zip(wave,*[v[q] for v in (uniform,hemi) for q in "RTA"]))
        np.savez_compressed(output/f"spectra_{number:02d}.npz",wavelengths_nm=wave,angles_deg=angles,
                            gauss_angles_deg=gauss_angles,gauss_weights=weights,**spectra)
        print(json.dumps({"design":label,"enhancement":[{k:x[k] for k in ("view_factor","chi","temperature_k")}
                                                         for x in metrics]},ensure_ascii=False),flush=True)
    assumptions = [
        "Appendix B reflection-only black-emitter model: epsilon_eff=1-f^2 R. No PPO reward or weights are used.",
        "Hemisphere weights 2*sin(theta)*cos(theta); Gauss-Legendre in sin(theta)^2. Curve also shows uniform 0..89 deg mean.",
        "Planck exitance M=pi*B; full thermal spectrum via Stefan-Boltzmann minus in-window recycled radiation.",
        "Two emitting faces are inferred from the reported T0 about 2578 K, not an explicit paper geometry; default T0=2576.81 K.",
        "Outside the measured wavelength window R uses the explicit constant closure; no nk extrapolation. R=0/1 sensitivity bounds are not measured designs.",
        "Sharpe 2005 energy response is reconstructed from Eq.5 and Stockman-Sharpe cone fundamentals, verified against Table 3; visible integral 400..780 nm.",
        "The paper model counts non-returned energy via 1-f^2 R; it does not separately account for coating absorption/re-emission. Check the visible A diagnostic.",
        "No tungsten emissivity, conduction, convection, melting, temperature-dependent nk or coating heat balance; temperatures are ideal model values.",
        "This is a reproducible baseline under disclosed closure assumptions, not a verified numerical reproduction of the paper's 16.60.",
    ]
    manifest = {"paper_url":"https://arxiv.org/html/2006.11940#A2",
                "repository":"https://github.com/hammer-wang/oml-ppo",
                "repository_commit":"ef63afe85e9c09cb3ccf0df01218c6264535ac9c",
                "source_audit":"outputs/references/oml_enhancement_source/audit.json",
                "timestamp":datetime.now().astimezone().isoformat(),"training_executed":False,
                "training_wavelength_grid":training_grid,
                "paper_curve":paper_curve_source,
                "settings":settings,"assumptions":assumptions,"material_datasets":optics.material_datasets,
                "source_sha256":{str(p.relative_to(PROJECT_DIR)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__),PROJECT_DIR/"film_rl/enhancement.py")}}
    write(output/"baseline.json",{"manifest":manifest,"results":results})
    with (output/"comparison.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True,layout="constrained")
    for label,uniform,hemi in curves:
        axes[0].plot(wave,uniform,label=label);axes[1].plot(wave,hemi,label=label)
    if paper_curve is not None:
        axes[0].plot(paper_curve[:,0],paper_curve[:,1],color="black",ls=":",lw=2,
                     label="Paper Fig.7 f=1 (plot-derived)")
    for ax,title in zip(axes,("Uniform 0-89 deg mean (curve comparison)","Hemisphere weighted (enhancement input)")):
        ax.axvspan(480,700,color="green",alpha=.10)
        ax.plot(wave,np.where((wave>=480)&(wave<=700),0.,1.),"k--",lw=1,label="Paper target R")
        ax.set(title=title,ylabel="Reflectance",ylim=(-.02,1.02));ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[-1].set_xlabel("Wavelength (nm)")
    fig.savefig(output/"reflection.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5),layout="constrained")
    bare=results[0]["enhancement"][0]["bare_temperature_k"]
    ax.plot(wave,planck_exitance(wave,bare),"k--",label=f"Bare blackbody {bare:.1f} K")
    for label,factor,temp,eps in emission_curves:
        ax.plot(wave,planck_exitance(wave,temp)*eps,label=f"{label}, f={factor:g}, {temp:.1f} K")
    ax.set(xlabel="Wavelength (nm)",ylabel="Net exitance per emitting face (W m$^{-2}$ nm$^{-1}$)",
           title="Appendix B effective emission model, equal input power")
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(output/"emission.png",dpi=170);plt.close(fig)
    lines=["# 灯泡滤光片 baseline：反射率曲线与增强因子","",
           "反射率见 reflection.png（均匀角度及半球加权各一幅）。增强因子不使用训练 score 或波段权重。",
           "","| 设计 | 层数 | f | chi | 平衡温度 K | 波段外 R=0/1 的 chi 范围 |",
           "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        interval = (f'{row["chi_outside_R0"]:.4f}–{row["chi_outside_R1"]:.4f}'
                    if row["chi_outside_R0"] is not None else "未计算")
        lines.append(f'| {row["design"]} | {row["layers"]} | {row["view_factor"]} | {row["chi"]:.4f} | {row["temperature_k"]:.2f} | {interval} |')
    lines += ["","默认主值采用波段外 R=0；范围上端是假设波段外全部反射的敏感性界限，并非实际性能。",
              "不同设计必须使用同一 baseline_settings.json、光学数据和波长范围。","",
              "原文报告 chi=16.60 (f=1)、10.67 (f=0.95)，这些是文献值，未当作本程序算出的参考值。",
              "复现限制：辐射面积归一化和波段外光谱未完全公开，因此不宣称严格复现16.60。","",
              "## 可追溯计算约定","",*[f"- {x}" for x in assumptions],"",
              "baseline.json 包含输入文件校验、材料来源、模型假设、各f下的chi、温度、功率残差和膜结构。",
              "comparison.csv 用于汇总；spectra_XX.csv 保存均匀及半球平均R/T/A，npz保留角度光谱。"]
    (output/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=PROJECT_DIR/"configs/incandescent_task.json")
    parser.add_argument("--settings",type=Path,default=PROJECT_DIR/"configs/incandescent_baseline.json")
    parser.add_argument("--design",type=Path,nargs="+",default=[])
    parser.add_argument("--labels",nargs="+",help="One unique label for each design")
    parser.add_argument("--include-paper-reference",action="store_true",help="Recalculate Table 3 with current nk")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    labels=args.labels or [f"{i:02d}_{p.parent.name}_{p.stem}" for i,p in enumerate(args.design,1)]
    if len(labels)!=len(args.design):parser.error("--labels count must match --design")
    try:
        cfg=load_config(args.config,resolve_materials=True,interactive=False)
        settings=json.loads(args.settings.read_text(encoding="utf-8-sig"))
        for key in ("eye_csv","paper_curve_csv"):
            if settings.get(key):
                path=Path(settings[key])
                settings[key]=str((path if path.is_absolute() else args.settings.resolve().parent/path).resolve())
        output=args.output or PROJECT_DIR/"outputs"/datetime.now().strftime("lamp_baseline_%Y%m%d_%H%M%S")
        torch.set_num_threads(1)
        result=evaluate(cfg,args.design,labels,settings,output,args.include_paper_reference)
    except (ValueError,KeyError,FileNotFoundError,FileExistsError) as error:
        parser.error(str(error))
    print(f"Baseline report: {result/'README.md'}")


if __name__=="__main__":
    main()
