"""Evaluate published Table 3 under current optical data, without training."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(1)

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from film_rl.config import load_config
from film_rl.design import compact_design
from film_rl.optics import FilmOptics

TABLE = [
    ('SiO2',289),('SiN',268),('MgF2',185),('SiN',189),('SiC',214),('SiN',214),
    ('MgF2',210),('SiN',206),('SiC',205),('SiN',183),('MgF2',184),('SiN',179),
    ('SiC',203),('SiN',273),('SiC',210),('SiN',168),('MgF2',200),('SiC',227),
    ('SiN',242),('MgF2',222),('SiC',228),('MgF2',216),('SiC',229),('MgF2',203),
    ('SiC',101),('MgF2',209),('SiC',121),('MgF2',225),('SiC',117),('MgF2',224),
    ('SiC',122),('MgF2',235),('SiC',127),('MgF2',230),('SiC',234),('MgF2',218),
    ('SiC',235),('MgF2',220),('SiC',231),('MgF2',216),('SiC',233),('Al2O3',95),
]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def hemispherical(values, angles):
    """Integrate 2*sin(theta)*cos(theta); endpoint weight at 90 deg is zero."""
    theta = np.deg2rad(angles)
    weights = 2*np.sin(theta)*np.cos(theta)
    numerator = np.trapz(np.vstack((values*weights[:,None], np.zeros((1,values.shape[1])))),
                         np.append(theta, np.pi/2), axis=0)
    denominator = np.trapz(np.append(weights, 0), np.append(theta,np.pi/2))
    return numerator/denominator


def independent_rt(indices, thickness_nm, wavelength_nm, angle_deg, pol):
    """Independent characteristic-matrix calculation at a single point."""
    sine = indices[0]*np.sin(np.deg2rad(angle_deg))
    cosine = np.sqrt(1-(sine/indices)**2+0j)
    eta = indices*cosine if pol == 's' else indices/cosine
    matrix = np.eye(2, dtype=complex)
    for n, c, admittance, d in zip(indices[1:-1], cosine[1:-1], eta[1:-1], thickness_nm):
        phase = 2*np.pi*n*c*d/wavelength_nm
        matrix = matrix @ np.array([[np.cos(phase), -1j*np.sin(phase)/admittance],
                                     [-1j*admittance*np.sin(phase),np.cos(phase)]])
    b = matrix[0,0]+matrix[0,1]*eta[-1]
    c = matrix[1,0]+matrix[1,1]*eta[-1]
    r = (eta[0]*b-c)/(eta[0]*b+c)
    t = 2*eta[0]/(eta[0]*b+c)
    return abs(r)**2, float(eta[-1].real/eta[0].real)*abs(t)**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/incandescent_task.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    output = args.output or PROJECT/'outputs/references/oml_ppo_task2'/datetime.now().strftime('table3_%Y%m%d_%H%M%S')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    optics = FilmOptics(cfg)
    mapping = {'SiN': 'Si3N4'}
    layers = [(optics.names.index(mapping.get(n,n)),float(d)) for n,d in TABLE]
    assert len(layers)==42
    design = {'layer_count':42, 'total_thickness_nm':sum(d for _,d in layers),
              'source':'Published Table 3, MLST 2 (2021) 025013',
              'material_mapping':mapping,
              'layers':[{'material_id':i, 'material':optics.names[i], 'thickness_nm':d,
                         'paper_material':TABLE[j][0]} for j,(i,d) in enumerate(layers)]}
    write(output/'input_design.json',design)
    write(output/'config_snapshot.json',cfg)
    report = optics.evaluate(layers)
    write(output/'design_summary.json',compact_design(report))
    np.savez_compressed(output/'spectra.npz', wavelengths_nm=optics.wavelengths_nm,
                        angles_deg=optics.angles_deg, **{q:np.asarray(report[q]) for q in 'RTA'})
    s_cfg=deepcopy(cfg);s_cfg['polarization']='s'
    s_optics=FilmOptics(s_cfg);s_report=s_optics.evaluate(layers)
    zero_optics=FilmOptics(cfg)
    si_index=zero_optics.names.index('Si3N4')
    zero_optics.material_indices[si_index]=zero_optics.material_indices[si_index].real.astype(complex)
    zero_report=zero_optics.evaluate(layers)
    wave=optics.wavelengths_nm;angles=optics.angles_deg
    current={q:np.asarray(report[q]) for q in 'RTA'}
    curves={
        'current_uniform':current['R'].mean(axis=0),
        'current_hemisphere':hemispherical(current['R'],angles),
        's_only_uniform':np.asarray(s_report['R']).mean(axis=0),
        'SiN_zero_k_uniform':np.asarray(zero_report['R']).mean(axis=0),
        'SiN_zero_k_hemisphere':hemispherical(np.asarray(zero_report['R']),angles),
    }
    # Confirm TMM-fast using an independent implementation (no data assumptions changed).
    max_diff=0.
    ds=np.asarray([d for _,d in layers])
    for angle in (0.,30.,60.):
        aidx=int(np.flatnonzero(angles==angle)[0])
        for wavelength in (400.,550.,1000.,2000.,3000.):
            widx=int(np.flatnonzero(wave==wavelength)[0])
            indices=np.array([1]+[optics.material_indices[i,widx] for i,_ in layers]+[1],dtype=complex)
            values=[independent_rt(indices,ds,wavelength,angle,p) for p in ('s','p')]
            independent=np.mean(values,axis=0)
            max_diff=max(max_diff,float(abs(independent[0]-current['R'][aidx,widx])),
                         float(abs(independent[1]-current['T'][aidx,widx])))
    assert max_diff<1e-9, max_diff
    paper_path=PROJECT/'outputs/references/oml_ppo_task2/figure7_omlppo_f1_vector.csv'
    paper=np.loadtxt(paper_path,delimiter=',',skiprows=1)
    # Calibration roundoff: paper endpoints are about 400 and 2999 nm.
    overlap=(wave>=400)&(wave<=2990)
    reference=np.interp(wave[overlap],paper[:,0],paper[:,1])
    differences={name:float(np.abs(values[overlap]-reference).mean()) for name,values in curves.items()}
    band_rows=[]
    for low,high in [(300.,470.),(480.,700.),(710.,3000.)]:
        mask=(wave>=low)&(wave<=high)
        p_mask=mask&overlap
        band_rows.append({'range_nm':[low,high],
            **{f'mean_{q}':float(current[q][:,mask].mean()) for q in 'RTA'},
            'normal_R':float(current['R'][0,mask].mean()),
            'normal_T':float(current['T'][0,mask].mean()),
            'normal_A':float(current['A'][0,mask].mean()),
            'hemispherical_R':float(curves['current_hemisphere'][mask].mean()),
            'paper_curve_mean_R_estimate':float(np.interp(wave[p_mask],paper[:,0],paper[:,1]).mean()),
            'current_mean_R_on_paper_grid':float(current['R'][:,p_mask].mean()),
            'hemispherical_mean_R_on_paper_grid':float(curves['current_hemisphere'][p_mask].mean()),
            'paper_comparison_range_nm':[float(wave[p_mask][0]),float(wave[p_mask][-1])],
        })
    summary={'layer_count':42, 'total_thickness_nm':design['total_thickness_nm'],
        'config':str(config_path),'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'material_mapping':mapping,'score_current_weights':report['score'],
        'bands':band_rows,'comparison_curve_MAE_over_400_2990nm':differences,
        'independent_solver_max_RT_difference':max_diff,'independent_solver_points':15,
        'energy_conservation_max_error':float(abs(current['R']+current['T']+current['A']-1).max()),
        'source_optical_datasets':report['material_datasets'],
        'sensitivity_bands':{
            name:[{k:t[k] for k in ('wavelength_range_nm','mean_actual','mae')}
                  for t in result['target_errors']]
            for name,result in [('s_only',s_report),('SiN_k_zero',zero_report)]},
        'counterfactual_note':'s-only and SiN-zero-k cases are sensitivity checks only. No active config/data changed.',
        'paper_curve_note':'PDF vector-curve extraction, not raw author data. Paper curve begins at about 400 nm; no paper comparison below 400 nm.',
        'angular_note':'Uniform 0-89 deg mean is the current task. Hemisphere integral uses sin(theta)*cos(theta), plus a zero-weight 90 deg endpoint.',
        'training_executed':False}
    write(output/'comparison.json',summary)
    with (output/'angle_averaged_spectra.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['wavelength_nm','mean_R','mean_T','mean_A','normal_R','normal_T','normal_A',*curves.keys()])
        for j,w in enumerate(wave):
            writer.writerow([w,*[current[q][:,j].mean() for q in 'RTA'],
                             *[current[q][0,j] for q in 'RTA'],*[v[j] for v in curves.values()]])
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    axes[0,0].plot(paper[:,0],paper[:,1]*100,label='Paper Fig.7(a), f=1 (vector-extracted)',color='black',lw=2)
    axes[0,0].plot(wave,curves['current_uniform']*100,label='Current data: uniform angle mean')
    axes[0,0].plot(wave,curves['current_hemisphere']*100,label='Current data: hemisphere weighted',ls='--')
    axes[0,0].set(title='Published structure: reflectance comparison',xlim=(400,3000),ylabel='Reflectance (%)')
    for q in 'RTA':axes[0,1].plot(wave,current[q].mean(axis=0)*100,label=q)
    axes[0,1].set(title='Current data: unpolarized, 0-89 deg mean',xlim=(300,3000),ylabel='Power fraction (%)')
    axes[1,0].plot(paper[:,0],paper[:,1]*100,label='Paper Fig.7(a), f=1',color='black',lw=2)
    for key,label in [('current_uniform','Current data'),('SiN_zero_k_uniform','SiN k=0 (diagnostic only)'),('s_only_uniform','s polarization only')]:
        axes[1,0].plot(wave,curves[key]*100,label=label)
    axes[1,0].set(title='Sensitivity checks; training configuration unchanged',xlim=(400,3000),ylabel='Reflectance (%)')
    visible=(wave>=480)&(wave<=700)
    for q in 'RTA':axes[1,1].plot(angles,current[q][:,visible].mean(axis=1)*100,label=q)
    axes[1,1].set(title='480-700 nm: angle dependence',xlabel='Incidence angle (deg)',ylabel='Band mean (%)',xlim=(0,89))
    for ax in axes.flat:
        ax.set_ylim(0,100);ax.grid(alpha=.25);ax.legend(fontsize=8)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.set_xlabel('Wavelength (nm)');ax.axvspan(480,700,color='gold',alpha=.1)
    fig.suptitle('OML-PPO Table 3: 42 fixed layers, current optical constants; no optimization')
    fig.savefig(output/'comparison.png',dpi=170)
    plt.close(fig)
    notes=['# Published 42-layer filter: current-data TMM check','',
           f'Configuration: {config_path}',f'Total thickness: {design["total_thickness_nm"]} nm. No training or optimization.',
           '','The paper SiN label is mapped explicitly to current Si3N4.yml; equality of the physical optical constants is not assumed.',
           'comparison.png overlays current calculations and the PDF vector curve. No reflection band means were tabulated in the paper.',
           'Paper estimates here are derived from its graph; current values are computed directly.',
           'The graph blue curve covers about 400-3000 nm, so its UV comparison only uses 400-470 nm.',
           'The active config uses 300-3000 nm. These two domains are not identical.',
           '','Full per-angle spectra are in spectra.npz; readable design_summary.json is compact.',
           'Sensitivity cases change only in-memory copies. They are not repairs or verified original-paper settings.',
           'No visible enhancement factor chi is claimed: it needs additional spectral/thermal assumptions.',
           'Independent 2x2 characteristic-matrix checks agree with TMM-fast; see comparison.json.']
    (output/'README.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output), 'layer_count':42, 'total_nm':design['total_thickness_nm'],
                      'score':report['score'],'bands':band_rows,'curve_MAE':differences,
                      'independent_solver_max_difference':max_diff},indent=2))


if __name__=='__main__':
    main()
