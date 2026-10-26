"""Plot a saved design against the supplied OML-PPO image and ideal target.

Example: python plot_design_spectrum.py --design outputs/incandescent_improvement/best_design.json
"""
import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from calculate_enhancement import design_config, source_record
from evaluate_baseline import calculate_spectra, read_layers
from film_rl.config import grid
from film_rl.enhancement import hemisphere_quadrature
from film_rl.optics import FilmOptics

REFERENCE = Path(__file__).parent/'film_rl/data/omlppo_f1_image_reference.csv'


def plot_design(design_path, config_path=None, output=None, wavelength_step_nm=2., averaging='uniform'):
    design_path=Path(design_path).resolve()
    data=json.loads(design_path.read_text(encoding='utf-8-sig'))
    cfg,config_source=design_config(design_path,config_path)
    cfg=deepcopy(cfg)
    if not np.isfinite(wavelength_step_nm) or wavelength_step_nm<=0:
        raise ValueError('wavelength step must be positive and finite')
    original_wave=grid(cfg['wavelengths_nm'],'wavelengths_nm')
    low,high=max(400.,original_wave[0]),min(3000.,original_wave[-1])
    if low>=high:
        raise ValueError('Design wavelength range does not overlap reference range 400-3000 nm')
    cfg['wavelengths_nm']=np.unique(np.r_[np.arange(original_wave[0],original_wave[-1],wavelength_step_nm),original_wave[-1]]).tolist()
    cfg['tmm_device']='cpu'
    cfg['polarization']='unpolarized'
    optics=FilmOptics(cfg)
    for row in data.get('material_datasets',[]):
        now=next((x for x in optics.material_datasets if x['material']==row['material']),None)
        if now is None or row['sha256']!=now['sha256'] or row.get('assumed_zero_k',False)!=now.get('assumed_zero_k',False):
            raise ValueError('Material data/absorption assumption changed: '+row['material'])
    layers=read_layers(data,optics)
    if averaging=='uniform':
        angles=np.arange(90,dtype=float); weights=np.ones(90)/90
    else:
        angles,weights=hemisphere_quadrature(64)
    spectra=calculate_spectra(optics,layers,angles,128)
    selected=(optics.wavelengths_nm>=low)&(optics.wavelengths_nm<=high)
    curves={q:(weights@spectra[q])[selected] for q in 'RTA'}
    ref=np.loadtxt(REFERENCE,delimiter=',',skiprows=1)
    wave=optics.wavelengths_nm[selected]
    reference=np.interp(wave,ref[:,0],ref[:,1])
    target=np.where((wave>=480)&(wave<=700),0.,1.)
    output=Path(output).resolve() if output else design_path.parent/'spectrum_visualization'
    if output.suffix:
        raise ValueError('--output must be a directory, not an image filename')
    output.mkdir(parents=True,exist_ok=True)
    files=[output/name for name in ('spectrum.png','spectrum.csv','metadata.json')]
    if any(p.exists() for p in files):
        raise FileExistsError('Output already contains results; choose a new --output directory')
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(wave/1000,curves['R'],color='crimson',lw=2,label='Your design')
    ax.plot(ref[:,0]/1000,ref[:,1],color='blue',lw=2,label='OML-PPO (f = 1, image-digitized)')
    # Separate segments preserve the inclusive [480,700] zero-reflection band.
    ax.plot([.4,.48,.48,.7,.7,3.],[1,1,0,0,1,1],color='black',ls='--',lw=2,label='Target Spectrum')
    ax.set(xlabel='Wavelength (um)',ylabel='Average Reflectivity',xlim=(.4,3.),ylim=(-.02,1.02))
    ax.set_title(f'{len(layers)} layers | '+('Uniform mean: 0-89 deg, step 1 deg' if averaging=='uniform' else 'Hemisphere weighted mean'))
    ax.grid(alpha=.18);ax.legend(loc='lower right',fontsize=10)
    fig.tight_layout();fig.savefig(files[0],dpi=180);plt.close(fig)
    with files[1].open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['wavelength_nm','design_R','design_T','design_A','OMLPPO_image_R','target_R'])
        writer.writerows(zip(wave,curves['R'],curves['T'],curves['A'],reference,target))
    meta={'design':source_record(design_path)[1],'config':config_source,'averaging':averaging,
          'angles_deg':angles.tolist(),'polarization':'unpolarized','wavelength_step_nm':wavelength_step_nm,
          'reference':'Blue solid f=1 curve digitized from user-supplied 901x651 image; approximate, not author raw data. No optical recalculation of reference.',
          'reference_pixel_calibration':{'x400nm':185,'x3000nm':839,'y_R1':60,'y_R0':542},
          'target':{'R':0,'interval_nm_inclusive':[480,700],'R_elsewhere':1},
          'note':'This displayed ideal target is independent of the saved training objective. No f^2 scaling of design reflectance. Reference angular averaging is not verified from this image. 90 deg excluded as singular grazing endpoint.',
          'reference_MAE':float(np.mean(abs(curves['R']-reference))),
          'ideal_target_MAE':float(np.mean(abs(curves['R']-target))),
          'training_executed':False}
    files[2].write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',required=True,type=Path,help='Saved design JSON path')
    parser.add_argument('--config',type=Path,help='Original config if no nearby snapshot exists')
    parser.add_argument('--output',type=Path,help='New output directory')
    parser.add_argument('--wavelength-step-nm',type=float,default=2.)
    parser.add_argument('--averaging',choices=['uniform','hemisphere'],default='uniform')
    args=parser.parse_args()
    torch.set_num_threads(1)
    try:
        files=plot_design(args.design,args.config,args.output,args.wavelength_step_nm,args.averaging)
    except (ValueError,OSError,KeyError) as error:
        parser.error(str(error))
    for file in files:print(file)

if __name__=='__main__':main()
