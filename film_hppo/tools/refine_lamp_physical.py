"""HPPO-seeded joint thickness optimization with physical band constraints.

This is explicitly mixed optimization. A candidate is deployable only after
ordinary TMM re-evaluation and common dense-grid validation.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
import numpy as np
import torch
from scipy.optimize import minimize
from tmm_fast import coh_tmm
from film_rl.config import load_config
from film_rl.design import compact_design
from film_rl.improve import ThicknessObjective,layers_of,train_warm_start
from film_rl.lamp_metrics import lamp_metrics,lamp_key,IR_BANDS
from film_rl.optics import FilmOptics
from tools.optimize_lamp_hppo import split_targets,read,write


class PhysicalObjective(ThicknessObjective):
    def __init__(self,optics,ids,folder):
        super().__init__(optics,ids)
        self.ids=ids;self.folder=folder;self.best=None;self.best_key=None;self.history=[]
        self.low=optics.cfg['thickness_min_nm'];self.span=optics.cfg['thickness_max_nm']-self.low
        w=optics.wavelengths_nm
        self.masks=[(w>=lo)&(w<=hi) for lo,hi in [(480,700),(300,470),(710,3000),*IR_BANDS]]

    def __call__(self,unit):
        x=torch.tensor(unit,dtype=torch.float64,requires_grad=True)
        d=self.low+x*self.span
        thick=torch.cat((x.new_tensor([np.inf]),d,x.new_tensor([np.inf])))[None]*1e-9
        raw=[coh_tmm(pol,self.indices,thick,self.angles,self.wave,device='cpu') for pol in self.pols]
        r=torch.stack([z['R'][0] for z in raw]).mean(0)
        t=torch.stack([z['T'][0] for z in raw]).mean(0)
        a=1-r-t
        visible_t=t[:,self.masks[0]].mean();visible_r=r[:,self.masks[0]].mean();visible_a=a[:,self.masks[0]].mean()
        uv=r[:,self.masks[1]].mean();ir=r[:,self.masks[2]].mean()
        sub=torch.stack([r[:,mask].mean() for mask in self.masks[3:]])
        # Smooth squared constraint penalties preserve meaningful gradients.
        # Save feasible intermediate iterates independently from the loss.
        loss=-visible_t+.1*visible_r+150*torch.relu(.75-uv).square()+200*torch.relu(.87-ir).square()
        loss=loss+80*torch.relu(.75-sub).square().sum()+100*torch.relu(visible_a-.05).square()
        grad,=torch.autograd.grad(loss,x)
        if not torch.isfinite(loss) or not torch.isfinite(grad).all():
            raise FloatingPointError('Nonfinite physical objective or gradient')
        self.calls+=1
        m=dict(visible_T=float(visible_t.detach()),visible_R=float(visible_r.detach()),
               visible_A=float(visible_a.detach()),UV_R=float(uv.detach()),IR_R=float(ir.detach()),
               IR_subband_R=sub.detach().tolist())
        feasible=m['UV_R']>=.70 and m['IR_R']>=.85 and min(m['IR_subband_R'])>=.70 and m['visible_A']<=.05
        key=(feasible,m['visible_T'] if feasible else -float(loss.detach()))
        row=dict(call=self.calls,loss=float(loss.detach()),feasible=feasible,**m)
        self.history.append(row)
        if self.best_key is None or key>self.best_key:
            self.best_key=key;self.best=d.detach().numpy().copy()
            report=self.optics.evaluate(list(zip(self.ids,self.best)))
            report['application_metrics']=lamp_metrics(report)
            write(self.folder/'best_design.json',compact_design(report))
        if self.calls%10==0:
            print(json.dumps(row),flush=True)
            write(self.folder/'progress.json',row)
        return float(loss.detach()),grad.detach().numpy()


def main():
    torch.set_num_threads(1)
    root=PROJECT/'outputs'/sys.argv[1]
    recover='--recover' in sys.argv
    folder=root/'trials'/'physical_refinement';folder.mkdir(parents=True,exist_ok=recover)
    cfg=load_config(PROJECT/'outputs/lamp_four_targets/20261021_195347/evaluation_config.json')
    cfg['angles_deg']=list(range(2,90,5));cfg['spectral_targets']=split_targets();cfg['design_selection']='lamp_pareto'
    cfg['_reference']={'method':'HPPO material sequences + TMM-autograd physical-constraint thickness refinement',
        'note':'Mixed optimization, with supervised trajectory initialization before final PPO. No paper initialization.'}
    optics=FilmOptics(cfg)
    paths=[PROJECT/'outputs/lamp_four_targets/20261021_195347/best_design.json']
    paths += [root/'trials'/name/'candidate_0.json' for name in ['control_sigma08','narrow_sigma03']]
    summaries=read(folder/'summary.json') if recover else []
    reports=[]
    if recover:
        for branch in sorted(folder.glob('seed*')):
            path=branch/'best_design.json'
            if not path.is_file():continue
            report=read(path);reports.append(report)
            number=int(branch.name[4:])
            if not any(row['seed']==number for row in summaries):
                summaries.append(dict(seed=number,source=str(paths[number]),
                    after=report['application_metrics'],
                    optimization={'success':False,'message':'Progress JSON atomic replace hit a Windows sharing violation. Recovered saved incumbent; no optimization restart.'}))
        write(folder/'summary.json',summaries)
        paths=[]
    for number,path in enumerate(paths):
        if not path.is_file():continue
        data=read(path);layers=layers_of(data);ids=[i for i,_ in layers]
        branch=folder/f'seed{number}';branch.mkdir()
        objective=PhysicalObjective(optics,ids,branch)
        x=(np.array([d for _,d in layers])-objective.low)/objective.span
        initial=optics.evaluate(layers);start=time.perf_counter()
        # Central finite-difference check of a real TMM gradient before updates.
        if number==0:
            value,gradient=objective(x)
            errors=[]
            for position in [0,len(x)//2]:
                delta=np.zeros_like(x);delta[position]=1e-6
                numerical=(objective(x+delta)[0]-objective(x-delta)[0])/2e-6
                error=abs(numerical-gradient[position])/max(1.,abs(numerical),abs(gradient[position]))
                errors.append(float(error))
            if max(errors)>2e-4:raise AssertionError(('Gradient verification failed',errors))
            write(folder/'gradient_check.json',{'relative_errors':errors,'passed':True})
        result=minimize(objective,x,jac=True,method='SLSQP',bounds=[(0.,1.)]*len(x),
            options={'maxiter':100,'ftol':1e-8})
        report=optics.evaluate(list(zip(ids,objective.best)))
        report['application_metrics']=lamp_metrics(report)
        report['optimization']={'method':'physical-constraint SLSQP + TMM autograd',
            'pure_ppo':False,'source':str(path),'iterations':int(result.nit),'calls':objective.calls,
            'success':bool(result.success),'message':str(result.message)}
        report['candidate_source']={'stage':'physical_refinement','source_path':str(path),'seed_index':number}
        write(branch/'best_design.json',compact_design(report));write(branch/'history.json',objective.history)
        record=dict(seed=number,source=str(path),seconds=time.perf_counter()-start,
            before=lamp_metrics(initial),after=report['application_metrics'],optimization=report['optimization'])
        summaries.append(record);reports.append(report)
        write(folder/'summary.json',summaries)
        print(json.dumps(record),flush=True)
    def key(report):
        m=report['application_metrics'];feasible=m['UV_R']>=.70 and m['IR_R']>=.85 and m['IR_min_subband_R']>=.70 and m['visible_A']<=.05
        return(feasible,m['visible_T'],m['balanced_harmonic'])
    reports.sort(key=key,reverse=True);best=reports[0]
    write(folder/'best_design.json',compact_design(best))
    production=root/'production';production.mkdir()
    cfg['training'].update(seed=23,initial_sigma=.012,learning_rate=3e-6,n_sample=3000,
        batch_size=256,epoch_per_collect=3,entropy_weight_discrete=.0001,entropy_weight_continuous=0.,
        max_env_step=60000,max_train_iter=1000000,eval_freq=360,gae_lambda=1.,
        exp_name='runs/'+root.name+'/production')
    cfg['training'].pop('initial_model',None)
    write(production/'refined_design.json',compact_design(best))
    # This stage is explicitly supervised, not presented as a PPO discovery.
    train_warm_start(cfg,best,production,steps=2400)
    cfg['training']['initial_model']=str(production/'warm_start_model.pt')
    write(production/'training_config.json',cfg)
    write(folder/'complete.json',{'status':'refined_and_policy_initialized','summaries':summaries,
                               'production_config':str(production/'training_config.json')})
    print('REFINEMENT_AND_INITIALIZATION_COMPLETE',flush=True)


if __name__=='__main__':main()
