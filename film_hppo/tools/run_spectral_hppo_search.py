"""Bounded configuration-only native HPPO search; no design post-optimization."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from film_rl.config import load_config
from run_seed_batch import prepare, run_one, refresh_summary


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--budget',type=int,default=180000)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    root=args.root.resolve()
    if (root/'plan.json').exists():raise FileExistsError('Existing plan preserved; use a new experiment root')
    if args.budget<3000 or args.workers<1:parser.error('Invalid budget/workers')
    cfg=load_config(root/'evaluation_config.json')
    cfg['angles_deg']=list(range(2,90,5))
    cfg['training'].pop('initial_model',None)
    cfg['training'].update(n_sample=3000,batch_size=128,epoch_per_collect=5,
                           learning_rate=5e-5,entropy_weight=.001,max_env_step=args.budget,
                           max_train_iter=1000000,eval_freq=480)
    cfg['_reference']['angle_grid_note']='Training: 18 five-degree-bin centers; final validation uses original 90 angles.'
    cfg['_reference']['training_note']='Fresh HPPO/VAC, same architecture/actions/terminal linear reward, no improve, no imitation, no thickness optimizer.'
    batches=[]
    for label,gae in [('l095',.95),('l100',1.)]:
        trial=deepcopy(cfg);trial['training']['gae_lambda']=gae
        path=root/'configs'/f'{label}.json';write(path,trial)
        batch=prepare(path,group=f'lamp_R_hppo_{label}',seeds=[0,1])
        manifest=json.loads((batch/'manifest.json').read_text(encoding='utf-8'))
        batches.append({'label':label,'gae_lambda':gae,'batch':str(batch),'manifest':manifest})
    plan={'method':'Existing FilmPPOPolicy + hybrid FilmVAC; configuration-only matched GAE comparison',
          'budget_per_run':args.budget,'seeds':[0,1],'workers':args.workers,'batches':batches,
          'training_angles':cfg['angles_deg'],'validation_angles':list(range(90)),
          'goals':cfg['spectral_targets'],'no_improve':True,'no_paper_start':True}
    write(root/'plan.json',plan)
    jobs=[(b,e) for seed in [0,1] for b in batches for e in b['manifest']['runs'] if e['seed']==seed]
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(run_one,e,Path(b['batch']),b['manifest']['runtime_environment']) for b,e in jobs}
        while pending:
            done,pending=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:future.result()
            for b in batches:refresh_summary(Path(b['batch']),b['manifest'],args.workers,started)
    print('SEARCH_COMPLETE='+str(root),flush=True)


if __name__=='__main__':main()
