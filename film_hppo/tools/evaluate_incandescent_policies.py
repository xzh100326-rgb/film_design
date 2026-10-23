"""Independent, dense-grid sampling evaluation of saved native HPPO policies."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from ding.config import compile_config
from ding.utils import set_pkg_seed
from film_rl.config import load_config
from film_rl.design import compact_design
from film_rl.env import ThinFilmEnv
from film_rl.hppo import build_configs, build_model
from film_rl.policy import FilmPPOPolicy

torch.set_num_threads(1)


def write(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=64)
    parser.add_argument('--variants',nargs='+',help='Only evaluate these plan batch labels')
    parser.add_argument('--seeds',type=int,nargs='+',help='Only evaluate these training seeds')
    parser.add_argument('--checkpoint',choices=['best','last'],default='best')
    parser.add_argument('--deterministic',action='store_true',help='One greedy-material / Gaussian-mean inference; no optimization')
    parser.add_argument('--no-aggregate',action='store_true',help='Write per-checkpoint files only for parallel evaluation')
    args=parser.parse_args()
    if args.deterministic:args.episodes=1
    elif args.episodes<2:parser.error('At least two episodes are required for dispersion statistics')
    root=args.root.resolve()
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    evaluation_cfg=load_config(root/'evaluation_config.json')
    evaluation_digest=hashlib.sha256(json.dumps(evaluation_cfg,sort_keys=True).encode('utf-8')).hexdigest()
    output=root/'policy_evaluation'
    output.mkdir(exist_ok=True)
    summaries=[]
    for batch in plan['batches']:
        batch_label=batch.get('label') or ('e'+str(batch['epochs']))
        if args.variants and batch_label not in args.variants:continue
        for path,expected in batch['manifest']['material_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f'Material changed since training: {path}')
        for entry in batch['manifest']['runs']:
            if args.seeds and entry['seed'] not in args.seeds:continue
            record=Path(entry['record'])
            if not record.exists() or json.loads(record.read_text(encoding='utf-8'))['status']!='complete':
                continue
            label=f'{batch_label}_seed{entry["seed"]}'
            if args.checkpoint=='best':
                checkpoint=Path(entry['run_dir'])/'ckpt/ckpt_best.pth.tar'
            else:
                checkpoint=max((Path(entry['run_dir'])/'ckpt').glob('iteration_*.pth.tar'),
                               key=lambda p:int(p.name.split('_')[1].split('.')[0]))
                label+='_last'
            if args.deterministic:label+='_deterministic'
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            case=output/label
            case.mkdir(exist_ok=True)
            if (case/'summary.json').exists():
                previous=json.loads((case/'summary.json').read_text(encoding='utf-8'))
                if (previous['checkpoint_sha256']==digest and previous['episodes']==args.episodes
                        and previous.get('evaluation_config_sha256')==evaluation_digest):
                    summaries.append(previous)
                    continue
            cfg=load_config(entry['config'])
            main_cfg,create=build_configs(cfg)
            create.policy.type+='_command'
            compiled=compile_config(main_cfg,seed=cfg['training']['seed'],auto=True,create_cfg=create,save_cfg=False)
            model=build_model(cfg,main_cfg)
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(payload['model'],strict=True)
            policy=FilmPPOPolicy(compiled.policy,model=model,enable_field=['eval'])
            dense_cfg=deepcopy(evaluation_cfg)
            dense_cfg['min_layers']=cfg['min_layers']
            env=ThinFilmEnv(dense_cfg)
            rows=[]
            best_score=-1
            try:
                for episode in range(args.episodes):
                    set_pkg_seed(20261021+episode,use_cuda=False)
                    observation=env.reset()
                    policy.eval_mode.reset([0])
                    for _ in range(cfg['max_layers']+1):
                        if args.deterministic:
                            model.eval()
                            with torch.no_grad():
                                logits=model(torch.from_numpy(observation).unsqueeze(0),mode='compute_actor')['logit']
                            action={'action_type':np.asarray(int(logits['action_type'][0].argmax().item())),
                                    'action_args':logits['action_args']['mu'][0].detach().cpu().numpy()}
                        else:
                            result=policy.eval_mode.forward({0:torch.from_numpy(observation)})[0]
                            action={name:value.detach().cpu().numpy() for name,value in result['action'].items()}
                        timestep=env.step(action)
                        observation=timestep.obs
                        if timestep.done:
                            report=compact_design(timestep.info)
                            bands=report['target_errors']
                            rows.append(dict(episode=episode,score=report['score'],layers=report['layer_count'],
                                             visible_T=bands[1]['mean_T'],IR_R=bands[2]['mean_R'],
                                             visible_A=bands[1]['mean_A']))
                            if report['score']>best_score:
                                best_score=report['score']
                                write(case/'best_sampled_design.json',report)
                                np.savez_compressed(case/'best_sampled_spectra.npz',
                                                    wavelengths_nm=np.asarray(timestep.info['wavelengths_nm']),
                                                    angles_deg=np.asarray(timestep.info['angles_deg']),
                                                    **{q:np.asarray(timestep.info[q]) for q in 'RTA'})
                            break
                    else:
                        raise RuntimeError('Episode exceeded its legal length')
            finally:
                env.close()
            scores=np.array([r['score'] for r in rows])
            summary=dict(experiment=label,checkpoint=str(checkpoint),checkpoint_sha256=digest,
                         run_dir=entry['run_dir'],training_config=entry['config'],
                         evaluation_config_sha256=evaluation_digest,
                         selection=f'Native {args.checkpoint} checkpoint; no checkpoint selection on these evaluation episodes',
                         sampling=('Greedy material and Gaussian latent mean from the same trained FilmVAC; no optimization'
                                   if args.deterministic else 'Unmodified FilmPPOPolicy.eval_mode; stochastic native hybrid wrapper'),
                         deterministic=args.deterministic,
                         episodes=len(rows),episode_seeds=list(range(20261021,20261021+args.episodes)),
                         score_mean=float(scores.mean()),score_std=float(scores.std(ddof=1)) if len(rows)>1 else 0.,
                         score_standard_error=float(scores.std(ddof=1)/np.sqrt(len(rows))) if len(rows)>1 else 0.,
                         score_max=float(scores.max()),score_min=float(scores.min()),
                         mean_layers=float(np.mean([r['layers'] for r in rows])),
                         mean_visible_T=float(np.mean([r['visible_T'] for r in rows])),
                         mean_IR_R=float(np.mean([r['IR_R'] for r in rows])),
                         mean_visible_A=float(np.mean([r['visible_A'] for r in rows])),
                         best_sampled_design=str(case/'best_sampled_design.json'))
            write(case/'summary.json',summary)
            with (case/'episodes.csv').open('w',newline='',encoding='utf-8-sig') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            summaries.append(summary)
            print(label,'mean',summary['score_mean'],'max',summary['score_max'],'layers',summary['mean_layers'],flush=True)
    if args.no_aggregate:
        print('PER_CHECKPOINT_EVALUATION='+str(output),flush=True)
        return
    # A filtered pass must not discard summaries produced for other checkpoints.
    summaries=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(output.glob('*/summary.json'))]
    write(output/'summary.json',summaries)
    if summaries:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels=[r['experiment'] for r in summaries]
        x=np.arange(len(labels))
        fig,axes=plt.subplots(1,2,figsize=(15,5.5))
        axes[0].errorbar(x,[r['score_mean'] for r in summaries],
                         yerr=[1.96*r['score_standard_error'] for r in summaries],fmt='o',capsize=3)
        axes[0].set_xticks(x,labels,rotation=65,ha='right')
        axes[0].set_ylim(0,1);axes[0].set_ylabel('Dense target score')
        axes[0].set_title('Native checkpoints: new episodes, mean +/- 1.96 SE')
        for r in summaries:
            axes[1].scatter(r['mean_visible_T'],r['mean_IR_R'],label=r['experiment'])
        axes[1].legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=7)
        axes[1].set_xlim(0,1);axes[1].set_ylim(0,1)
        axes[1].set_xlabel('Mean visible transmission');axes[1].set_ylabel('Mean infrared reflection')
        axes[1].set_title('Average design trade-off; higher on both axes is better')
        for ax in axes:ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(output/'policy_quality.png',dpi=170);plt.close(fig)
    print('POLICY_EVALUATION='+str(output),flush=True)


if __name__=='__main__':
    main()
