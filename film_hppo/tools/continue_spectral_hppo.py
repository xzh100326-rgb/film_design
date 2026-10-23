"""Continue a COMPLETED pure-HPPO model, using the same physics and framework.

Restores VAC weights (actor and critic), not optimizer moments or RNG state.
This is model-only continuation, never imitation or design optimization.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from film_rl.config import load_config
from film_rl.improve import problem_fingerprint  # hash helper only; no improve/refine call
from run_seed_batch import prepare,run_one,refresh_summary


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--variant',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--budget',type=int,default=240000)
    args=parser.parse_args();root=args.root.resolve()
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    source_batch=next(b for b in plan['batches'] if b['label']==args.variant)
    entry=next(e for e in source_batch['manifest']['runs'] if e['seed']==args.seed)
    record=json.loads(Path(entry['record']).read_text(encoding='utf-8'))
    if record['status']!='complete':raise ValueError('Source training must finish before continuation')
    cfg=load_config(entry['config'])
    for path,expected in source_batch['manifest']['material_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError('Material changed')
    # Native learner saves its final iteration on completion; retain best for
    # evaluation, use latest training state to continue rather than pick a design.
    candidates=list((Path(entry['run_dir'])/'ckpt').glob('iteration_*.pth.tar'))
    checkpoint=max(candidates,key=lambda p:int(p.name.split('_')[1].split('.')[0]))
    checkpoint_iteration=int(checkpoint.name.split('_')[1].split('.')[0])
    if checkpoint_iteration < int(record['result']['train_iter'])-1:
        raise RuntimeError('Final learner checkpoint missing; refuse to continue an old/initial model')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    label=f'{args.variant}_continue_s{args.seed}'
    if any(b['label']==label for b in plan['batches']):raise FileExistsError('Continuation already exists')
    destination=root/'continuations'/label;destination.mkdir(parents=True,exist_ok=False)
    # This checkpoint was produced by this project's completed native learner.
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model_file=destination/'native_hppo_weights.pt'
    torch.save({'model':payload['model'],'problem_fingerprint':problem_fingerprint(cfg),
                'method':'Model-only continuation of a completed pure HPPO run; no imitation',
                'source_checkpoint':str(checkpoint),'source_sha256':digest},model_file)
    trial=deepcopy(cfg)
    trial['training'].update(initial_model=str(model_file),learning_rate=3e-5,entropy_weight=.0001,
                            epoch_per_collect=3,n_sample=6000,batch_size=256,max_env_step=args.budget,
                            max_train_iter=1000000,eval_freq=288)
    # Only training fields changed; enforce exactly the same optical problem.
    assert problem_fingerprint(trial)==problem_fingerprint(cfg)
    path=destination/'config.json';write(path,trial)
    batch=prepare(path,group=f'lamp_R_hppo_{label}',seeds=[args.seed])
    manifest=json.loads((batch/'manifest.json').read_text(encoding='utf-8'))
    item={'label':label,'batch':str(batch),'manifest':manifest,'kind':'model_only_continuation',
          'source_checkpoint':str(checkpoint),'source_sha256':digest,
          'source_run_env_steps':record['result']['env_step'],'optimizer_restored':False}
    plan['batches'].append(item);write(root/'plan.json',plan)
    started=time.perf_counter()
    for job in manifest['runs']:run_one(job,batch,manifest['runtime_environment'])
    refresh_summary(batch,manifest,1,started)
    print('CONTINUATION_COMPLETE='+label,flush=True)


if __name__=='__main__':main()
