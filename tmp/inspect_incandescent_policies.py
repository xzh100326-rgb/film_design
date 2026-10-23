"""Read local, self-generated checkpoints without training or modifying them."""
import json
from pathlib import Path
import re
import sys
import numpy as np
import torch

project=Path('E:/lithography_machine_lens/film_hppo')
sys.path.insert(0,str(project))
from film_rl.config import load_config
from film_rl.env import ThinFilmEnv
from film_rl.hppo import build_configs, build_model

torch.set_num_threads(1)
root=Path(sys.argv[1]).resolve()
plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
rows=[]
for batch in plan['batches']:
    for entry in batch['manifest']['runs']:
        state_path=Path(entry['record'])
        if not state_path.exists() or json.loads(state_path.read_text(encoding='utf-8'))['status']!='complete':
            continue
        ckpts=list((Path(entry['run_dir'])/'ckpt').glob('iteration_*.pth.tar'))
        if not ckpts:
            continue
        checkpoint=max(ckpts,key=lambda p:int(re.search(r'iteration_(\d+)',p.name)[1]))
        cfg=load_config(entry['config'])
        main,_=build_configs(cfg)
        model=build_model(cfg,main)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(payload['model'],strict=True)
        model.eval()
        env=ThinFilmEnv(cfg)
        samples=[]
        for layers in [[],[(3,111.7)]]:
            env.layers=layers
            with torch.no_grad():
                output=model(torch.from_numpy(env._observation()).unsqueeze(0),'compute_actor')
            logit=output['logit']['action_type'][0]
            p=logit.softmax(-1)
            top=int(p.argmax())
            mu=output['logit']['action_args']['mu'][0]
            sigma=output['logit']['action_args']['sigma'][0]
            mapped=lambda values:(cfg['thickness_min_nm']+(torch.tanh(values)+1)/2*(cfg['thickness_max_nm']-cfg['thickness_min_nm'])).tolist()
            samples.append(dict(layers=layers, probabilities=p.tolist(),EOS_probability=float(p[-1]),
                                categorical_entropy=float(torch.distributions.Categorical(logits=logit).entropy()),
                                mu_latent=mu.tolist(),sigma_latent=sigma.tolist(),
                                mapped_mu_nm=mapped(mu), q15p9_thickness_nm=mapped(mu-sigma),
                                q84p1_thickness_nm=mapped(mu+sigma)))
        row=dict(group=batch.get('label','e'+str(batch['epochs'])),seed=entry['seed'],
                 checkpoint=str(checkpoint),states=samples)
        rows.append(row)
        print(row['group'],row['seed'],'first SiC probability',samples[0]['probabilities'][3],
              'EOS after SiC 111.7 nm',samples[1]['EOS_probability'],flush=True)
(root/'policy_diagnostics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
