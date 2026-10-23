"""Wait for the authorized hybrid initialization, train PPO and check generation."""
from pathlib import Path
import os
import subprocess
import sys
import time
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
import numpy as np
import torch
from film_rl.config import load_config
from film_rl.hppo import build_model
from film_rl.env import ThinFilmEnv
from film_rl.design import compact_design
from film_rl.improve import problem_fingerprint
from tools.optimize_lamp_hppo import read,write


def main():
    torch.set_num_threads(1)
    root=PROJECT/'outputs'/sys.argv[1]
    folder=root/(sys.argv[2] if len(sys.argv)>2 else 'production')
    deadline=time.monotonic()+3600
    while not (root/'trials/physical_refinement/complete.json').is_file():
        if time.monotonic()>deadline:raise TimeoutError('No completed refinement within one hour')
        time.sleep(5)
    cfg=load_config(folder/'training_config.json')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    with (folder/'training.log').open('w',encoding='utf-8') as stream:
        process=subprocess.Popen([sys.executable,'-B','-m','film_rl','train','--config',
            str(folder/'training_config.json'),'--non-interactive'],cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        write(folder/'status.json',{'status':'training','pid':process.pid})
        code=process.wait()
    if code:
        write(folder/'status.json',{'status':'failed','exit_code':code})
        raise RuntimeError('Production training failed')
    run=PROJECT/cfg['training']['exp_name']
    checkpoints=list((run/'ckpt').glob('iteration_*.pth.tar'))
    checkpoint=max(checkpoints,key=lambda p:int(p.name.split('_')[1].split('.')[0]))
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
    model=build_model(cfg);model.load_state_dict(state,strict=True);model.eval()
    torch.save({'model':state,'problem_fingerprint':problem_fingerprint(cfg),
                'method':'Final PPO after HPPO-seeded physical thickness refinement and supervised initialization'},folder/'final_model.pt')
    cfg.pop('result_dir',None)
    generator=torch.Generator().manual_seed(71023);records=[]
    for episode in range(33):
        environment=ThinFilmEnv(cfg);obs=environment.reset()
        with torch.no_grad():
            for _ in range(cfg['max_layers']+1):
                logits=model(torch.as_tensor(obs)[None],mode='compute_actor')['logit']
                if episode==0:
                    kind=int(logits['action_type'].argmax(-1).item());args=logits['action_args']['mu'][0]
                else:
                    kind=int(torch.multinomial(logits['action_type'].softmax(-1),1,generator=generator).item())
                    args=logits['action_args']['mu'][0]+logits['action_args']['sigma'][0]*torch.randn(environment.material_count,generator=generator)
                timestep=environment.step({'action_type':kind,'action_args':args.numpy()});obs=timestep.obs
                if timestep.done:break
        data=compact_design(timestep.info)
        if episode==0:write(folder/'final_policy_deterministic_design.json',data)
        records.append(data['application_metrics']);environment.close()
    write(folder/'final_policy_generation.json',{'deterministic':records[0],
        'stochastic_metrics':records[1:], 'mean_visible_T':float(np.mean([x['visible_T'] for x in records[1:]])),
        'mean_IR_R':float(np.mean([x['IR_R'] for x in records[1:]])),
        'note':'Generated candidates participate in evaluation; not an independent held-out test.'})
    write(folder/'status.json',{'status':'complete','run':str(run),'checkpoint':str(checkpoint),
        'deterministic':records[0]})
    print('PRODUCTION_TRAINING_AND_GENERATION_COMPLETE',flush=True)


if __name__=='__main__':main()
