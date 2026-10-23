from copy import deepcopy
import json
from pathlib import Path

project=Path('E:/lithography_machine_lens/film_hppo')
root=project/'outputs/incandescent_improve/20261021_094617'
rows=json.loads((project/'outputs/incandescent_hppo_experiments/20261021_004517/validation/20261021_013033/comparison.json').read_text())['results']
base=json.loads((root/'config_search0.json').read_text())
cases=[]
for tag,name in [('medium12','e5_ir6_seed0'),('long45','old_R_objective_winner')]:
    item=next(r for r in rows if r['experiment']==name)
    cfg=deepcopy(base)
    cfg['improvement'].update(structure_rounds=0,max_thickness_iterations=100)
    cfg['_experiment_start_design']=item['source']
    path=root/f'config_{tag}_thickness.json'
    path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    cases.append(dict(case=tag,source=item['source'],config=str(path),layers=item['layers']))
(root/'long_start_cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
print(json.dumps(cases,indent=2))
