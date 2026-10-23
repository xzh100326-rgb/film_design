"""Measure sparse-angle quadrature error on independent random legal stacks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from film_rl.config import load_config
from film_rl.design import compact_design
from film_rl.optics import FilmOptics


def main():
    root=Path(sys.argv[1]).resolve()
    cfg=load_config(root/'evaluation_config.json')
    coarse=deepcopy(cfg);coarse['angles_deg']=list(range(2,90,5))
    dense_optics=FilmOptics(cfg);coarse_optics=FilmOptics(coarse)
    torch.set_num_threads(1)
    rng=np.random.default_rng(1729)
    rows=[]
    for count in [1,2,4,8,16,32,45]*3:
        layers=[]
        for _ in range(count):
            legal=[i for i in range(len(dense_optics.names)) if not layers or layers[-1][0]!=i]
            layers.append((int(rng.choice(legal)),float(rng.uniform(15,400))))
        dense=compact_design(dense_optics.evaluate(layers))
        sparse=compact_design(coarse_optics.evaluate(layers))
        rows.append(dict(layers=count, score_error=sparse['score']-dense['score'],
                         visible_T_error=sparse['target_errors'][1]['mean_T']-dense['target_errors'][1]['mean_T'],
                         IR_R_error=sparse['target_errors'][2]['mean_R']-dense['target_errors'][2]['mean_R']))
    summary=dict(samples=len(rows),rng_seed=1729,results=rows,
                 max_abs_score_error=max(abs(r['score_error']) for r in rows),
                 max_abs_visible_T_error=max(abs(r['visible_T_error']) for r in rows),
                 max_abs_IR_R_error=max(abs(r['IR_R_error']) for r in rows))
    (root/'angle_grid_random_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__=='__main__':
    main()
