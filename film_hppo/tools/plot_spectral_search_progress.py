"""Compare native HPPO training curves against actual environment steps."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True,layout='constrained')
    for batch in plan['batches']:
        for entry in batch['manifest']['runs']:
            log=Path(entry['run_dir'])/'log/serial'
            if not log.exists():continue
            acc=EventAccumulator(str(log),size_guidance={'scalars':0});acc.Reload()
            tags=acc.Tags()['scalars']
            step_tag='collector_iter/total_envstep_count'
            if step_tag not in tags:continue
            counts=acc.Scalars(step_tag)
            if not counts:continue
            times=[e.wall_time for e in counts];steps=[e.value for e in counts]
            offset=batch.get('source_run_env_steps',0)
            label=f"{batch['label']}/seed{entry['seed']}"
            for ax,tag in zip(axes,('evaluator_iter/reward_mean','collector_iter/avg_envstep_per_episode')):
                if tag not in tags:continue
                events=acc.Scalars(tag)
                x=np.interp([e.wall_time for e in events],times,steps,left=0)+offset
                ax.plot(x,[e.value for e in events],label=label,alpha=.85)
    axes[0].set(ylabel='Mean evaluation score',title='Native HPPO; training grid: 18 angles, 271 wavelengths',ylim=(.4,.85))
    axes[1].set(ylabel='Mean collection episode steps',xlabel='Environment steps (continuation includes parent budget)',ylim=(0,82))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(root/'training_comparison.png',dpi=170);plt.close(fig)
    print(root/'training_comparison.png')


if __name__=='__main__':main()
