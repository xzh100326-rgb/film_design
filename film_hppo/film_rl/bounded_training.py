"""DI-engine's on-policy workflow with bounds-aware evaluation and stopping.

Only configurations with active spectral bounds use this entry point. Learner,
collector, commander and PPO losses remain native DI-engine components.
"""

from copy import deepcopy
from functools import partial
from pathlib import Path
import pickle
import time

import numpy as np
from tensorboardX import SummaryWriter
from ding.config import compile_config
from ding.envs import get_vec_env_setting, create_env_manager
from ding.policy import create_policy
from ding.utils import get_rank, set_pkg_seed
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_serial_collector


def evaluation_status(info, target, expected_count):
    """Fail closed: every evaluated episode must pass, not just the mean flag."""
    passed = np.asarray(info.get('constraints_satisfied', []), dtype=float).reshape(-1)
    scores = np.asarray(info.get('design_score', []), dtype=float).reshape(-1)
    if 'stopping_metric' in info:
        scores = np.asarray(info['stopping_metric'], dtype=float).reshape(-1)
    rewards = np.asarray(info.get('training_reward', []), dtype=float).reshape(-1)
    complete = len(passed) == len(scores) == len(rewards) == expected_count
    if not complete or not all(np.isfinite(x).all() for x in (passed, scores, rewards)):
        return False, False, (0, -float('inf'))
    feasible = bool(np.all(passed == 1.))
    return feasible, bool(feasible and scores.mean() >= target), (int(feasible), float(rewards.mean()))


class BoundsEvaluator(InteractionSerialEvaluator):
    def eval(self, save_ckpt_fn=None, train_iter=-1, envstep=-1, n_episode=None, **kwargs):
        target = self._stop_value
        # Native scalar stopping must not bypass the all-episode feasibility gate.
        self._stop_value = float('inf')
        try:
            _, info = super().eval(None, train_iter, envstep, n_episode=n_episode, **kwargs)
        finally:
            self._stop_value = target
        expected = self._default_n_episode if n_episode is None else n_episode
        feasible, reached, key = evaluation_status(info, target, expected)
        previous = getattr(self, '_best_bounds_key', None)
        if previous is None or key > previous:
            self._best_bounds_key = key
            if save_ckpt_fn is not None and get_rank() == 0:
                save_ckpt_fn('ckpt_best.pth.tar')
        stop = reached and train_iter > 0
        self._logger.info('Spectral bounds: all_episodes_satisfied=%s, target_reached=%s', feasible, stop)
        return stop, info


def serial_pipeline_bounded(input_cfg, seed=0, model=None, max_train_iter=int(1e10), max_env_step=int(1e10)):
    main, create = deepcopy(input_cfg)
    create.policy.type += '_command'
    cfg = compile_config(main, seed=seed, auto=True, create_cfg=create, save_cfg=True,
                         renew_dir=not main.policy.learn.get('resume_training', False))
    env_fn, collector_cfgs, evaluator_cfgs = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_cfgs])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_cfgs])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])
    tb = SummaryWriter(str(Path(cfg.exp_name)/'log/serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb, exp_name=cfg.exp_name)
    collector = create_serial_collector(cfg.policy.collect.collector, env=collector_env,
                                       policy=policy.collect_mode, tb_logger=tb, exp_name=cfg.exp_name)
    evaluator = BoundsEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb, exp_name=cfg.exp_name)
    commander = BaseSerialCommander(cfg.policy.other.commander, learner, collector, evaluator, None, policy.command_mode)
    stop, eval_info = False, {}
    try:
        learner.call_hook('before_run')
        if cfg.policy.learn.get('resume_training', False):
            collector.envstep = learner.collector_envstep
        while True:
            collect_kwargs = commander.step()
            if evaluator.should_eval(learner.train_iter):
                stop, eval_info = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break
            samples = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
            learner.train(samples, collector.envstep)
            if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
                break
        learner.call_hook('after_run')
        values = eval_info.get('eval_episode_return', [])
        with (Path(cfg.exp_name)/'result.pkl').open('wb') as stream:
            pickle.dump({'stop': stop, 'env_step': collector.envstep, 'train_iter': learner.train_iter,
                         'eval_value': float(np.mean(values)) if len(values) else None,
                         'eval_value_raw': values, 'finish_time': time.ctime(),
                         'evaluation_metric': ('enhancement_minus_penalty' if cfg.env.get('reward_objective') == 'enhancement'
                                               else 'penalized_reward'),
                         'constraints_satisfied': eval_info.get('constraints_satisfied', [])}, stream)
        return policy
    finally:
        collector.close()
        evaluator.close()
        if tb is not None:
            tb.close()
