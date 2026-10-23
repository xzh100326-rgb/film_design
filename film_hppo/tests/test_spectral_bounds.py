"""Optional bound penalties, gradient agreement and end-to-end PPO compatibility."""

from copy import deepcopy
import io
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fixtures import yaml_config
from film_rl.config import PROJECT_DIR
from film_rl.design import compact_design, design_key, search_target_reached
from film_rl.env import ThinFilmEnv
from film_rl.improve import ThicknessObjective, discover_seeds, improve, refine
from film_rl.optics import FilmOptics
from film_rl.targets import SpectralObjective


class SpectralBoundsTests(unittest.TestCase):
    def setUp(self):
        scratch = PROJECT_DIR/'outputs'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='bounds_test_', dir=scratch)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(scratch.resolve())
        self.addCleanup(self.temp.cleanup)
        self.cfg = yaml_config(self.root, optical_constants=[('clear', 1.4, 0.), ('lossy', 2.1, .2)])
        self.cfg.update(wavelengths_nm=[400., 600., 1000.], angles_deg=[0., 35.], reward_alpha=0.,
                        spectral_targets=[{'quantity': 'T', 'target': 1.}])
        torch.set_num_threads(1)

    def objective(self, targets):
        return SpectralObjective({'spectral_targets': targets}, [400., 600., 1000.], [0., 35.])

    def test_manual_lower_and_upper_penalties_are_not_normalized_by_target_weights(self):
        objective = self.objective([
            {'quantity': 'T', 'target': 1., 'weight': 9., 'bounds': {'min': .7, 'penalty_weight': 10.}},
            {'quantity': 'R', 'target': 0., 'weight': 1., 'bounds': {'max': .3, 'penalty_weight': 2.}}])
        result = objective.evaluate_spectra({'T': np.full((2, 3), .2), 'R': np.full((2, 3), .6)})
        self.assertAlmostEqual(result['score'], 1-(.9*.8+.1*.6))
        self.assertAlmostEqual(result['constraint_penalty'], 10*.5**2+2*.3**2)
        self.assertFalse(result['constraints_satisfied'])
        self.assertEqual(result['evaluation_metric'], 'penalized_reward')

    def test_pointwise_differs_from_mean_and_does_not_scale_with_sample_count(self):
        target = {'quantity': 'T', 'target': 1., 'bounds': {'min': .6, 'penalty_weight': 10.}}
        actual = np.array([[.4, .8, .6], [.4, .8, .6]])
        mean = self.objective([target]).evaluate_spectra({'T': actual})
        self.assertAlmostEqual(mean['constraint_penalty'], 0.)
        self.assertTrue(mean['constraints_satisfied'])
        target['bounds']['scope'] = 'pointwise'
        point = self.objective([target]).evaluate_spectra({'T': actual})
        self.assertAlmostEqual(point['constraint_penalty'], 10*.2**2/3)
        self.assertFalse(point['constraints_satisfied'])
        repeated = SpectralObjective({'spectral_targets': [target]}, [400., 600., 1000.], [0., 10., 20., 30.])
        self.assertEqual(point['constraint_penalty'], repeated.evaluate_spectra({'T': np.tile(actual, (2, 1))})['constraint_penalty'])

    def test_varying_pointwise_bounds_and_torch_derivative(self):
        target = {'quantity': 'T', 'target': 1., 'bounds': {
            'min': [.6, .7, .8], 'scope': 'pointwise', 'penalty_weight': 3.}}
        objective = self.objective([target])
        values = np.array([[.4, .8, .5], [.4, .8, .5]])
        result = objective.evaluate_spectra({'T': values})
        tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        loss = objective.torch_loss({'T': tensor}, tensor, {})
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), result['mae']+result['constraint_penalty'])
        expected = (-np.ones_like(values)-6*np.maximum(np.array([.6, .7, .8])-values, 0))/values.size
        np.testing.assert_allclose(tensor.grad.numpy(), expected, atol=1e-14)

    def test_invalid_bounds_are_rejected_before_running_optics(self):
        invalid = [{'min': .9, 'max': .8}, {'min': float('nan')}, {'max': 1.1}, {'min': True},
                   {'penalty_weight': -1., 'min': .2}, {'min': .2, 'scope': 'wrong'},
                   {'min': [.1, .2]}, {'max': .5}, {'minimum': .2}]
        for limits in invalid:
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                self.objective([{'quantity': 'T', 'target': 1., 'bounds': limits}])

    def test_missing_or_empty_limits_preserve_legacy_reward_and_metric(self):
        for alpha in [0., .5, 1.]:
            cfg = deepcopy(self.cfg)
            cfg['reward_alpha'] = alpha
            before = FilmOptics(cfg).evaluate([(1, 55.)])
            log_reward = -math.log10(max(before['mae'], 1e-6))/6.
            self.assertEqual(before['reward'], (1-alpha)*before['score']+alpha*log_reward)
            self.assertNotIn('has_constraints', before)
            for limits in [{}, {'min': None, 'max': None}]:
                cfg['spectral_targets'][0]['bounds'] = limits
                after = FilmOptics(cfg).evaluate([(1, 55.)])
                self.assertEqual(before['reward'], after['reward'])
                self.assertEqual(before['target_errors'], after['target_errors'])
                self.assertEqual(before['evaluation_metric'], after['evaluation_metric'])

    def test_negative_reward_reaches_environment_and_compact_exports(self):
        cfg = deepcopy(self.cfg)
        cfg['spectral_targets'][0]['bounds'] = {'min': 1., 'penalty_weight': 1000.}
        env = ThinFilmEnv(cfg)
        env.reset()
        env.step({'action_type': 1, 'action_args': np.zeros(2)})
        step = env.step({'action_type': env.EOS, 'action_args': np.zeros(2)})
        self.assertLess(float(step.reward[0]), 0.)
        self.assertTrue(env.reward_space.contains(step.reward))
        self.assertEqual(step.info['eval_episode_return'], step.info['reward'])
        self.assertAlmostEqual(step.info['reward'], step.info['base_reward']-step.info['constraint_penalty'])
        self.assertEqual(step.info['episode_info']['constraints_satisfied'], 0)
        exported = compact_design(step.info)
        self.assertEqual(exported['constraint_violations'], step.info['constraint_violations'])
        self.assertIn('bounds', exported['target_errors'][0])
        self.assertEqual(compact_design(exported), exported)
        env.close()

    def test_bounded_gradient_matches_reward_and_finite_differences(self):
        for scope in ['band_mean', 'pointwise']:
            for alpha in [0., .5, 1.]:
                with self.subTest(scope=scope, alpha=alpha):
                    cfg = deepcopy(self.cfg)
                    cfg.update(reward_alpha=alpha, substrate_thickness_nm=200., exit_n=1.)
                    cfg['spectral_targets'] = [
                        {'quantity': 'T', 'target': 1., 'bounds': {'min': .9, 'scope': scope, 'penalty_weight': 4.}},
                        {'quantity': 'R', 'target': 0., 'bounds': {'max': .05, 'scope': scope, 'penalty_weight': 3.}}]
                    optics = FilmOptics(cfg)
                    objective = ThicknessObjective(optics, [0, 1])
                    values = np.array([40., 55.])
                    error, gradient = objective(values)
                    self.assertAlmostEqual(error, 1-optics.evaluate(list(zip([0, 1], values)))['reward'], places=12)
                    differences = []
                    for position in range(2):
                        delta = np.zeros(2)
                        delta[position] = 1e-3
                        plus = optics.evaluate(list(zip([0, 1], values+delta)))['reward']
                        minus = optics.evaluate(list(zip([0, 1], values-delta)))['reward']
                        differences.append(-(plus-minus)/2e-3)
                    np.testing.assert_allclose(gradient, differences, atol=1e-8, rtol=1e-5)

    def test_selection_and_refinement_do_not_discard_feasibility_for_higher_raw_score(self):
        good = {'score': .7, 'reward': .7, 'has_constraints': True, 'constraints_satisfied': True}
        bad = {'score': .95, 'reward': .8, 'has_constraints': True, 'constraints_satisfied': False}
        self.assertGreater(design_key(good), design_key(bad))
        self.assertFalse(search_target_reached(bad, .9))
        self.assertTrue(search_target_reached(good, .6))
        optics = FilmOptics(self.cfg)
        with patch.object(optics, 'evaluate', side_effect=[good.copy(), bad.copy()]), \
                patch('scipy.optimize.minimize', return_value=SimpleNamespace(x=np.array([.5]), success=True, message='mock')):
            kept = refine(optics, [(0, 40.)])
        self.assertTrue(kept['constraints_satisfied'])

    def test_directory_recomputes_bounds_and_prefers_feasible_design(self):
        cfg = deepcopy(self.cfg)
        optics = FilmOptics(cfg)
        designs = [optics.evaluate([(0, 40.)]), optics.evaluate([(1, 55.)])]
        low, high = sorted(designs, key=lambda r: r['mean_transmission'])
        lower = (low['mean_transmission']+high['mean_transmission'])/2
        cfg['spectral_targets'][0]['bounds'] = {'min': lower}
        directory = self.root/'designs'
        directory.mkdir()
        for index, report in enumerate(designs):
            report['score'] = 100-index  # Saved rankings are deliberately meaningless.
            (directory/f'candidate{index}.json').write_text(json.dumps(compact_design(report)), encoding='utf-8')
        found = discover_seeds(cfg, FilmOptics(cfg), design_dir=directory, limit=2)
        self.assertTrue(found[0]['constraints_satisfied'])
        self.assertFalse(found[1]['constraints_satisfied'])

    def test_early_stop_requires_every_evaluation_episode_to_pass(self):
        from film_rl.bounded_training import evaluation_status
        info = {'constraints_satisfied': [1, 0], 'design_score': [.99, .99], 'training_reward': [.9, .9]}
        self.assertFalse(evaluation_status(info, .95, 2)[1])
        info['constraints_satisfied'] = [1, 1]
        self.assertTrue(evaluation_status(info, .95, 2)[1])
        self.assertFalse(evaluation_status(info, .95, 3)[1])
        info['design_score'] = [.8, .8]
        self.assertFalse(evaluation_status(info, .95, 2)[1])

    def test_bounded_improve_does_not_report_success_for_violating_high_score(self):
        cfg = deepcopy(self.cfg)
        cfg['spectral_targets'][0]['bounds'] = {'min': 1., 'penalty_weight': 10.}
        cfg['improvement'] = {'structure_rounds': 0, 'max_thickness_iterations': 1,
                              'warm_start_steps': 0, 'target_score': .001}
        source = self.root/'start.json'
        source.write_text(json.dumps(compact_design(FilmOptics(cfg).evaluate([(1, 55.)]))), encoding='utf-8')
        with patch('sys.stdout', new_callable=io.StringIO):
            summary = improve(cfg, self.root/'bounded_improve', design=source)
        self.assertGreater(summary['best_score'], .001)
        self.assertFalse(summary['constraints_satisfied'])
        self.assertFalse(summary['search_target_reached'])
        self.assertLess(summary['best_reward'], summary['best_score'])

    def test_batch_export_prefers_feasible_lower_raw_score(self):
        from run_seed_batch import select_saved_design
        folder = self.root/'batch_run/designs'
        folder.mkdir(parents=True)
        for name, score, reward, passed in [('bad', .99, .9, False), ('good', .7, .7, True)]:
            data = {'layers': [], 'score': score, 'reward': reward,
                    'has_constraints': True, 'constraints_satisfied': passed}
            (folder/f'best_{name}.json').write_text(json.dumps(data), encoding='utf-8')
        path, _, best, _ = select_saved_design(folder.parent)
        self.assertEqual(path.name, 'best_good.json')
        self.assertTrue(best['constraints_satisfied'])

    def test_bounded_training_can_stop_when_all_episodes_pass(self):
        cfg = deepcopy(self.cfg)
        cfg['spectral_targets'][0]['bounds'] = {'min': 0., 'max': 1.}
        cfg['max_layers'] = 2
        run = self.root/'passing_run'
        cfg['training'].update(exp_name=run.relative_to(PROJECT_DIR).as_posix(),
                               collector_env_num=1, evaluator_env_num=1, n_evaluator_episode=2,
                               n_sample=8, batch_size=8, epoch_per_collect=1, max_env_step=128,
                               max_train_iter=100, eval_freq=1, gae_lambda=1., stop_value=0.)
        config = self.root/'passing_config.json'
        config.write_text(json.dumps(cfg), encoding='utf-8')
        completed = subprocess.run([sys.executable, '-m', 'film_rl', 'train', '--config', str(config), '--non-interactive'],
                                   cwd=PROJECT_DIR, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stdout[-2000:]+completed.stderr[-3000:])
        with (run/'result.pkl').open('rb') as stream:
            result = pickle.load(stream)
        self.assertTrue(result['stop'])
        self.assertEqual(result['constraints_satisfied'], [1, 1])

    def test_native_bounded_training_smoke_runs_and_saves_finite_results(self):
        cfg = deepcopy(self.cfg)
        cfg['spectral_targets'][0]['bounds'] = {'min': .99, 'penalty_weight': 1000.}
        cfg['max_layers'] = 2
        run = self.root/'smoke_run'
        cfg['training'].update(exp_name=run.relative_to(PROJECT_DIR).as_posix(),
                               collector_env_num=1, evaluator_env_num=1, n_evaluator_episode=2,
                               n_sample=8, batch_size=8, epoch_per_collect=1, max_env_step=8,
                               max_train_iter=1, eval_freq=1, gae_lambda=1.)
        config = self.root/'smoke_config.json'
        config.write_text(json.dumps(cfg), encoding='utf-8')
        completed = subprocess.run([sys.executable, '-m', 'film_rl', 'train', '--config', str(config), '--non-interactive'],
                                   cwd=PROJECT_DIR, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stdout[-2000:]+completed.stderr[-3000:])
        with (run/'result.pkl').open('rb') as stream:
            result = pickle.load(stream)
        self.assertFalse(result['stop'])
        self.assertGreaterEqual(result['train_iter'], 1)
        self.assertTrue(np.isfinite(result['eval_value']))
        self.assertEqual(result['evaluation_metric'], 'penalized_reward')
        for path in (run/'designs').glob('best_*.json'):
            report = json.loads(path.read_text(encoding='utf-8'))
            self.assertIn('constraint_penalty', report)


if __name__ == '__main__':
    unittest.main()
