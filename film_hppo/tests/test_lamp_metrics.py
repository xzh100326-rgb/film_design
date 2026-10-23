import unittest
from copy import deepcopy
import numpy as np
from film_rl.lamp_metrics import lamp_metrics, lamp_key, dominates


class LampMetricsTests(unittest.TestCase):
    def report(self, visible_t=.75, infrared_r=.90):
        wave = np.arange(300., 3001., 10.)
        visible = (wave >= 480) & (wave <= 700)
        r = np.full((2, len(wave)), infrared_r)
        t = 1-r
        r[:, visible] = 1-visible_t
        t[:, visible] = visible_t
        return dict(wavelengths_nm=wave, angles_deg=[0., 60.], R=r, T=t, A=1-r-t, score=.1)

    def test_high_visible_transmission_does_not_hide_bad_ir(self):
        good = lamp_metrics(self.report())
        bad = lamp_metrics(self.report(.95, .1))
        self.assertGreater(lamp_key(good), lamp_key(bad))
        self.assertTrue(good['application_targets_met'])
        self.assertFalse(bad['application_targets_met'])

    def test_pareto_retains_tradeoffs_and_ignores_score(self):
        a = self.report(); b = deepcopy(a); b['score'] = .99
        self.assertEqual(lamp_key(lamp_metrics(a)), lamp_key(lamp_metrics(b)))
        improved = lamp_metrics(self.report(.80, .92))
        self.assertTrue(dominates(improved, lamp_metrics(a)))
        tradeoff = lamp_metrics(self.report(.90, .80))
        self.assertFalse(dominates(improved, tradeoff))
        self.assertFalse(dominates(tradeoff, improved))

    def test_each_ir_band_is_measured(self):
        report = self.report(); report['R'][:, (report['wavelengths_nm'] >= 710) &
            (report['wavelengths_nm'] <= 900)] = .2
        metrics = lamp_metrics(report)
        self.assertAlmostEqual(metrics['IR_min_subband_R'], .2)
        self.assertFalse(metrics['application_targets_met'])

    def test_sigma_is_applied_after_native_ppo_initialization(self):
        import torch
        from ding.config import compile_config
        from fixtures import base_config
        from film_rl.hppo import build_configs, build_model
        from film_rl.policy import FilmPPOPolicy
        cfg = base_config('.')
        cfg['training']['initial_sigma'] = .04
        main, create = build_configs(cfg)
        create.policy.type += '_command'
        compiled = compile_config(main, seed=0, auto=True, create_cfg=create, save_cfg=False)
        policy = FilmPPOPolicy(compiled.policy, model=build_model(cfg, main), enable_field=['learn', 'collect'])
        for name, value in policy._model.state_dict().items():
            if name.endswith('log_sigma_param'):
                torch.testing.assert_close(value.exp(), torch.full_like(value, .04))

    def test_environment_saves_physical_improvement_even_when_score_drops(self):
        import json
        import tempfile
        from pathlib import Path
        from fixtures import base_config
        from film_rl.env import ThinFilmEnv
        scratch = Path(__file__).resolve().parents[1] / 'outputs'
        with tempfile.TemporaryDirectory(prefix='lamp_test_', dir=scratch) as folder:
            cfg = base_config(folder)
            cfg.update(design_selection='lamp_pareto', result_dir=folder)
            env = ThinFilmEnv(cfg)
            first = self.report(.70, .90)
            second = self.report(.80, .92)
            for report, score in [(first, .99), (second, .10)]:
                report.update(layers=[], target_errors=[], layer_count=0, score=score)
                env._save_best(report)
            saved = json.loads(next(Path(folder).glob('best_*.json')).read_text(encoding='utf-8'))
            self.assertEqual(saved['score'], .10)
            self.assertAlmostEqual(saved['application_metrics']['visible_T'], .80)
            self.assertEqual(len(env._pareto), 1)
