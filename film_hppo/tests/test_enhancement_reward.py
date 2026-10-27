"""Opt-in reward integration, selection, stopping and refinement regressions."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from fixtures import yaml_config, write_substrate
from film_rl.optics import FilmOptics
from film_rl.enhancement import enhancement, hemisphere_quadrature
from film_rl.config import validate
from film_rl.design import compact_design, design_key, search_target_reached
from film_rl.env import ThinFilmEnv
from film_rl.bounded_training import evaluation_status
from film_rl.improve import ThicknessObjective, refine, problem_fingerprint


class EnhancementRewardTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.cfg=yaml_config(self.root,optical_constants=[('Low',1.4,0.),('High',2.1,0.)])
        write_substrate(self.root,n=1.)
        self.cfg.update(wavelengths_nm={'start':400.,'stop':2000.,'num':81},angles_deg=[0.,30.,60.],
                        reward_alpha=0., reward_objective='enhancement', enhancement_reward={'angle_order':12})
        self.layers=[(0,40.),(1,60.)]

    def test_default_is_legacy_and_mode_changes_fingerprint(self):
        legacy=deepcopy(self.cfg);legacy.pop('reward_objective');legacy.pop('enhancement_reward')
        r=FilmOptics(legacy).evaluate(self.layers)
        self.assertEqual(r['reward'],r['score'])
        explicit=deepcopy(legacy);explicit['reward_objective']='spectral'
        self.assertEqual(FilmOptics(explicit).evaluate(self.layers)['reward'],r['reward'])
        self.assertNotEqual(problem_fingerprint(legacy),problem_fingerprint(self.cfg))

    def test_matches_standalone_hemisphere_and_keeps_spectral_score(self):
        r=FilmOptics(self.cfg).evaluate(self.layers)
        legacy=deepcopy(self.cfg);legacy['reward_objective']='spectral'
        original=FilmOptics(legacy).evaluate(self.layers)
        np.testing.assert_allclose(r['R'],original['R'],atol=1e-13)
        self.assertAlmostEqual(r['score'],original['score'],places=13)
        angles,weights=hemisphere_quadrature(12)
        legacy['angles_deg']=angles.tolist()
        thermal=FilmOptics(legacy).evaluate(self.layers)
        expected=enhancement(thermal['wavelengths_nm'],weights@np.array(thermal['R']),view_factor=.95)
        self.assertAlmostEqual(r['reward'],expected['chi'],places=12)
        self.assertEqual(compact_design(r)['enhancement_factor'],r['reward'])
        self.assertGreater(ThinFilmEnv(self.cfg).reward_space.high[0],1.)

    def test_penalty_subtracted_once_and_stopping_uses_new_metric(self):
        self.cfg['spectral_targets']=[{'quantity':'T','target':1.,'bounds':{'min':1.,'penalty_weight':10.}}]
        r=FilmOptics(self.cfg).evaluate(self.layers)
        self.assertGreater(r['constraint_penalty'],0.)
        self.assertAlmostEqual(r['reward'],r['enhancement_factor']-r['constraint_penalty'])
        self.assertFalse(search_target_reached(r,.001))
        self.assertEqual(design_key(r),(0,r['reward']))
        info={'constraints_satisfied':[1,1],'design_score':[.1,.1],
              'training_reward':[5.,5.],'stopping_metric':[5.,5.]}
        self.assertTrue(evaluation_status(info,4.,2)[1])
        self.assertFalse(evaluation_status(info,6.,2)[1])
        clean=FilmOptics({**self.cfg,'spectral_targets':[{'quantity':'T','target':1.}]}).evaluate(self.layers)
        self.assertTrue(search_target_reached(clean,clean['reward']-.01))
        self.assertEqual(design_key(clean),(1,clean['reward']))

    def test_invalid_mode_grid_geometry_and_settings(self):
        for update in [dict(reward_objective='bad'),dict(wavelengths_nm=[500.,600.,700.]),
                       dict(polarization='s'),dict(enhancement_reward={'view_factor':2.}),
                       dict(enhancement_reward={'angle_order':True}),dict(design_selection='lamp_pareto')]:
            with self.subTest(update=update), self.assertRaises(ValueError):validate({**self.cfg,**update})
        write_substrate(self.root,n=1.5)
        with self.assertRaisesRegex(ValueError,'air on both sides'):FilmOptics(self.cfg)

    def test_environment_return_and_numerical_refinement(self):
        env=ThinFilmEnv(self.cfg);env.reset()
        env.step({'action_type':0,'action_args':np.zeros(2)})
        last=env.step({'action_type':2,'action_args':np.zeros(2)})
        self.assertAlmostEqual(float(last.reward[0]),last.info['reward'],places=6)
        self.assertEqual(last.info['eval_episode_return'],last.info['reward'])
        self.assertEqual(last.info['episode_info']['stopping_metric'],last.info['reward'])
        env.close()
        optics=FilmOptics(self.cfg)
        obj=ThicknessObjective(optics,[0,1]);value,grad=obj([40.,60.])
        self.assertAlmostEqual(value,-optics.evaluate(self.layers)['reward'],places=12)
        h=.01
        expected=(-optics.evaluate([(0,40.+h),(1,60.)])['reward']+optics.evaluate([(0,40.-h),(1,60.)])['reward'])/(2*h)
        self.assertAlmostEqual(grad[0],expected,delta=1e-6)
        before=optics.evaluate(self.layers)
        after=refine(optics,self.layers,max_iterations=1)
        self.assertGreaterEqual(design_key(after),design_key(before))
        self.assertIn('finite-difference',after['optimization']['method'])

if __name__=='__main__':unittest.main()
