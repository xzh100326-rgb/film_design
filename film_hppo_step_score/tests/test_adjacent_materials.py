"""Action-mask consistency and physical-design validation; no training."""

from copy import deepcopy
import unittest

import numpy as np
import torch
from ding.config import compile_config

from film_rl.config import load_config, validate
from film_rl.env import ThinFilmEnv
from film_rl.hppo import build_configs, build_model
from film_rl.optics import FilmOptics
from film_rl.policy import FilmPPOPolicy


class AdjacentMaterialTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.cfg.update(angles_deg=[0.], min_layers=2, forbid_adjacent_same_material=True,
                        max_total_thickness_nm=None)

    @staticmethod
    def action(material):
        return {"action_type": material, "action_args": np.zeros(5, dtype=np.float32)}

    def test_repeated_material_is_rejected_without_changing_state(self):
        env = ThinFilmEnv(self.cfg)
        env.reset()
        first = env.step(self.action(3))
        previous_layers = list(env.layers)
        with self.assertRaisesRegex(ValueError, "Adjacent coating"):
            env.step(self.action(3))
        self.assertEqual(env.layers, previous_layers)
        np.testing.assert_array_equal(env._observation(), first.obs)
        self.assertFalse(env._done)
        self.assertEqual(env.optics.evaluation_count, 1)
        # A-B-A is allowed: only consecutive identical materials are forbidden.
        env.step(self.action(4))
        env.step(self.action(3))
        end = env.step({"action_type": 5})
        self.assertTrue(end.done)
        self.assertEqual([i for i, _ in env.layers], [3, 4, 3])

    def test_mask_uses_last_material_per_sample_in_all_actor_forwards(self):
        env = ThinFilmEnv(self.cfg)
        observations = [env.reset()]
        for material in (3, 4, 3):
            observations.append(env.step(self.action(material)).obs)
        batch = torch.as_tensor(np.stack(observations))
        original = batch.clone()
        model = build_model(self.cfg)
        for mode in ("compute_actor", "compute_actor_critic"):
            with torch.no_grad():
                logits = model(batch, mode)["logit"]["action_type"]
            probabilities = logits.softmax(dim=-1)
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue((probabilities[0, :5] > 0).all())
            self.assertEqual(float(probabilities[0, 5]), 0.)
            self.assertEqual(float(probabilities[1, 5]), 0.)
            for row, previous in ((1, 3), (2, 4), (3, 3)):
                self.assertEqual(float(probabilities[row, previous]), 0.)
                allowed = [i for i in range(5) if i != previous]
                self.assertTrue((probabilities[row, allowed] > 0).all())
            self.assertGreater(float(probabilities[2, 5]), 0.)
            self.assertGreater(float(probabilities[3, 5]), 0.)
        torch.testing.assert_close(batch, original)

    def test_real_collect_and_eval_modes_generate_legal_sequences(self):
        main, create = build_configs(self.cfg)
        main.policy.cuda = False
        create.policy.type += "_command"
        compiled = compile_config(main, seed=0, auto=True, create_cfg=create, save_cfg=False)
        model = build_model(self.cfg, main)
        policy = FilmPPOPolicy(compiled.policy, model=model, enable_field=["learn", "collect", "eval"])
        before = {k: v.clone() for k, v in model.state_dict().items()}
        for mode in (policy.collect_mode, policy.eval_mode):
            env = ThinFilmEnv(self.cfg)
            try:
                for _ in range(3):
                    obs = env.reset()
                    for _ in range(self.cfg["max_layers"] + 1):
                        output = mode.forward({0: torch.as_tensor(obs)})[0]
                        action = {k: v.detach().cpu().numpy() for k, v in output["action"].items()}
                        step = env.step(action)
                        self.assertTrue(all(a[0] != b[0] for a, b in zip(env.layers, env.layers[1:])))
                        obs = step.obs
                        if step.done:
                            break
                    else:
                        self.fail("Policy episode did not terminate")
                    mode.reset([0])
            finally:
                env.close()
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_optics_rejects_adjacent_duplicates_before_solver(self):
        optics = FilmOptics(self.cfg)
        thickness = (self.cfg["thickness_min_nm"] + self.cfg["thickness_max_nm"]) / 2
        with self.assertRaisesRegex(ValueError, "Adjacent coating"):
            optics.evaluate([(3, thickness), (3, thickness)])
        self.assertEqual(optics.evaluation_count, 0)
        report = optics.evaluate([(3, thickness), (4, thickness)])
        self.assertTrue(report["forbid_adjacent_same_material"])

    def test_disabled_option_preserves_historical_evaluation(self):
        cfg = deepcopy(self.cfg)
        cfg["forbid_adjacent_same_material"] = False
        env = ThinFilmEnv(cfg)
        env.reset()
        obs = env.step(self.action(3)).obs
        model = build_model(cfg)
        with torch.no_grad():
            logits = model(torch.as_tensor(obs).unsqueeze(0), "compute_actor")["logit"]["action_type"]
        self.assertGreater(float(logits[0, 3]), -1e8)
        env.step(self.action(3))
        end = env.step({"action_type": 5})
        self.assertTrue(end.done)
        self.assertFalse(end.info["forbid_adjacent_same_material"])

    def test_invalid_flag_is_rejected(self):
        for value in ("true", 1, None):
            cfg = deepcopy(self.cfg)
            cfg["forbid_adjacent_same_material"] = value
            with self.assertRaisesRegex(ValueError, "must be boolean"):
                validate(cfg)


if __name__ == "__main__":
    unittest.main()
