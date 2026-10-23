"""Physical, slot-state, masking and best-so-far regression tests."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from film_rl.actions import decode_tokens, encode_tokens, observation_size
from film_rl.config import PROJECT_DIR, load_config, validate
from film_rl.env import ThinFilmEnv
from film_rl.hppo import build_configs, build_model
from film_rl.optics import FilmOptics

torch.set_num_threads(1)


class MaskedDesignTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.cfg.update(angles_deg=[0.], max_layers=6)

    @staticmethod
    def fill(env, positions, tokens, values=None):
        env.pending_positions = list(positions)
        return env.step({"action_type": encode_tokens(tokens, env.material_count, env.fill_max),
                         "action_args": np.zeros(env.fill_max) if values is None else np.asarray(values)})

    def test_joint_encoding_roundtrip(self):
        for code in range(216):
            self.assertEqual(encode_tokens(decode_tokens(code, 5, 3), 5, 3), code)
        for invalid in [-1, 216, 1.5, float("nan"), [1, 2]]:
            with self.assertRaises(ValueError):
                decode_tokens(invalid, 5, 3)

    def test_no_eos_skips_masks_and_preserves_spatial_order(self):
        env = ThinFilmEnv(self.cfg)
        observation = env.reset()
        self.assertEqual(observation.shape, (observation_size(self.cfg),))
        self.assertTrue(env.observation_space.contains(observation))
        self.assertTrue((env.tokens == env.MASK).all())
        step = self.fill(env, [1, 4], [3, 4], [-100, 100, 0])
        self.assertFalse(step.done)
        self.assertEqual(env.layers, [(3, 10.), (4, 140.)])
        self.assertEqual(step.info["active_positions"], [1, 4])
        self.assertTrue(env.observation_space.contains(step.obs))
        self.assertEqual(env.optics.evaluation_count, 1)
        expected = env.optics.evaluate([(3, 10.), (4, 140.)])
        np.testing.assert_allclose(step.info["T"], expected["T"])

    def test_eos_waits_for_prefix_holes_and_ignores_suffix(self):
        env = ThinFilmEnv(self.cfg)
        env.reset()
        step = self.fill(env, [2, 3, 5], [4, env.EOS, 2])
        self.assertFalse(step.done)
        self.assertEqual(step.info["active_positions"], [2])
        self.assertEqual(step.info["eos_position"], 3)
        self.assertEqual(env.layers, [(4, 75.)])
        self.assertTrue(all(p < 3 for p in env.pending_positions))
        step = self.fill(env, [0, 1], [3, 0])
        self.assertTrue(step.done)
        self.assertEqual(env.layers, [(3, 75.), (0, 75.), (4, 75.)])
        self.assertEqual(step.info["termination_reason"], "eos_prefix_complete")
        self.assertEqual(step.info["slots"][5]["material_id"], 2)
        self.assertEqual(step.info["eval_episode_return"], step.info["reward"])
        self.assertAlmostEqual(step.info["training_return"], step.info["reward"] + env.optics.evaluate([(4, 75.)])["reward"])
        with self.assertRaises(RuntimeError):
            env.step(env.random_action())

    def test_new_earlier_eos_moves_boundary(self):
        env = ThinFilmEnv(self.cfg)
        env.reset()
        self.fill(env, [3, 5], [3, env.EOS])
        step = self.fill(env, [1], [env.EOS])
        self.assertEqual(step.info["eos_position"], 1)
        self.assertEqual(env.layers, [])
        self.assertFalse(step.done)
        step = self.fill(env, [0], [4])
        self.assertTrue(step.done)
        self.assertEqual(env.layers, [(4, 75.)])

    def test_full_twenty_slots_without_eos_terminates(self):
        cfg = deepcopy(self.cfg)
        cfg["max_layers"] = 20
        env = ThinFilmEnv(cfg)
        env.reset()
        for index in range(20):
            step = self.fill(env, [index], [index % 5])
            self.assertEqual(step.done, index == 19)
        self.assertEqual(step.info["layer_count"], 20)
        self.assertEqual(step.info["termination_reason"], "all_slots_filled")
        self.assertEqual(env.optics.evaluation_count, 20)

    def test_multiple_eos_uses_first_and_allows_bare_substrate(self):
        cfg = deepcopy(self.cfg)
        cfg["min_layers"] = 0
        env = ThinFilmEnv(cfg)
        env.reset()
        step = self.fill(env, [0, 2, 4], [env.EOS, 3, env.EOS])
        self.assertTrue(step.done)
        self.assertEqual(env.layers, [])
        self.assertEqual(step.info["eos_position"], 0)
        self.assertEqual(env.optics.evaluation_count, 1)

    def test_invalid_batch_is_rejected_before_mutation(self):
        env = ThinFilmEnv(self.cfg)
        env.reset()
        env.pending_positions = [0]
        bad = [([env.EOS], [0, 0, 0]), ([1, 2], [0, 0, 0]), ([3], [0]), ([3], [0, np.nan, 0])]
        for tokens, args in bad:
            with self.assertRaises(ValueError):
                env.step({"action_type": encode_tokens(tokens, 5, 3), "action_args": args})
            self.assertTrue((env.tokens == env.MASK).all())
            self.assertEqual(env.optics.evaluation_count, 0)

    def test_random_selection_reproducible_unique_and_only_unfilled_prefix(self):
        a, b = ThinFilmEnv(self.cfg), ThinFilmEnv(self.cfg)
        a.seed(31, dynamic_seed=False)
        b.seed(31, dynamic_seed=False)
        np.testing.assert_array_equal(a.reset(), b.reset())
        for _ in range(6):
            self.assertEqual(a.pending_positions, b.pending_positions)
            self.assertEqual(len(a.pending_positions), len(set(a.pending_positions)))
            self.assertTrue(all(a.tokens[p] == a.MASK and p < a._eos_position() for p in a.pending_positions))
            x, y = a.random_action(), b.random_action()
            self.assertEqual(x["action_type"], y["action_type"])
            np.testing.assert_array_equal(x["action_args"], y["action_args"])
            one, two = a.step(x), b.step(y)
            np.testing.assert_array_equal(one.obs, two.obs)
            if one.done:
                break
        else:
            self.fail("Episode exceeded max_layers steps")

    def test_reward_mae_mse_and_angle_polarization(self):
        cfg = deepcopy(self.cfg)
        cfg["angles_deg"] = [0., 25., 50.]
        results = {}
        for pol in ("s", "p", "unpolarized"):
            cfg["polarization"] = pol
            results[pol] = FilmOptics(cfg).evaluate([(3, 35.)])
        avg = np.array(results["unpolarized"]["T"])
        np.testing.assert_allclose(avg, (np.array(results["s"]["T"]) + results["p"]["T"]) / 2)
        self.assertAlmostEqual(results["unpolarized"]["reward"], avg.mean())
        cfg["reward_metric"] = "mse"
        env = ThinFilmEnv(cfg)
        env.reset()
        step = self.fill(env, [1], [3])
        self.assertAlmostEqual(float(step.reward[0]), 1 - step.info["mse"], places=6)
        self.assertNotAlmostEqual(step.info["reward"], step.info["mean_transmission"])

    def test_fresnel_and_lossless_quarter_wave_physics(self):
        optics = FilmOptics(self.cfg)
        bare = optics.evaluate([])
        ns = optics.substrate_index[0]
        self.assertAlmostEqual(bare["R"][0][0], abs((1 - ns) / (1 + ns)) ** 2, places=10)
        self.assertAlmostEqual(bare["T"][0][0], 4 * ns.real / abs(1 + ns) ** 2, places=10)
        optics.material_indices[0, :] = 1.4 + 0j
        optics.substrate_index[:] = 1.6 + 0j
        result = optics.evaluate([(0, 193 / (4 * 1.4))])
        reflectance = ((1.6 - 1.4 ** 2) / (1.6 + 1.4 ** 2)) ** 2
        self.assertAlmostEqual(result["R"][0][0], reflectance, places=10)
        self.assertAlmostEqual(result["T"][0][0], 1 - reflectance, places=10)

    def test_substrate_and_material_subset_are_configurable(self):
        cfg = deepcopy(self.cfg)
        cfg["material_files"] = cfg["material_files"][3:]
        cfg["substrate_file"] = cfg["material_files"][1]
        env = ThinFilmEnv(cfg)
        env.reset()
        self.assertEqual(env.EOS, 2)
        self.assertEqual(env.action_space["action_type"].n, 27)
        step = self.fill(env, [0, 1], [0, 1])
        self.assertEqual(step.info["layer_count"], 2)
        model = build_model(cfg)
        out = model(torch.as_tensor(step.obs).unsqueeze(0), "compute_actor")
        self.assertEqual(out["logit"]["action_type"].shape, (1, 27))

    def test_model_masks_match_in_all_actor_modes_and_unused_ranks_frozen(self):
        env = ThinFilmEnv(self.cfg)
        env.reset()
        env.pending_positions = [0, 4]
        obs = torch.as_tensor(env._observation()).repeat(2, 1)
        model = build_model(self.cfg)
        legal_codes = [code for code in range(216)
                       if decode_tokens(code, 5, 3)[0] != 5 and decode_tokens(code, 5, 3)[2] == 0]
        for mode in ("compute_actor", "compute_actor_critic"):
            out = model(obs, mode)["logit"]
            self.assertEqual(torch.where(out["action_type"][0] > -1e8)[0].tolist(), legal_codes)
            self.assertTrue((out["action_args"]["mu"][:, 2] == 0).all())
            self.assertTrue((out["action_args"]["sigma"][:, 2] == 1).all())
            self.assertTrue(torch.isfinite(out["action_type"]).all())

    def test_save_best_intermediate_and_global_best_use_physical_score(self):
        scratch = PROJECT_DIR / "outputs"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            cfg = deepcopy(self.cfg)
            cfg["result_dir"] = directory
            env = ThinFilmEnv(cfg)
            env.reset()
            step = self.fill(env, [2], [3])
            self.assertFalse(step.done)
            best = json.loads((Path(directory) / "best_design.json").read_text(encoding="utf-8"))
            self.assertFalse(best["terminal"])
            self.assertEqual(best["active_positions"], [2])
            weaker = {**best, "reward": best["reward"] - .01, "training_return": 1000.}
            other = ThinFilmEnv(cfg)
            other._save_best(weaker)
            self.assertEqual(json.loads((Path(directory) / "best_design.json").read_text())["reward"], best["reward"])
            optics = FilmOptics(cfg)
            self.assertAlmostEqual(optics.evaluate_slots(best["slots"])["reward"], best["reward"])

    def test_intermediate_below_min_layers_not_saved(self):
        scratch = PROJECT_DIR / "outputs"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            cfg = deepcopy(self.cfg)
            cfg.update(min_layers=3, result_dir=directory)
            env = ThinFilmEnv(cfg)
            env.reset()
            step = self.fill(env, [4], [3])
            self.assertFalse(step.info["eligible_design"])
            self.assertEqual(list(Path(directory).glob("*.json")), [])

    def test_single_position_action_and_tail_smaller_than_fill_min(self):
        for fill_min, fill_max in [(1, 1), (2, 3)]:
            cfg = {**self.cfg, "fill_min": fill_min, "fill_max": fill_max}
            env = ThinFilmEnv(cfg)
            env.reset()
            env.tokens[:5] = 3
            env.thicknesses[:5] = 35.
            env._select_positions()
            self.assertEqual(env.pending_positions, [5])
            obs = torch.as_tensor(env._observation()).unsqueeze(0)
            model = build_model(cfg)
            out = model(obs, "compute_actor")["logit"]
            self.assertEqual(out["action_args"]["mu"].shape, (1, fill_max))
            step = env.step({"action_type": encode_tokens([3], 5, fill_max),
                             "action_args": np.zeros(fill_max)})
            self.assertTrue(step.done)
            self.assertEqual(step.info["layer_count"], 6)

    def test_independent_slot_evaluation_truncates_first_eos(self):
        optics = FilmOptics(self.cfg)
        result = optics.evaluate_slots([None, {"material_id": 3, "thickness_nm": 35.},
                                        {"token": "EOS"}, {"material_id": 999}])
        expected = optics.evaluate([(3, 35.)])
        np.testing.assert_allclose(result["T"], expected["T"])
        self.assertEqual(result["active_positions"], [1])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            optics.evaluate_slots([{"material_id": 3, "material": "wrong", "thickness_nm": 35.}])

    def test_new_run_names_and_constraints_fail_explicitly(self):
        for update in ({"fill_max": 8}, {"fill_min": 0}, {"reward_metric": "wrong"},
                       {"max_total_thickness_nm": 400}, {"forbid_adjacent_same_material": True}):
            with self.assertRaises(ValueError):
                validate({**self.cfg, **update})
        scratch = PROJECT_DIR / "outputs"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            cfg = deepcopy(self.cfg)
            cfg["training"]["exp_name"] = str(Path(directory).relative_to(PROJECT_DIR))
            main, _ = build_configs(cfg, new_run=True)
            self.assertNotEqual(main.exp_name, cfg["training"]["exp_name"])
            self.assertEqual(Path(main.env.result_dir), PROJECT_DIR / main.exp_name / "designs")


if __name__ == "__main__":
    unittest.main()
