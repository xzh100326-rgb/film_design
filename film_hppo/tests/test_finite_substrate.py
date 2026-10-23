"""Finite substrate physics, legacy boundary condition and coating budgets."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fixtures import yaml_config, write_substrate
from film_rl.config import validate
from film_rl.optics import FilmOptics


class FiniteSubstrateTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "outputs"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="finite_substrate_test_", dir=scratch)
        self.root = Path(self.temp.name)
        assert self.root.resolve().is_relative_to(scratch.resolve())
        self.addCleanup(self.temp.cleanup)
        self.cfg = yaml_config(self.root)
        write_substrate(self.root, n=1.45)
        self.cfg.update(wavelengths_nm=[400., 550., 1000., 2000.], angles_deg=[0.],
                        substrate_thickness_nm=500., exit_n=1.)

    def test_glass_slab_matches_airy_formula(self):
        result = FilmOptics(self.cfg).evaluate([])
        interface_r = ((1.45 - 1) / (1.45 + 1)) ** 2
        phase = 2 * np.pi * 1.45 * 500 / np.asarray(self.cfg["wavelengths_nm"])
        transmission = 1 / (1 + 4 * interface_r / (1-interface_r)**2 * np.sin(phase)**2)
        np.testing.assert_allclose(result["T"][0], transmission, atol=1e-12, rtol=0)
        np.testing.assert_allclose(result["R"][0], 1-transmission, atol=1e-12, rtol=0)
        np.testing.assert_allclose(result["A"], 0, atol=1e-12)

    def test_legacy_semi_infinite_matches_fresnel(self):
        self.cfg.pop("substrate_thickness_nm")
        self.cfg.pop("exit_n")
        result = FilmOptics(self.cfg).evaluate([])
        np.testing.assert_allclose(result["R"], ((1.45-1)/(1.45+1))**2, atol=1e-12)
        self.assertIsNone(result["substrate_thickness_nm"])

    def test_substrate_excluded_from_coating_constraints(self):
        self.cfg.update(max_layers=1, max_total_thickness_nm=10.)
        result = FilmOptics(self.cfg).evaluate([(0, 10.)])
        self.assertEqual(result["layer_count"], 1)
        self.assertEqual(result["total_thickness_nm"], 10.)
        self.assertEqual(result["substrate_thickness_nm"], 500.)

    def test_invalid_boundary_configuration_rejected(self):
        for thickness in [0, -1, True, float("inf"), "500"]:
            with self.subTest(thickness=thickness), self.assertRaises(ValueError):
                validate({**self.cfg, "substrate_thickness_nm": thickness})
        for index in [0, -1, True, float("nan")]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate({**self.cfg, "exit_n": index})
        with self.assertRaises(ValueError):
            validate({**self.cfg, "substrate_thickness_nm": None})


if __name__ == "__main__":
    unittest.main()
