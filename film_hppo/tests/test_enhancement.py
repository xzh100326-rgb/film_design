"""Independent limiting cases and spectral/angle integration checks."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.constants import Stefan_Boltzmann
from scipy.integrate import simpson

from film_rl.enhancement import enhancement, eye_response, hemisphere_quadrature, planck_exitance
from film_rl.optics import FilmOptics
from fixtures import yaml_config, write_substrate
from evaluate_baseline import calculate_spectra, read_layers


class EnhancementTests(unittest.TestCase):
    def setUp(self):
        self.wave = np.linspace(300.,3000.,271)

    def test_bare_filter_and_zero_view_factor_are_unity(self):
        for r,f in [(0.,1.),(.8,0.)]:
            result=enhancement(self.wave,np.full_like(self.wave,r),view_factor=f)
            self.assertAlmostEqual(result["chi"],1.,places=12)
            self.assertAlmostEqual(result["temperature_k"],result["bare_temperature_k"],places=9)
        expected=(100/(2*20e-6*Stefan_Boltzmann))**.25
        self.assertAlmostEqual(result["bare_temperature_k"],expected,places=10)

    def test_constant_grey_filter_matches_analytic_temperature(self):
        r,f=.75,.95
        result=enhancement(self.wave,np.full_like(self.wave,r),view_factor=f,outside_reflectance=r)
        expected=result["bare_temperature_k"]/(1-f*f*r)**.25
        self.assertAlmostEqual(result["temperature_k"],expected,places=6)
        self.assertLess(abs(result["power_balance_residual_w"]),1e-7)

    def test_planck_nm_units_match_stefan_boltzmann(self):
        wave=np.geomspace(1.,1e8,100001)
        temp=3000.
        integrated=simpson(planck_exitance(wave,temp),x=wave)
        self.assertAlmostEqual(integrated/(Stefan_Boltzmann*temp**4),1.,places=7)

    def test_hemisphere_weights_integrate_cosine_law(self):
        angles,weights=hemisphere_quadrature(32)
        self.assertAlmostEqual(weights.sum(),1.,places=14)
        # R(theta)=sin(theta)^2 integrates to 1/2, not a uniform-theta mean.
        self.assertAlmostEqual(float(weights@np.sin(np.deg2rad(angles))**2),.5,places=14)
        self.assertTrue(np.all((angles>0)&(angles<90)))

    def test_original_sharpe2005_matches_published_table3(self):
        wave,v,source=eye_response()
        # Values from the original 2005 Table 3, energy units, NOT CIE 2008.
        table={390:.00041276,400:.0024477,480:.18348,555:.99987,600:.66879,700:.0039376,830:.00000066438}
        for wavelength,value in table.items():
            self.assertAlmostEqual(float(np.interp(wavelength,wave,v))/value,1.,delta=5e-5)
        self.assertIn("1.624340",source["formula"])

    def test_tail_bounds_and_invalid_inputs(self):
        r=np.full_like(self.wave,.8)
        lower=enhancement(self.wave,r,outside_reflectance=0.)
        upper=enhancement(self.wave,r,outside_reflectance=1.)
        self.assertGreater(upper["chi"],lower["chi"])
        for options in ({"view_factor":1.01},{"outside_reflectance":-1},{"power_w":0},{"radiating_sides":True}):
            with self.assertRaises(ValueError):enhancement(self.wave,r,**options)
        with self.assertRaises(ValueError):enhancement([500,600,700],[.1,.2,.3])
        with self.assertRaises(ValueError):enhancement(self.wave,np.ones_like(self.wave),outside_reflectance=1.)

    def test_chunked_tmm_matches_production_and_rejects_wrong_material(self):
        scratch=Path(__file__).resolve().parents[1]/"outputs"
        with tempfile.TemporaryDirectory(prefix="baseline_test_",dir=scratch) as folder:
            root=Path(folder)
            cfg=yaml_config(root)
            write_substrate(root,n=1.)
            cfg.update(wavelengths_nm=[400.,500.,600.,700.,1000.,1500.,2000.],angles_deg=[0.,30.,70.],
                       polarization="unpolarized")
            optics=FilmOptics(cfg)
            layers=[(0,33.),(1,67.)]
            expected=optics.evaluate(layers)
            actual=calculate_spectra(optics,layers,np.asarray(cfg["angles_deg"]),2)
            for q in "RTA":np.testing.assert_allclose(actual[q],expected[q],atol=1e-12)
            with self.assertRaisesRegex(ValueError,"material"):
                read_layers({"layers":[{"material_id":0,"material":"WRONG","thickness_nm":33.}]},optics)


if __name__=="__main__":unittest.main()
