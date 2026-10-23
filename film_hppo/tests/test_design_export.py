"""Exports preserve the objective and structure while removing dense arrays."""

from copy import deepcopy
import json
import unittest

from film_rl.design import compact_design


class DesignExportTests(unittest.TestCase):
    def report(self):
        return {
            'layer_count': 1, 'total_thickness_nm': 40., 'score': .8, 'reward': .7, 'mae': .2,
            'evaluation_metric': 'score', 'wavelengths_nm': [400., 500., 600.], 'angles_deg': [0., 45.],
            'R': [[.2, .3, .4], [.4, .5, .6]], 'T': [[.8, .7, .6], [.6, .5, .4]],
            'A': [[0., 0., 0.], [0., 0., 0.]],
            'target_errors': [{'quantity': 'R', 'wavelength_range_nm': [400., 500.],
                               'target': [[0., 0.], [0., 0.]], 'weight': 1., 'normalized_weight': 1.,
                               'mean_actual': .35, 'mae': .35, 'min_actual': .2, 'max_actual': .5,
                               'wavelengths_nm': [400., 500.]}],
            'coating_optical_constants': [{'n': [1.5, 1.5, 1.5]}],
            'layers': [{'material_id': 0, 'material': 'low', 'thickness_nm': 40.}],
            'material_datasets': [{'file': 'data/main/low/nk/local.yml', 'sha256': 'source-hash'}],
        }

    def test_band_means_and_exact_structure_without_mutation(self):
        full = self.report()
        before = deepcopy(full)
        exported = compact_design(full)
        self.assertEqual(full, before)
        self.assertEqual(list(exported)[:3], ['layer_count', 'total_thickness_nm', 'score'])
        self.assertEqual(list(exported)[-1], 'layers')
        self.assertEqual(exported['layers'], full['layers'])
        self.assertEqual(exported['material_datasets'], full['material_datasets'])
        for key in ('R', 'T', 'A', 'wavelengths_nm', 'angles_deg', 'coating_optical_constants'):
            self.assertNotIn(key, exported)
        term = exported['target_errors'][0]
        self.assertEqual(term['target'], 0.)
        self.assertAlmostEqual(term['mean_R'], .35)
        self.assertAlmostEqual(term['mean_T'], .65)
        self.assertEqual(term['mean_A'], 0.)
        self.assertEqual(term['angle_count'], 2)
        self.assertEqual(term['wavelength_count'], 2)
        self.assertEqual(json.loads(json.dumps(exported)), exported)
        self.assertEqual(compact_design(exported), exported)

    def test_varying_target_is_not_misrepresented_as_constant(self):
        full = self.report()
        full['target_errors'][0]['target'] = [[0., .2], [.4, .6]]
        target = compact_design(full)['target_errors'][0]['target']
        self.assertEqual(target['type'], 'varying_spectrum')
        self.assertEqual(target['min'], 0.)
        self.assertEqual(target['max'], .6)

    def test_non_design_json_is_unchanged(self):
        config = {'training': {'seed': 1}, 'spectral_targets': [{'target': [0, 1]}]}
        self.assertEqual(compact_design(config), config)

    def test_generic_s_formulas_are_retained_without_dense_spectra(self):
        full = self.report()
        full.update(score_formula="1 - sum_b(w_b * MAE(S_b,S_target_b))",
                    reward_formula="score", spectral_aggregation="weighted_band_mae")
        result = compact_design(full)
        for key in ("score_formula", "reward_formula", "spectral_aggregation"):
            self.assertEqual(result[key], full[key])
        self.assertNotIn("R", result)


if __name__ == '__main__':
    unittest.main()
