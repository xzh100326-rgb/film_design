"""Gradient consistency, constraints, incumbent retention and checkpoint guards."""

from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fixtures import yaml_config
from film_rl.config import PROJECT_DIR
from film_rl.improve import ThicknessObjective, discover_seeds, improve, neighbors, problem_fingerprint, project_thickness, refine
from film_rl.optics import FilmOptics


class ImprovementTests(unittest.TestCase):
    def setUp(self):
        scratch = PROJECT_DIR/"outputs"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="improve_test_", dir=scratch)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(scratch.resolve())
        self.addCleanup(self.temp.cleanup)
        self.cfg = yaml_config(self.root, optical_constants=[("low", 1.4, .001), ("lossy", 2.1, .2)])
        self.cfg.update(wavelengths_nm=[400., 600., 1000., 1800.], angles_deg=[0., 35.],
                        substrate_thickness_nm=500., exit_n=1., reward_alpha=0.,
                        spectral_targets=[{"quantity": "A", "target": 1.}])
        torch.set_num_threads(1)

    def test_autograd_matches_independent_finite_differences(self):
        for finite in (True, False):
            cfg = deepcopy(self.cfg)
            if not finite:
                cfg.pop("substrate_thickness_nm")
                cfg.pop("exit_n")
            optics = FilmOptics(cfg)
            objective = ThicknessObjective(optics, [0, 1])
            values = np.array([40., 55.])
            error, gradient = objective(values)
            self.assertAlmostEqual(error, optics.evaluate([(0, 40.), (1, 55.)])["mae"], places=12)
            differences = []
            for position in range(2):
                delta = np.zeros(2)
                delta[position] = 1e-3
                plus = optics.evaluate(list(zip([0, 1], values+delta)))["mae"]
                minus = optics.evaluate(list(zip([0, 1], values-delta)))["mae"]
                differences.append((plus-minus)/2e-3)
            np.testing.assert_allclose(gradient, differences, atol=1e-9, rtol=1e-5)

    def test_projection_and_neighbors_obey_original_constraints(self):
        self.cfg.update(max_total_thickness_nm=100., thickness_max_nm=90.)
        values = project_thickness([90., 90., -2.], self.cfg)
        self.assertLessEqual(values.sum(), 100.+1e-9)
        self.assertTrue(np.all((values >= 5.) & (values <= 90.)))
        for _, layers in neighbors([(0, 35.), (1, 45.)], self.cfg, 2, np.random.default_rng(3)):
            FilmOptics(self.cfg).evaluate(layers)
        with self.assertRaises(ValueError):
            project_thickness([10.]*21, self.cfg)

    def test_refinement_preserves_incumbent_and_constraints(self):
        optics = FilmOptics(self.cfg)
        initial = optics.evaluate([(0, 40.), (1, 55.)])
        result = refine(optics, [(0, 40.), (1, 55.)], max_iterations=20)
        self.assertGreaterEqual(result["score"], initial["score"])
        self.assertLessEqual(result["total_thickness_nm"], self.cfg["max_total_thickness_nm"]+1e-9)

    def test_fingerprint_pins_materials_substrate_and_constraints_not_seed(self):
        original = problem_fingerprint(self.cfg)
        changed = deepcopy(self.cfg)
        changed["training"]["seed"] += 1
        self.assertEqual(original, problem_fingerprint(changed))
        changed["substrate_thickness_nm"] += 1
        self.assertNotEqual(original, problem_fingerprint(changed))
        changed = deepcopy(self.cfg)
        changed["materials"].reverse()
        self.assertNotEqual(original, problem_fingerprint(changed))

    def test_discovery_accepts_null_budget_and_ranks_current_objective(self):
        cfg = deepcopy(self.cfg)
        cfg['max_total_thickness_nm'] = None
        optics = FilmOptics(cfg)
        candidates = [optics.evaluate([(0,40.)]), optics.evaluate([(1,55.)])]
        paths = []
        for index, report in enumerate(candidates):
            path = self.root/'runs'/f'run{index}'/'designs'/f'best_{index}.json'
            path.parent.mkdir(parents=True)
            # Deliberately invert the archived score ranking.
            saved = {'layers':report['layers'], 'score':1-report['score']}
            path.write_text(json.dumps(saved),encoding='utf-8')
            paths.append(path)
        expected = max(candidates,key=lambda r:r['score'])['score']
        with patch('film_rl.improve.PROJECT_DIR',self.root):
            found = discover_seeds(cfg,optics,limit=1)
        self.assertEqual(len(found),1)
        self.assertAlmostEqual(found[0]['score'],expected,places=12)
        self.assertEqual(len(discover_seeds(cfg,optics,design=paths[0],limit=1)),1)

    def test_discovery_prefers_distinct_material_sequences(self):
        cfg = deepcopy(self.cfg)
        cfg['max_total_thickness_nm'] = None
        optics = FilmOptics(cfg)
        for index,layers in enumerate([[(0,30.)],[(0,40.)],[(0,50.)],[(1,55.)]]):
            report = optics.evaluate(layers)
            path = self.root/'runs'/f'r{index}'/'designs'/f'best_{index}.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'layers':report['layers'],'score':report['score']}),encoding='utf-8')
        with patch('film_rl.improve.PROJECT_DIR',self.root):
            found = discover_seeds(cfg,optics,limit=2)
        self.assertEqual({tuple(x['material_id'] for x in r['layers']) for r in found},{(0,),(1,)})

    def directory_candidates(self):
        directory = self.root/'selected_designs'
        directory.mkdir()
        optics = FilmOptics(self.cfg)
        expected = []
        for index, thickness in enumerate([15., 25., 35., 45., 55., 65., 75.]):
            report = optics.evaluate([(1, thickness)])
            path = directory/'nested'/f'candidate{index}.json'
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({'layers': report['layers'], 'score': 1-report['score']}), encoding='utf-8')
            expected.append(report)
        # Duplicate, incompatible design, metadata and malformed JSON must not displace valid seeds.
        (directory/'duplicate.json').write_text((directory/'nested/candidate0.json').read_text(), encoding='utf-8')
        (directory/'metadata.json').write_text('{}', encoding='utf-8')
        (directory/'broken.json').write_text('{', encoding='utf-8')
        (directory/'incompatible.json').write_text(json.dumps({'layers': [
            {'material_id': 0, 'material': 'wrong', 'thickness_nm': 40.}]}), encoding='utf-8')
        return directory, sorted(expected, key=lambda r: r['score'], reverse=True)

    def test_directory_ranks_top_five_under_current_objective_without_diversity_filter(self):
        directory, expected = self.directory_candidates()
        scan = {}
        found = discover_seeds(self.cfg, FilmOptics(self.cfg), design_dir=directory, limit=5, scan_report=scan)
        self.assertEqual([r['score'] for r in found], [r['score'] for r in expected[:5]])
        self.assertTrue(all(Path(r['seed_origin']).is_relative_to(directory) for r in found))
        self.assertEqual(scan['candidates_evaluated'], 7)
        self.assertEqual(scan['duplicates_skipped'], 1)
        self.assertEqual(len(scan['skipped']), 3)

    def test_directory_supports_pareto_lists_and_records_source_index(self):
        directory = self.root/'pareto_inputs'
        directory.mkdir()
        optics = FilmOptics(self.cfg)
        rows = [optics.evaluate([(1, d)]) for d in [30., 60.]]
        (directory/'archive.json').write_text(json.dumps(rows), encoding='utf-8')
        found = discover_seeds(self.cfg, optics, design_dir=directory, limit=5)
        self.assertEqual({r['seed_origin_index'] for r in found}, {0, 1})

    def test_directory_improvement_refines_all_five_and_exports_each_result(self):
        directory, expected = self.directory_candidates()
        cfg = deepcopy(self.cfg)
        cfg['improvement'] = {'structure_rounds': 0, 'max_thickness_iterations': 2,
                              'warm_start_steps': 0, 'seed_limit': 1}
        output = self.root/'directory_improvement'
        summary = improve(cfg, output, design_dir=directory)
        seeds = json.loads((output/'seed_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(seeds['selected_count'], 5)  # Directory default overrides legacy seed_limit.
        self.assertEqual([r['score'] for r in seeds['seeds']], [r['score'] for r in expected[:5]])
        self.assertEqual(len(summary['seed_refinements']), 5)
        for row in summary['seed_refinements']:
            self.assertGreaterEqual(row['refined_score'], row['initial_score'])
            result = json.loads(Path(row['design']).read_text(encoding='utf-8'))
            self.assertEqual(result['seed_origin'], row['origin'])
            self.assertTrue(result['target_errors'])

    def test_explicit_empty_or_incompatible_inputs_never_fall_back_to_random(self):
        directory = self.root/'empty_inputs'
        directory.mkdir()
        with patch('film_rl.improve.refine') as refinement:
            with self.assertRaisesRegex(ValueError, 'No compatible designs'):
                improve(self.cfg, self.root/'empty_result', design_dir=directory)
            refinement.assert_not_called()
        bad = directory/'bad.json'
        bad.write_text(json.dumps({'layers': [{'material_id': 0, 'material': 'wrong', 'thickness_nm': 40.}]}))
        with self.assertRaisesRegex(ValueError, 'Incompatible design'):
            discover_seeds(self.cfg, FilmOptics(self.cfg), design=bad)

    def test_directory_cli_dispatch_and_argument_validation(self):
        from film_rl.__main__ import main
        directory = self.root/'cli_inputs'
        directory.mkdir()
        args = ['film_rl', 'improve', '--design-dir', str(directory), '--top-k', '5',
                '--config', 'specified.json', '--output', str(self.root/'cli_output')]
        with patch('sys.argv', args), patch('film_rl.__main__.load_config', return_value=self.cfg) as load, \
                patch('film_rl.improve.improve') as search:
            main()
            self.assertEqual(load.call_args.args[0], Path('specified.json'))
            search.assert_called_once_with(self.cfg, self.root/'cli_output', None, design_dir=directory, top_k=5)
        for extra in [['--design', 'one.json'], ['--top-k', '0']]:
            with patch('sys.argv', args+extra), patch('sys.stderr', new_callable=__import__('io').StringIO):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

    def test_warm_start_survives_native_ppo_initialization(self):
        from ding.config import compile_config
        from film_rl.__main__ import load_initial_model
        from film_rl.hppo import build_configs, build_model
        from film_rl.policy import FilmPPOPolicy
        model = build_model(self.cfg)
        path = self.root/"warm_start.pt"
        torch.save({"model": model.state_dict(), "problem_fingerprint": problem_fingerprint(self.cfg)}, path)
        self.cfg["training"]["initial_model"] = str(path)
        main, create = build_configs(self.cfg)
        self.assertFalse(main.policy.learn.ppo_param_init)
        create.policy.type += "_command"
        compiled = compile_config(main, seed=0, auto=True, create_cfg=create, save_cfg=False)
        loaded = build_model(self.cfg)
        load_initial_model(self.cfg, loaded)
        policy = FilmPPOPolicy(compiled.policy, model=loaded, enable_field=["learn", "collect", "eval"])
        for name, value in policy._model.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        self.cfg["substrate_thickness_nm"] += 1
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_initial_model(self.cfg, loaded)


if __name__ == "__main__":
    unittest.main()
