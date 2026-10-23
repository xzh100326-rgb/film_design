"""Read-only audit of historical single-/multi-angle runs; never trains a model.

Run from the project root: python diagnostics/audit_multiangle.py
Historical missing constraints are restored explicitly rather than inheriting
the newer environment defaults. Checkpoints are loaded with weights_only=True.
"""

import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from film_rl.config import load_config
from film_rl.env import ThinFilmEnv
from film_rl.hppo import build_model
from film_rl.optics import FilmOptics
from film_rl.policy import FilmPPOPolicy
from easydict import EasyDict
from ding.rl_utils.gae import gae, gae_data


def historical_config(run):
    class InfLiteral(ast.NodeTransformer):
        def visit_Call(self, node):
            if (isinstance(node.func, ast.Name) and node.func.id == "float"
                    and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "inf"):
                return ast.copy_location(ast.Constant(value=None), node)
            raise ValueError("Unexpected executable expression in saved config")

    tree = ast.parse((run / "total_config.py").read_text(encoding="utf-8"))
    return ast.literal_eval(InfLiteral().visit(tree.body[0].value))


def restore_env(saved):
    cfg = load_config()
    cfg.update(deepcopy(saved["env"]))
    # Historical configs without a source used the original .lm interpolation.
    cfg["material_source"] = saved["env"].get("material_source", "lm")
    if cfg["material_source"] == "lm":
        cfg.pop("fixed_materials_193nm", None)
        cfg.pop("fixed_materials_reference", None)
    cfg.pop("result_dir", None)  # Audit must not write designs into training runs.
    cfg["max_total_thickness_nm"] = saved["env"].get("max_total_thickness_nm")
    cfg["forbid_adjacent_same_material"] = saved["env"].get("forbid_adjacent_same_material", False)
    cfg["reward_alpha"] = saved["env"].get("reward_alpha", 0.0)
    cfg["reward_epsilon"] = saved["env"].get("reward_epsilon", 1e-6)
    return cfg


def compare_configs(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            out.extend(compare_configs(a.get(key), b.get(key), path + key + "."))
    elif a != b:
        out.append({"key": path.rstrip("."), "single": a, "multi": b})
    return out


def reconstructed_initial_state(cfg, saved):
    # The native pipeline reseeds immediately before create_policy; learning
    # initialization resets model weights when ppo_param_init is enabled.
    # Instantiate the policy, but never call its learn/forward/backward methods.
    torch.manual_seed(saved["seed"])
    model = build_model(cfg)
    torch.manual_seed(saved["seed"])
    policy = FilmPPOPolicy(EasyDict(deepcopy(saved["policy"])), model=model, enable_field=["learn"])
    return {"model": {key: value.detach().clone() for key, value in policy._model.state_dict().items()}}


def independent_matrix_transmission(optics, stack, polarization):
    # NumPy characteristic-admittance matrix, independent of TMM-fast's
    # interface/propagation-matrix implementation. n+i*k uses -i*sin(delta).
    rows = []
    for angle in optics.angles_deg:
        row = []
        for wave_index, wavelength in enumerate(optics.wavelengths_nm):
            n0 = optics.incident_index[wave_index]
            indices = [n0] + [optics.material_indices[i, wave_index] for i, _ in stack] + [optics.substrate_index[wave_index]]
            cosines = np.sqrt(1 - (n0 * np.sin(np.deg2rad(angle)) / np.asarray(indices)) ** 2)
            admittance = np.asarray(indices) * cosines if polarization == "s" else np.asarray(indices) / cosines
            matrix = np.eye(2, dtype=complex)
            for layer_index, (_, thickness) in enumerate(stack, 1):
                delta = 2 * np.pi * indices[layer_index] * cosines[layer_index] * thickness / wavelength
                c, s, q = np.cos(delta), np.sin(delta), admittance[layer_index]
                matrix = matrix @ np.array([[c, -1j * s / q], [-1j * q * s, c]])
            q0, qs = admittance[0], admittance[-1]
            amplitude = 2 * q0 / (q0 * matrix[0, 0] + q0 * qs * matrix[0, 1] + matrix[1, 0] + qs * matrix[1, 1])
            row.append(float(abs(amplitude) ** 2 * qs.real / q0.real))
        rows.append(row)
    return np.asarray(rows)


@torch.no_grad()
def model_rollouts(cfg, checkpoint, count=128):
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    # Use the same sampling seed for both policies. Only forward calls occur.
    rng = torch.Generator().manual_seed(71015)
    lengths, rewards, eos_probabilities, reasons = [], [], [], {}
    for _ in range(count):
        env = ThinFilmEnv(cfg)
        obs = env.reset()
        first_eos = None
        for step in range(cfg["max_layers"] + 1):
            logits = model(torch.from_numpy(obs).unsqueeze(0), "compute_actor")["logit"]
            probabilities = logits["action_type"].softmax(-1)
            if len(env.layers) == 1:
                first_eos = float(probabilities[0, 5])
            kind = int(torch.multinomial(probabilities, 1, generator=rng).item())
            args = logits["action_args"]
            latent = args["mu"] + args["sigma"] * torch.randn(args["mu"].shape, generator=rng)
            timestep = env.step({"action_type": kind, "action_args": latent[0].numpy()})
            obs = timestep.obs
            if timestep.done:
                lengths.append(step + 1)
                rewards.append(timestep.info["reward"])
                if first_eos is not None:
                    eos_probabilities.append(first_eos)
                reason = timestep.info["termination_reason"]
                reasons[reason] = reasons.get(reason, 0) + 1
                break
        else:
            raise AssertionError("Rollout exceeded valid length")
    return {"episodes": count, "mean_steps": float(np.mean(lengths)),
            "two_step_fraction": float(np.mean(np.asarray(lengths) == 2)),
            "mean_reward": float(np.mean(rewards)),
            "mean_EOS_probability_after_one_layer": float(np.mean(eos_probabilities)),
            "termination_reasons": reasons}


def audit_physics(cfg):
    multi = deepcopy(cfg)
    multi["angles_deg"] = list(np.linspace(0, 45, 10))
    optics = FilmOptics(multi)
    stacks = [[], [(3, 35.)], [(3, 103.), (1, 96.), (3, 65.), (3, 82.)],
              [(i % 5, 30. + (i * 31) % 181) for i in range(16)]]
    max_error, max_polarization_error, max_reward_error, max_matrix_error = 0., 0., 0., 0.
    for stack in stacks:
        result = optics.evaluate(stack)
        individual = []
        for angle in multi["angles_deg"]:
            single = deepcopy(multi)
            single["angles_deg"] = [angle]
            individual.append(FilmOptics(single).evaluate(stack)["T"])
        difference = np.max(np.abs(np.asarray(result["T"]) - np.concatenate(individual, axis=0)))
        max_error = max(max_error, float(difference))
        polarizations = []
        for polarization in ("s", "p"):
            polarized = deepcopy(multi)
            polarized["polarization"] = polarization
            polarized_optics = FilmOptics(polarized)
            calculated = np.asarray(polarized_optics.evaluate(stack)["T"])
            polarizations.append(calculated)
            max_matrix_error = max(max_matrix_error, float(np.max(np.abs(
                calculated - independent_matrix_transmission(polarized_optics, stack, polarization)))))
        max_polarization_error = max(max_polarization_error, float(np.max(np.abs(
            np.asarray(result["T"]) - np.mean(polarizations, axis=0)))))
        max_reward_error = max(max_reward_error, abs(result["reward"] - np.asarray(result["T"]).mean()))
    # Independent oblique-incidence Fresnel formula for the bare substrate.
    bare = optics.evaluate([])
    n0, ns = multi["incident_n"], optics.substrate_index[0]
    theta = np.deg2rad(optics.angles_deg)
    ct0 = np.cos(theta)
    cts = np.sqrt(1 - (n0 * np.sin(theta) / ns) ** 2)
    rs = (n0 * ct0 - ns * cts) / (n0 * ct0 + ns * cts)
    rp = (ns * ct0 - n0 * cts) / (ns * ct0 + n0 * cts)
    expected_t = 1 - (np.abs(rs) ** 2 + np.abs(rp) ** 2) / 2
    fresnel_error = float(np.max(np.abs(np.asarray(bare["T"])[:, 0] - expected_t)))
    assert max(max_error, max_polarization_error, max_reward_error, fresnel_error, max_matrix_error) < 1e-10

    # Also check unequal angle and wavelength dimensions to detect transposes.
    multiwave = deepcopy(multi)
    multiwave["wavelengths_nm"] = [193., 200., 250.]
    multiwave_solver = FilmOptics(multiwave)
    calculated = np.asarray(multiwave_solver.evaluate(stacks[2])["T"])
    reference = np.mean([independent_matrix_transmission(multiwave_solver, stacks[2], p) for p in ("s", "p")], axis=0)
    assert calculated.shape == (10, 3)
    multiwave_error = float(np.max(np.abs(calculated - reference)))
    assert multiwave_error < 1e-10

    # The same actions must produce identical states/done flags for both angles.
    single = deepcopy(multi)
    single["angles_deg"] = [0.]
    envs = [ThinFilmEnv(single), ThinFilmEnv(multi)]
    np.testing.assert_array_equal(envs[0].reset(), envs[1].reset())
    for kind in [3, 1, 4, 5]:
        results = [env.step({"action_type": kind, "action_args": np.zeros(5)}) for env in envs]
        np.testing.assert_array_equal(results[0].obs, results[1].obs)
        assert results[0].done == results[1].done
        if kind != 5:
            assert all(float(result.reward[0]) == 0 for result in results)
    assert all(env.optics.evaluation_count == 1 for env in envs)
    return {"stacks_checked": len(stacks), "angles": multi["angles_deg"],
            "vector_vs_individual_max_error": max_error,
            "unpolarized_vs_mean_s_p_max_error": max_polarization_error,
            "reward_vs_mean_T_max_error": max_reward_error,
            "bare_substrate_vs_fresnel_max_error": fresnel_error,
            "multilayer_vs_independent_numpy_matrix_max_error": max_matrix_error,
            "multiple_wavelengths_vs_independent_matrix_max_error": multiwave_error,
            "same_actions_identical_states_and_termination": True,
            "one_TMM_evaluation_per_episode": True}


def saved_designs(cfg):
    # Re-evaluate the exact same structures at both objectives (no optimization).
    optical = deepcopy(cfg)
    optical["angles_deg"] = list(np.linspace(0, 45, 10))
    solver = FilmOptics(optical)
    rows = []
    for name in ["duv193_hppo_shapesafe_seed1", "duv193_hppo_shapesafe_seed1_angle_range0_45"]:
        for path in sorted((ROOT / "runs" / name / "designs").glob("*.json")):
            stored = json.loads(path.read_text(encoding="utf-8"))
            layers = [(layer["material_id"], layer["thickness_nm"]) for layer in stored["layers"]]
            result = solver.evaluate(layers)
            rows.append({"source": name, "file": path.name, "layers": len(layers),
                         "T_normal": result["T"][0][0], "T_multi_mean": result["mean_transmission"]})
    return rows


@torch.no_grad()
def initial_policy_length_probe(cfg, checkpoint, count=64):
    # Exploratory evidence only: force a fixed length, ignore EOS, and sample
    # materials from the initial policy conditional on selecting a material.
    # These are not on-policy trajectories and do not measure final optimality.
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    rng = torch.Generator().manual_seed(204193)
    obs = torch.zeros(count, cfg["max_layers"] * 6 + 1)
    stacks = [[] for _ in range(count)]
    optical = deepcopy(cfg)
    optical["angles_deg"] = list(np.linspace(0, 45, 10))
    solver = FilmOptics(optical)
    rows = []
    for index in range(32):
        logits = model(obs, "compute_actor")["logit"]
        kinds = torch.multinomial(logits["action_type"][:, :5].softmax(-1), 1, generator=rng).squeeze(-1)
        args = logits["action_args"]
        latent = args["mu"] + args["sigma"] * torch.randn(args["mu"].shape, generator=rng)
        normalized = (latent[torch.arange(count), kinds].tanh() + 1) / 2
        thickness = cfg["thickness_min_nm"] + normalized * (cfg["thickness_max_nm"] - cfg["thickness_min_nm"])
        obs[torch.arange(count), index * 6 + kinds] = 1
        obs[:, index * 6 + 5] = normalized
        obs[:, -1] = (index + 1) / cfg["max_layers"]
        for row in range(count):
            stacks[row].append((int(kinds[row]), float(thickness[row])))
        if index + 1 in [1, 2, 4, 8, 16, 32]:
            scores = [solver.evaluate(stack) for stack in stacks]
            normal = np.asarray([result["T"][0][0] for result in scores])
            multi = np.asarray([result["mean_transmission"] for result in scores])
            rows.append({"layers": index + 1, "samples": count,
                         "mean_T_normal": float(normal.mean()), "mean_T_multi": float(multi.mean()),
                         "max_T_normal": float(normal.max()), "max_T_multi": float(multi.max())})
    return rows


def log_summary(run):
    acc = EventAccumulator(str(run / "log" / "serial"), size_guidance={"scalars": 0})
    acc.Reload()
    steps = acc.Scalars("collector_iter/avg_envstep_per_episode")
    crossing = next((i for i in range(len(steps)) if all(e.value < 2.1 for e in steps[i:])), None)
    rewards = acc.Scalars("evaluator_iter/reward_mean")
    envsteps = acc.Scalars("collector_iter/total_envstep_count")
    crossing_step = steps[crossing].step if crossing is not None else None
    environment_count = next((e.value for e in envsteps if e.step == crossing_step), None)
    return {"always_below_2.1_after_update": crossing_step,
            "environment_steps_at_crossing": environment_count,
            "initial_mean_episode_steps": steps[0].value, "final_mean_episode_steps": steps[-1].value,
            "final_eval_mean_reward": rewards[-1].value,
            "best_eval_mean_reward": max(e.value for e in rewards)}


def gae_probe():
    rows = []
    for length in [2, 5, 10, 20, 50]:
        reward = torch.zeros(length)
        reward[-1] = .97
        done = torch.zeros(length)
        done[-1] = 1
        advantage = gae(gae_data(torch.zeros(length), torch.zeros(length), reward, done, done), 1., .95)
        rows.append({"episode_steps": length, "first_advantage_if_values_zero": float(advantage[0]),
                     "actual_undiscounted_return": .97})
    return rows


def main():
    torch.set_num_threads(1)
    names = ["duv193_hppo_shapesafe_seed1", "duv193_hppo_shapesafe_seed1_angle_range0_45"]
    runs = [ROOT / "runs" / name for name in names]
    saved = [historical_config(run) for run in runs]
    cfgs = [restore_env(item) for item in saved]
    # iteration_0 is saved by an after_iter hook AFTER the first whole learn
    # call; it is not an untrained checkpoint. Verify optimizer step counters.
    earliest = [torch.load(run / "ckpt" / "iteration_0.pth.tar", map_location="cpu", weights_only=True) for run in runs]
    earliest_info = [{"run": run.name, "last_iter": ckpt["last_iter"], "last_step": ckpt["last_step"],
                      "optimizer_steps": max(float(v["step"]) for v in ckpt["optimizer"]["state"].values())}
                     for run, ckpt in zip(runs, earliest)]
    initial = [reconstructed_initial_state(cfg, item) for cfg, item in zip(cfgs, saved)]
    identical = all(torch.equal(initial[0]["model"][key], initial[1]["model"][key]) for key in initial[0]["model"])
    assert identical
    print("Checking physics and historical checkpoints...", flush=True)
    report = {"training_executed": False, "core_code_modified": False,
              "historical_config_differences": compare_configs(saved[0], saved[1]),
              "reconstructed_initial_parameters_bitwise_identical": identical,
              "initial_state_caveat": "Reconstructed from saved configs and current installed initialization; historical untrained checkpoint is unavailable.",
              "earliest_saved_checkpoints_already_trained": earliest_info,
              "physics": audit_physics(cfgs[0]), "logs": {}, "final_policy_rollouts": {},
              "saved_designs_cross_evaluation": saved_designs(cfgs[0]),
              "GAE_zero_value_illustration": gae_probe()}
    for run, cfg in zip(runs, cfgs):
        report["logs"][run.name] = log_summary(run)
        last = max(run.glob("ckpt/iteration_*.pth.tar"), key=lambda path: int(path.name.split("_")[1].split(".")[0]))
        print("Forward-only rollouts:", run.name, flush=True)
        checkpoint = torch.load(last, map_location="cpu", weights_only=True)
        report["final_policy_rollouts"][run.name] = model_rollouts(cfg, checkpoint)
    print("Fixed-length initial-policy probe (not training)...", flush=True)
    report["initial_policy_fixed_length_probe"] = initial_policy_length_probe(cfgs[0], initial[0])
    destination = ROOT / "outputs" / "multiangle_audit.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(destination)
    print(json.dumps({key: report[key] for key in ["physics", "logs", "final_policy_rollouts", "initial_policy_fixed_length_probe", "GAE_zero_value_illustration"]}, indent=2))


if __name__ == "__main__":
    main()
