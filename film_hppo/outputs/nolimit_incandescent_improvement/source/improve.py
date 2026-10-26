"""Auditable hybrid design search and HPPO warm start; not a pure-PPO claim.

Structure mutations explore discrete choices. SLSQP uses TMM-fast autograd for
all coating thicknesses together, with the original box and total constraints.
Every accepted design is re-evaluated by FilmOptics, then an optional VAC learns
the successful trajectory. No literature structure is embedded in this module.
"""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tmm_fast import coh_tmm

from .config import PROJECT_DIR, optical_data_files
from .optics import FilmOptics
from .design import compact_design


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(compact_design(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def problem_fingerprint(cfg):
    """Pin physics, material ordering, constraints and objective, not training knobs."""
    physics = {k: v for k, v in cfg.items() if k not in (
        "training", "improvement", "material_dir", "refractiveindex_dir", "result_dir")}
    files = [{"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in optical_data_files(cfg)]
    return hashlib.sha256(json.dumps({"physics": physics, "files": files},
                                    sort_keys=True).encode()).hexdigest()


def project_thickness(values, cfg):
    """Euclidean projection onto the original thickness box and total budget."""
    values = np.asarray(values, dtype=np.float64)
    low, high = cfg["thickness_min_nm"], cfg["thickness_max_nm"]
    if len(values) < cfg["min_layers"] or len(values) > cfg["max_layers"]:
        raise ValueError("Infeasible layer count")
    budget = cfg.get("max_total_thickness_nm")
    if budget is not None and len(values)*low > budget:
        raise ValueError("Too many layers for the thickness budget")
    clipped = np.clip(values, low, high)
    if budget is None or clipped.sum() <= budget:
        return clipped
    left, right = 0., max(float(np.max(values-low)), 0.) + 1.
    for _ in range(70):
        midpoint = (left+right)/2
        if np.clip(values-midpoint, low, high).sum() > budget:
            left = midpoint
        else:
            right = midpoint
    return np.clip(values-right, low, high)


class ThicknessObjective:
    """Differentiable exact configured target score, using native TMM-fast."""
    def __init__(self, optics, material_ids):
        self.optics = optics
        device = optics.cfg["tmm_device"]
        indices = [optics.incident_index] + [optics.material_indices[i] for i in material_ids]
        indices.append(optics.substrate_index)
        if optics.substrate_thickness_nm is not None:
            indices.append(np.full_like(optics.incident_index, optics.cfg.get("exit_n", 1.)))
        self.indices = torch.as_tensor(np.stack(indices)[None], dtype=torch.complex128, device=device)
        self.wave = torch.as_tensor(optics.wavelengths_nm*1e-9, dtype=torch.float64, device=device)
        self.angles = torch.as_tensor(np.deg2rad(optics.angles_deg), dtype=torch.float64, device=device)
        self.pols = ("s", "p") if optics.cfg["polarization"] == "unpolarized" else (optics.cfg["polarization"],)
        self.calls = 0

    def __call__(self, values):
        x = torch.tensor(values, dtype=torch.float64, device=self.indices.device, requires_grad=True)
        pieces = [x.new_tensor([np.inf]), x]
        if self.optics.substrate_thickness_nm is not None:
            pieces.append(x.new_tensor([self.optics.substrate_thickness_nm]))
        pieces.append(x.new_tensor([np.inf]))
        thickness = torch.cat(pieces)[None]*1e-9
        results = [coh_tmm(pol, self.indices, thickness, self.angles, self.wave,
                           device=self.indices.device) for pol in self.pols]
        reflection = torch.stack([r["R"][0] for r in results]).mean(0)
        transmission = torch.stack([r["T"][0] for r in results]).mean(0)
        spectra = {"R": reflection, "T": transmission, "A": 1-reflection-transmission}
        error = x.new_tensor(0.)
        for term in self.optics.objective.terms:
            target = torch.as_tensor(term["target"], dtype=x.dtype, device=x.device)
            error = error + term["normalized_weight"] * (
                spectra[term["quantity"]][:, term["mask"]]-target).abs().mean()
        gradient, = torch.autograd.grad(error, x)
        self.calls += 1
        if not torch.isfinite(error) or not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite TMM objective/gradient")
        return float(error.detach()), gradient.detach().cpu().numpy()


def refine(optics, layers, max_iterations=80):
    from scipy.optimize import minimize
    ids = [int(i) for i, _ in layers]
    x0 = project_thickness([d for _, d in layers], optics.cfg)
    initial = optics.evaluate(list(zip(ids, x0)))
    objective = ThicknessObjective(optics, ids)
    # Scale the error for SLSQP's stopping criteria; the reported score is raw.
    low, high = optics.cfg["thickness_min_nm"], optics.cfg["thickness_max_nm"]
    span = high-low
    def function(unit):
        error, grad = objective(low+unit*span)
        return error*100., grad*100.*span
    constraints = []
    budget = optics.cfg.get("max_total_thickness_nm")
    if budget is not None:
        constraints.append({"type": "ineq", "fun": lambda unit: (budget-(low+unit*span).sum())/span,
                            "jac": lambda unit: -np.ones_like(unit)})
    result = minimize(function, (x0-low)/span, jac=True, method="SLSQP", bounds=[(0., 1.)]*len(ids),
                      constraints=constraints,
                      options={"maxiter": max_iterations, "ftol": 1e-9})
    x = project_thickness(low+result.x*span, optics.cfg)
    report = optics.evaluate(list(zip(ids, x)))
    # A failed/constrained optimizer must never replace the incumbent with a worse design.
    if report["score"] < initial["score"]:
        report = initial
    report["optimization"] = {"method": "SLSQP with TMM-fast autograd", "gradient_calls": objective.calls,
                              "success": bool(result.success), "message": str(result.message),
                              "initial_score": initial["score"]}
    return report


def layers_of(report):
    return [(int(x["material_id"]), float(x["thickness_nm"])) for x in report["layers"]]


def legal_structure(ids, cfg, count):
    return (cfg["min_layers"] <= len(ids) <= cfg["max_layers"]
            and all(0 <= i < count for i in ids)
            and (not cfg.get("forbid_adjacent_same_material", True)
                 or all(a != b for a, b in zip(ids, ids[1:])))
            and (cfg.get("max_total_thickness_nm") is None
                 or len(ids)*cfg["thickness_min_nm"] <= cfg["max_total_thickness_nm"]))


def neighbors(layers, cfg, count, rng):
    ids, ds = [i for i, _ in layers], [d for _, d in layers]
    proposals = []
    for pos in range(len(ids)):
        for material in range(count):
            if material != ids[pos]:
                proposals.append(("replace", ids[:pos]+[material]+ids[pos+1:], ds.copy()))
        proposals.append(("delete", ids[:pos]+ids[pos+1:], ds[:pos]+ds[pos+1:]))
    low, high = cfg["thickness_min_nm"], cfg["thickness_max_nm"]
    for pos in range(len(ids)+1):
        for material in range(count):
            proposals.append(("insert", ids[:pos]+[material]+ids[pos:],
                              ds[:pos]+[float(rng.uniform(low, max(low, min(high, 80.))))]+ds[pos:]))
    rng.shuffle(proposals)
    for operation, proposed_ids, proposed_ds in proposals:
        if legal_structure(proposed_ids, cfg, count):
            yield operation, list(zip(proposed_ids, project_thickness(proposed_ds, cfg)))


def discover_seeds(cfg, optics, design=None, limit=8, *, design_dir=None, scan_report=None):
    """Explicit directories use score-ranked top-k; legacy discovery keeps diversity."""
    if design is not None and design_dir is not None:
        raise ValueError("Choose design or design_dir, not both")
    if type(limit) is not int or limit < 1:
        raise ValueError("Seed limit must be a positive integer")
    if design_dir is not None:
        root = Path(design_dir).resolve()
        if not root.is_dir():
            raise ValueError(f"Design directory does not exist: {root}")
        paths = sorted(root.rglob("*.json"))
    else:
        paths = [Path(design)] if design is not None else sorted((PROJECT_DIR/"runs").rglob("best_*.json"))
    reports, seen = [], set()
    scan = scan_report if scan_report is not None else {}
    scan.update(files_scanned=len(paths), candidates_evaluated=0, duplicates_skipped=0, skipped=[])
    budget = cfg.get("max_total_thickness_nm")

    def skip(path, reason, index=None):
        scan["skipped"].append({"path": str(path.resolve()), "index": index, "reason": reason})
        if design is not None:
            raise ValueError(f"Incompatible design {path}: {reason}")

    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            skip(path, str(error))
            continue
        records = list(enumerate(data)) if isinstance(data, list) else [(None, data)]
        for index, record in records:
            try:
                if not isinstance(record, dict) or "layers" not in record:
                    raise ValueError("Not a design object with layers")
                if any(type(row["material_id"]) is not int for row in record["layers"]):
                    raise ValueError("Material IDs must be integers")
                layers = layers_of(record)
                ids = [i for i, _ in layers]
                if not legal_structure(ids, cfg, len(optics.names)):
                    raise ValueError("Material IDs, layer count or adjacency violate current constraints")
                if any(row.get("material") != optics.names[row["material_id"]] for row in record["layers"]):
                    raise ValueError("Material names/IDs do not match current material ordering")
                if budget is not None and sum(d for _, d in layers) > budget + 1e-9:
                    raise ValueError("Total thickness exceeds current budget")
                if any(not np.isfinite(d) or not cfg["thickness_min_nm"] <= d <= cfg["thickness_max_nm"]
                       for _, d in layers):
                    raise ValueError("Layer thickness violates current bounds")
            except (ValueError, KeyError, IndexError, TypeError) as error:
                skip(path, str(error), index)
                continue
            signature = tuple((i, d if design_dir is not None else round(d, 2)) for i, d in layers)
            if signature in seen:
                scan["duplicates_skipped"] += 1
                continue
            seen.add(signature)
            report = optics.evaluate(layers)  # Rank under the CURRENT data and objective, before the limit.
            report["seed_origin"] = str(path.resolve())
            if index is not None:
                report["seed_origin_index"] = index
            reports.append(report)
            scan["candidates_evaluated"] += 1
    reports.sort(key=lambda row: row["score"], reverse=True)
    if design_dir is not None:
        return reports[:limit]
    selected, structures = [], set()
    for report in reports:
        signature = tuple(i for i, _ in layers_of(report))
        if signature in structures:
            continue
        structures.add(signature)
        selected.append(report)
        if len(selected) >= limit:
            break
    # Fill remaining slots with thickness alternatives only after distinct structures.
    origins = {report["seed_origin"] for report in selected}
    for report in reports:
        if len(selected) >= limit:
            break
        if report["seed_origin"] not in origins:
            selected.append(report)
            origins.add(report["seed_origin"])
    return selected


def train_warm_start(cfg, best, directory, steps=1800):
    """Train VAC on the searched trajectory, then test actual autoregressive outputs.

    This stage is supervised elite imitation, not PPO and not a replacement for
    on-policy rollout GAE. The subsequent `train` command can use native PPO.
    """
    from .env import ThinFilmEnv
    from .hppo import build_model
    torch.manual_seed(cfg["training"]["seed"])
    model = build_model(cfg).cpu()
    env_cfg = deepcopy(cfg)
    env_cfg.pop("result_dir", None)
    env = ThinFilmEnv(env_cfg)
    obs = env.reset()
    observations, kinds, fractions = [], [], []
    for material, thickness in layers_of(best):
        observations.append(obs.copy())
        kinds.append(material)
        used = sum(d for _, d in env.layers)
        required = max(env.min_layers-len(env.layers)-1, 0)
        low, high = cfg["thickness_min_nm"], cfg["thickness_max_nm"]
        if env.max_total_thickness_nm is not None:
            high = max(low, min(high, env.max_total_thickness_nm-used-required*low))
        fraction = .5 if high == low else (thickness-low)/(high-low)
        fractions.append(fraction)
        args = np.zeros(env.material_count)
        args[material] = np.arctanh(np.clip(2*fraction-1, -.999999, .999999))
        timestep = env.step({"action_type": material, "action_args": args})
        obs = timestep.obs
    if not timestep.done:
        observations.append(obs.copy())
        kinds.append(env.EOS)
        fractions.append(.5)
    env.close()
    x = torch.as_tensor(np.asarray(observations))
    target_kind = torch.tensor(kinds)
    target_fraction = torch.tensor(fractions, dtype=torch.float32)
    active = target_kind < model.material_count
    row = torch.arange(len(x))
    selected = target_kind.clamp(max=model.material_count-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    history = []
    for step in range(steps):
        output = model(x, mode="compute_actor_critic")
        logits = output["logit"]
        ce = torch.nn.functional.cross_entropy(logits["action_type"], target_kind)
        predicted = (logits["action_args"]["mu"][row, selected].tanh()+1)/2
        mse = (predicted[active]-target_fraction[active]).square().mean()
        value_loss = (output["value"]-best["score"]).square().mean()
        loss = ce + 20*mse + .1*value_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % 100 == 0 or step == steps-1:
            history.append({"step": step+1, "loss": float(loss.detach()), "categorical_loss": float(ce.detach()),
                            "fraction_mse": float(mse.detach())})
    # Local exploration around learned thicknesses. PPO may subsequently adapt sigma.
    with torch.no_grad():
        model.actor_head[1].log_sigma_param.fill_(-4.)
    model.eval()
    validation = []
    generator = torch.Generator().manual_seed(cfg["training"]["seed"]+10000)
    for episode in range(33):
        env = ThinFilmEnv(env_cfg)
        obs = env.reset()
        with torch.no_grad():
            for _ in range(cfg["max_layers"]+1):
                output = model(torch.as_tensor(obs)[None], mode="compute_actor")["logit"]
                if episode == 0:
                    kind = int(output["action_type"].argmax(-1).item())
                    args = output["action_args"]["mu"][0]
                else:
                    kind = int(torch.multinomial(output["action_type"].softmax(-1), 1, generator=generator).item())
                    args = output["action_args"]["mu"][0] + output["action_args"]["sigma"][0]*torch.randn(
                        model.material_count, generator=generator)
                timestep = env.step({"action_type": kind, "action_args": args.numpy()})
                obs = timestep.obs
                if timestep.done:
                    break
        report = timestep.info
        if episode == 0:
            write_json(directory/"policy_deterministic_design.json", report)
        validation.append({"episode": episode, "deterministic": episode == 0,
                           "score": report["score"], "mean_absorption": report["mean_absorption"],
                           "layer_count": report["layer_count"]})
        env.close()
    checkpoint = {"model": model.state_dict(), "problem_fingerprint": problem_fingerprint(cfg),
                  "method": "supervised elite trajectory warm start after mixed structure/thickness search"}
    torch.save(checkpoint, directory/"warm_start_model.pt")
    report = {"method": checkpoint["method"], "steps": steps, "history": history,
              "validation": validation,
              "deterministic_absorption": validation[0]["mean_absorption"],
              "stochastic_mean_absorption": float(np.mean([r["mean_absorption"] for r in validation[1:]])),
              "stochastic_min_absorption": min(r["mean_absorption"] for r in validation[1:])}
    write_json(directory/"policy_validation.json", report)
    return report


def improve(cfg, output, design=None, *, design_dir=None, top_k=None):
    started = time.perf_counter()
    settings = {"seed": cfg["training"]["seed"], "target_score": .96,
                "seed_limit": 6, "structure_rounds": 6, "candidates_per_round": 24,
                "beam_width": 3, "max_thickness_iterations": 80, "warm_start_steps": 1800}
    settings.update(cfg.get("improvement", {}))
    if design is not None and design_dir is not None:
        raise ValueError("Choose design or design_dir, not both")
    if top_k is not None and (design_dir is None or type(top_k) is not int or top_k < 1):
        raise ValueError("top_k requires design_dir and must be a positive integer")
    if design_dir is not None:
        if not Path(design_dir).is_dir():
            raise ValueError(f"Design directory does not exist: {design_dir}")
        settings["seed_limit"] = 5 if top_k is None else top_k
    if not 0 < float(settings["target_score"]) <= 1:
        raise ValueError("improvement.target_score must be in (0, 1]")
    for key in ("seed_limit", "structure_rounds", "candidates_per_round", "beam_width",
                "max_thickness_iterations", "warm_start_steps"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < 0:
            raise ValueError(f"improvement.{key} must be a nonnegative integer")
    if min(settings[k] for k in ("seed_limit", "beam_width", "max_thickness_iterations")) < 1:
        raise ValueError("seed_limit, beam_width and max_thickness_iterations must be positive")
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"manifest.json").exists():
        raise FileExistsError("Choose a new improvement output directory; existing experiment is preserved")
    torch.set_num_threads(1)
    rng = np.random.default_rng(settings["seed"])
    optics = FilmOptics(cfg)
    source_files = sorted((PROJECT_DIR/"film_rl").glob("*.py"))
    manifest = {"started_at": datetime.now().astimezone().isoformat(), "settings": settings,
                "problem_fingerprint": problem_fingerprint(cfg),
                "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                "optical_files": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                  for p in optical_data_files(cfg)],
                "method": ("HPPO design seeds + structure mutation + exact TMM thickness gradients"
                           + (" + VAC imitation" if settings["warm_start_steps"] else "")),
                "pure_ppo": False,
                "seed_input": {"design": str(Path(design).resolve()) if design is not None else None,
                               "design_dir": str(Path(design_dir).resolve()) if design_dir is not None else None,
                               "selection": "current_score_top_k" if design_dir is not None else "legacy_diverse_seeds"}}
    write_json(directory/"config_snapshot.json", cfg)
    write_json(directory/"manifest.json", manifest)
    (directory/"source").mkdir(exist_ok=True)
    for source in source_files:
        shutil.copy2(source, directory/"source"/source.name)
    def log(event, **values):
        record = {"elapsed_seconds": time.perf_counter()-started, "event": event, **values}
        with (directory/"search.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False)+"\n")
        print(json.dumps(record, ensure_ascii=True), flush=True)
    scan = {}
    seeds = discover_seeds(cfg, optics, design, int(settings["seed_limit"]), design_dir=design_dir, scan_report=scan)
    write_json(directory/"seed_scan.json", scan)
    if not seeds and (design is not None or design_dir is not None):
        raise ValueError(f"No compatible designs in the specified input; see {directory/'seed_scan.json'}. "
                         "No random seeds were generated.")
    # Independent random starts are used if no matching HPPO archive is available.
    if not seeds:
        for length in (4, 5, 6, 8):
            if not cfg["min_layers"] <= length <= cfg["max_layers"]:
                continue
            ids = []
            for _ in range(length):
                choices = [i for i in range(len(optics.names)) if not ids or i != ids[-1]]
                ids.append(int(rng.choice(choices)))
            ds = project_thickness(rng.uniform(cfg["thickness_min_nm"], cfg["thickness_max_nm"], length), cfg)
            seeds.append(optics.evaluate(list(zip(ids, ds))))
    if not seeds:
        raise ValueError("No feasible seed designs")
    seeds.sort(key=lambda row: row["score"], reverse=True)
    write_json(directory/"seed_manifest.json", {"requested_count": settings["seed_limit"],
        "selected_count": len(seeds), "seeds": [
        {"rank": rank, "origin": row.get("seed_origin", "random"), "origin_index": row.get("seed_origin_index"),
         "score": row["score"], "layers": row["layer_count"],
         "materials": [layer["material"] for layer in row["layers"]],
         "target_errors": compact_design(row)["target_errors"]}
        for rank, row in enumerate(seeds, 1)]})
    write_json(directory/"initial_best.json", seeds[0])
    log("initial", score=seeds[0]["score"], seed_count=len(seeds), materials=optics.names)
    archive, seen = [], set()
    def evaluate_candidate(layers, origin):
        key = tuple((i, d if design_dir is not None else round(d, 3)) for i, d in layers)
        if key in seen:
            return
        seen.add(key)
        report = refine(optics, layers, int(settings["max_thickness_iterations"]))
        report["search_origin"] = origin
        archive.append(report)
        archive.sort(key=lambda r: r["score"], reverse=True)
        log("candidate", origin=origin, score=report["score"], best_score=archive[0]["score"],
            layers=len(layers), gradient_calls=report["optimization"]["gradient_calls"],
            optimizer_success=report["optimization"]["success"])
        write_json(directory/"best_design.json", archive[0])
        return report
    seed_results = []
    for rank, seed in enumerate(seeds, 1):
        report = evaluate_candidate(layers_of(seed), seed.get("seed_origin", "random"))
        if report is not None:
            report.update(seed_origin=seed.get("seed_origin", "random"))
            if "seed_origin_index" in seed:
                report["seed_origin_index"] = seed["seed_origin_index"]
            refined_path = directory/"refined_seeds"/f"seed{rank}.json"
            write_json(refined_path, report)
            seed_results.append({"rank": rank, "origin": report["seed_origin"],
                                 "origin_index": seed.get("seed_origin_index"),
                                 "initial_score": seed["score"], "refined_score": report["score"],
                                 "design": str(refined_path)})
    for round_index in range(int(settings["structure_rounds"])):
        if archive[0]["score"] >= settings["target_score"]:
            break
        beam, signatures = [], set()
        for report in archive:
            signature = tuple(i for i, _ in layers_of(report))
            if signature not in signatures:
                beam.append(report)
                signatures.add(signature)
            if len(beam) >= settings["beam_width"]:
                break
        generators = [neighbors(layers_of(parent), cfg, len(optics.names), rng) for parent in beam]
        for candidate in range(int(settings["candidates_per_round"])):
            generator = generators[candidate % len(generators)]
            try:
                operation, layers = next(generator)
            except StopIteration:
                continue
            evaluate_candidate(layers, f"round{round_index+1}:{operation}")
            if archive[0]["score"] >= settings["target_score"]:
                break
    best = optics.evaluate(layers_of(archive[0]))
    best["search_origin"] = archive[0]["search_origin"]
    write_json(directory/"best_design.json", best)
    policy = None
    if settings["warm_start_steps"]:
        log("warm_start_started", score=best["score"])
        policy = train_warm_start(cfg, best, directory, int(settings["warm_start_steps"]))
        log("warm_start_completed", deterministic_absorption=policy["deterministic_absorption"],
            stochastic_mean_absorption=policy["stochastic_mean_absorption"])
        warm_cfg = deepcopy(cfg)
        warm_cfg["training"].update(initial_model=str(directory/"warm_start_model.pt"),
                                    learning_rate=3e-6, entropy_weight=1e-4, epoch_per_collect=3,
                                    eval_freq=20, exp_name="runs/"+directory.name+"_ppo")
        # stop_value still refers to the evaluation mean; it is not a guarantee.
        warm_cfg["training"]["stop_value"] = min(float(cfg["training"]["stop_value"]),
                                                    float(settings["target_score"]))
        write_json(directory/"warm_start_config.json", warm_cfg)
    summary = {"initial_score": seeds[0]["score"], "best_score": best["score"],
               "mean_absorption": best["mean_absorption"], "target_score": settings["target_score"],
               "search_target_reached": best["score"] >= settings["target_score"],
               "absorption_above_95_percent": best["mean_absorption"] > .95,
               "elapsed_seconds": time.perf_counter()-started, "candidates": len(archive),
               "gradient_calls": sum(r["optimization"]["gradient_calls"] for r in archive),
               "tmm_design_evaluations": optics.evaluation_count,
               "method": manifest["method"], "pure_ppo": False,
               "policy": {k: v for k, v in policy.items() if k not in ("validation", "history")} if policy else None,
               "seed_refinements": seed_results}
    write_json(directory/"summary.json", summary)
    log("completed", **summary)
    return summary
