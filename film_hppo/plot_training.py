"""Export existing TensorBoard scalars to a PNG and CSV without training."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/duv193_hppo_shapesafe"))
    parser.add_argument("--output", type=Path, default=Path("outputs/training_curves.png"))
    metric_options = parser.add_mutually_exclusive_group()
    metric_options.add_argument("--reward-as-transmission", action="store_true",
                        help="Only for target T=1; plot the evaluator's untransformed score as percent transmission")
    metric_options.add_argument("--evaluation-metric", choices=["score", "transmission", "reflection", "absorption", "penalized_reward"],
                                help="Default: infer the metric from saved designs; mixed targets use score")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent
    if not args.run.is_absolute():
        args.run = project_dir / args.run
    if not args.output.is_absolute():
        args.output = project_dir / args.output
    inferred_metric = None
    for path in sorted((args.run / "designs").glob("best_*.json")):
        try:
            candidate = json.loads(path.read_text(encoding="utf-8-sig")).get("evaluation_metric")
        except (OSError, ValueError):
            continue
        if candidate in ("score", "transmission", "reflection", "absorption", "penalized_reward"):
            inferred_metric = candidate
            break
    metric = "transmission" if args.reward_as_transmission else args.evaluation_metric or inferred_metric or "score"
    if inferred_metric and metric != "score" and metric != inferred_metric:
        parser.error(f"This run evaluates {inferred_metric}; cannot label its score as {metric}")
    metric_title = {"score": "Target matching score (1 - MAE)",
                    "transmission": "Target-band transmittance (%)",
                    "reflection": "Target-band reflectance (%)",
                    "absorption": "Target-band absorptance (%)",
                    "penalized_reward": "Reward after spectral-bound penalties"}[metric]
    log_dir = args.run / "log" / "serial"
    if not log_dir.is_dir() or not any(log_dir.glob("*tfevents*")):
        runs_dir = project_dir / "runs"
        available = sorted(p.name for p in runs_dir.iterdir()
                           if p.is_dir() and any((p / "log" / "serial").glob("*tfevents*"))) if runs_dir.is_dir() else []
        parser.error(f"TensorBoard event directory missing or empty: {log_dir}\n"
                     f"Available experiments in {runs_dir}: {', '.join(available) or '(none)'}\n"
                     "Use the exact training.exp_name as --run; check spelling and underscores.")
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    if not tags:
        parser.error(f"No scalar metrics recorded yet in {log_dir}")
    panels = [
        (metric_title,
         [("evaluator_iter/reward_mean", "Mean"),
          ("evaluator_iter/reward_max", "Best in evaluation")], 1 if metric in ("score", "penalized_reward") else 100),
        ("Total loss", [("learner_iter/total_loss_avg", "Total")], 1),
        ("Policy loss", [("learner_iter/policy_loss_avg", "Policy")], 1),
        ("Value loss", [("learner_iter/value_loss_avg", "Value")], 1),
        ("Approximate KL (native loss metric)", [("learner_iter/approx_kl_avg", "Approx. KL")], 1),
        ("Average episode steps (including EOS)", [("collector_iter/avg_envstep_per_episode", "Collection")], 1),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    rows = []
    for axis, (title, series, scale) in zip(axes.flat, panels):
        for tag, label in series:
            if tag not in tags:
                continue
            events = accumulator.Scalars(tag)
            axis.plot([e.step for e in events], [e.value * scale for e in events], label=label, linewidth=1.3)
            rows.extend((tag, e.step, e.wall_time, e.value) for e in events)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Optimizer updates")
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(fontsize=8)
        else:
            axis.text(.5, .5, "No data yet", ha="center", transform=axis.transAxes)
    figure.suptitle(f"{args.run.name} | raw logged metrics | snapshot", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tag", "optimizer_step", "wall_time", "value"])
        writer.writerows(rows)
    print(args.output.resolve())
    print(args.output.with_suffix(".csv").resolve())


if __name__ == "__main__":
    main()
