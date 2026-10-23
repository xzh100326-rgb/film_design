# Design JSON output

New training best designs, improve design exports, and check/evaluate reports use `design_format: compact_v1` by default.

Order:

1. `layer_count`, `total_thickness_nm`, `score`, `reward`, `mae`.
2. `target_errors`: one item for each requested wavelength band, with `quantity`, `target`, `mean_actual`, `mae`, `min_actual`, `max_actual`, and band means `mean_R`, `mean_T`, `mean_A`.
3. Sampling range/count, material source files/hashes, substrate and other small metadata.
4. `layers`: the exact material order and thicknesses, at the end.

All optical values are fractions: 0.95 means 95%. `mean_actual` refers to that item's requested quantity, e.g. R for the incandescent task. It averages over sampled angles and wavelengths in that band. The band bounds, weights and score are unchanged. A target that varies with wavelength/angle is explicitly labeled `varying_spectrum` and summarized by min/max/mean; its exact definition remains in the experiment config.

The export omits dense `R`/`T`/`A`, target arrays and n/k arrays. `FilmOptics.evaluate()` still provides complete arrays internally: TMM, rewards and PPO behavior are unchanged.

To recompute a complete spectral report from a compact design, use the same original config and material files:

```powershell
python -m film_rl evaluate --config configs/incandescent_task.json --design runs/incandescent_task2__hppo__terminal/seed0/designs/best_YOUR_ID.json --output outputs/full_spectrum.json --full-spectrum
```

The flag is optional. Without it, evaluate writes the compact format. `check` also supports `--full-spectrum`.

To compact existing results (each original gets an exact backup first):

```powershell
python tools/compact_designs.py --input runs/incandescent_task2__hppo__terminal/seed0/designs
```

Backups and a conversion manifest are stored under `outputs/design_format_backups/<timestamp>/`. Files detected as changing during conversion are skipped. This tool does not start or stop training. A running Python process may still use its already imported old exporter; new training processes use the changed exporter.

Existing consumers that read `layers`, `score`, `evaluation_metric`, or `target_errors` continue to work, including evaluate, improve, batch summaries and plot_training. Scripts that plot dense arrays directly from design JSON must first use evaluate --full-spectrum to obtain them. Historical full-spectrum files and original experiment snapshots are still accepted.

When a spectral target has active `bounds.min` or `bounds.max`, compact exports also retain
`has_constraints`, `base_reward`, `constraint_penalty`, `constraints_satisfied`,
`constraint_violations`, and `constraint_tolerance`. Each constrained `target_errors` item
includes a `bounds` summary. `score` remains the original target-matching score;
`reward = base_reward - constraint_penalty` may be negative. Bounded design selection
prefers designs satisfying all configured bounds, then compares penalized reward.
The evaluation metric is `penalized_reward`, never a transmittance percentage.
Bounds apply to the configured sample grid and scope; they do not certify unsampled
wavelengths/angles. Omitting bounds preserves historical reward and selection behavior.
