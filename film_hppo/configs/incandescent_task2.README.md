# OML-PPO Task 2 implemented with film_hppo

Run from `E:\lithography_machine_lens\film_hppo` in the yolov11 environment:

```powershell
python -m film_rl check --config configs/incandescent_task2.json --output outputs/incandescent_task2_check.json
python -m film_rl train --config configs/incandescent_task2.json
```

Creating and checking this configuration does not start training. The train command above does.

## Source and confirmed settings

- Wang et al., Machine Learning: Science and Technology 2 (2021) 025013, section 4.2 and Tables 3-4: https://doi.org/10.1088/2632-2153/abc327
- OML-PPO optical data: https://github.com/hammer-wang/oml-ppo/tree/ef63afe85e9c09cb3ccf0df01218c6264535ac9c/data
- Target reflection: zero within 480-700 nm and one outside. Low reflection is the paper's target; no additional T or A target is added.
- Paper says seven materials but names only six in the sentence. Its Table 3 includes SiN, represented here by the repository's Si3N4 dataset.
- The air/stack/air boundary follows the incandescent-filter benchmark. No glass or metal substrate is used.
- Maximum layers: 45; terminal-only reward; gamma=1; learning rate 5e-5; 3000 collected transitions per collection.

## Numerical choices and reproduction limits

The checked paper and public repository do not provide the exact Task 2 training wavelength/angle grids. This JSON is runnable but is not a claim of exact numerical reproduction.

- Figure 7 spans approximately 300-3000 nm. This configuration uses 271 points every 10 nm, an implementation choice.
- Paper describes an average over 0-90 degrees. The executable grid is 0-89 degrees every 1 degree, approximating that range while excluding exact grazing incidence. `film_hppo` rejects 90 degrees. Original angle step was not found.
- Each angle receives equal weight in training, following the discrete MAE structure of Eq. (1). The hemispherical cos(theta)*sin(theta) integral used for thermal emission and chi is a different postprocessing quantity and is not implemented by this JSON.
- Unpolarized light uses the mean of s/p reflectances, then the error is calculated. This choice is explicit; the exact original polarization treatment in Task 2 was not verified.
- Single-layer thickness is continuous 15-400 nm, which contains the published 42-layer design (95-289 nm per layer). This is an HPPO search-domain choice, not a claimed original Task 2 thickness grid. No total thickness budget is applied; the published stack is about 8.54 um thick.
- Existing HPPO/VAC architecture and linear material interpolation are retained. Original OML-PPO uses a GRU, discrete thickness sampling and quadratic material interpolation. No model checkpoint is loaded.
- `max_train_iter=10000` means DI-engine iterations, not a verified equivalent of 10000 paper epochs. `max_env_step` is an additional operational cap.

## Target weights

The active JSON uses band weights 1:4:3. Its generic S-spectrum reward is `1 - sum_b(normalized_weight_b * mean(abs(S_b - S_target_b)))`, with S=R for all three terms in this task. Other tasks may select S=T or S=A through the same `quantity` field and reward implementation.

The 271 wavelength samples split into 18 UV samples (300-470 nm), 23 transmission-window samples (480-700 nm), and 230 IR samples (710-3000 nm). If uniform averaging across all 271 wavelength samples is desired, set weights to 18/23/230. These are NOT the active 1:4:3 weights. Setting all weights to 1 instead gives each band equal total importance.

When changing wavelength resolution or band boundaries, update both bands and weights. At 10 nm spacing, the intervals between 470/480 and 700/710 contain no samples and therefore leave no unscored grid points.

Training reward and evaluator score are both `1 - MAE(R, target_R)` (`reward_alpha=0`). Neither reward nor stop_value denotes mean transmission or the visible-light enhancement factor 16.60. Computing chi additionally needs the emitter's thermal balance, view factor and visual-response curve.

## Data provenance

All seven files are pinned to `OML-PPO.yml` versions converted directly from the authors' root CSV files. Original constant extensions and sparse intervals remain present; numerical coverage does not establish measured data validity across the whole band. Existing absorption and DUV configurations are unchanged.
