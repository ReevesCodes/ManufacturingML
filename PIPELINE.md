> Current twelve-machine guide: [FLEET_MODELS.md](FLEET_MODELS.md). The material below records the earlier integration; use `--all --skip-existing` to preserve trained artifacts.

# Multi-machine synthetic training

Run from the ManufacturingML project root using the existing environment:

```bash
.venv/bin/python -m pip install -r FabCore_CLN101_Training/requirements.txt
.venv/bin/python generate_dataset.py --all
.venv/bin/python train_model.py --all
.venv/bin/python -m unittest discover -s tests -v
```

For one machine (also accepts CLN-101):

```bash
.venv/bin/python generate_dataset.py --machine CLN-102
.venv/bin/python train_model.py --machine CLN-102
```

Generator option `--cycles 180` controls independent maintenance cycles. Selection is required; `--all` and `--machine` are mutually exclusive. Re-running overwrites only the selected machines' new artifacts. Paths are relative to the scripts, independent of terminal directory.

Each `artifacts/<machine>/` contains `synthetic_telemetry.csv`, `synthetic_telemetry.metadata.json`, `model.joblib`, and `evaluation.json`. The model bundle contains its features, threshold, machine code, synthetic flag, version and evaluation metadata. Load only trusted joblib files.

## Illustrative profiles

These profiles are invented simulation assumptions, **not actual CLN equipment measurements, specifications, or observed failures**. Both use hourly readings, seven common inputs, initial latent degradation uniform 0.025–0.18, per-cycle temperature offset N(0, 0.7), and hourly degradation noise N(0, 0.002). Configuration lives in `training/config.py`; add another profile there to extend `--all`.

| Parameter | CLN-101 | CLN-102 |
|---|---|---|
| Random seed | 42 | 102 |
| Planned cycle hours (upper bound excluded) | 95–169 | 110–189 |
| Degradation drift per hour (uniform) | 0.0032–0.0095 | 0.0025–0.0075 |
| Baseline temperature °C / pressure kPa / vibration mm/s | 35.5 / 100.5 / 0.9 | 39 / 104 / 1.1 |
| Baseline flow L/min / filter differential kPa / current A | 18.5 / 4 / 6.6 | 21 / 3.5 / 7.4 |
| Degradation sensitivities, same order | 10 / 7.2 / 4 / −6 / 9 / 4.3 | 8 / 5.5 / 3.2 / −8 / 12 / 3.6 |
| Failure hazard onset / slope | 0.48 / 0.24 | 0.55 / 0.30 |

Each sensor is baseline + sensitivity × latent degradation + Gaussian noise; noise standard deviations in table order are 1.4, 1.7, 0.38, 0.9, 0.85, 0.6. Vibration, flow, filter differential and current have a 0.1 floor. CLN-102 illustrates a warmer, higher-flow process with stronger flow/filter degradation effects and slower wear.

Each hour samples a stochastic failure with probability `clip((degradation - onset) * slope, 0, 0.16)`. Telemetry stops before failure. A reading is positive only if a sampled failure occurs in its next 12 hours. Otherwise it is negative only with a known full horizon; final 12 hours before planned maintenance without failure are excluded as censored. Neither latent degradation nor the event hour is an input. Synthetic negatives do not establish how a real machine would behave.

## Evaluation and preservation

Independent maintenance cycles are grouped into 60% train, 20% validation and 20% held-out test (split seeds 42 and 17). No cycle crosses partitions; exact cycle IDs are saved. Random Forest uses seed 42. Validation alone selects a threshold maximizing F2 on 0.10–0.80 in 0.05 steps, with ties choosing the higher threshold. Test data does not select the model or threshold. Reports save both-class classification reports, precision, recall, average precision, feature importance and confusion matrices (rows actual, columns predicted, order 0 then 1), dataset SHA-256, simulation parameters and package versions. Identical seeds and package versions reproduce datasets and predictions; training timestamps change.

All scores evaluate synthetic data from the same simulator. They provide no evidence of field performance or cross-machine generalization. Different profiles do not guarantee different predictions, and no prediction is adjusted to manufacture a difference.

Existing `api.py`, `/`, `/predict`, `/predict/cln101`, `manufacturing_model.joblib`, the legacy CLN-101 model and all three original numbered scripts are preserved. The API continues serving its existing models; new training artifacts are deliberately separate and are not automatically deployed. The legacy scripts still use their legacy paths; use the root commands above for this pipeline. `03_try_prediction.py` continues demonstrating the preserved CLN-101 model.
