# FabCore full-fleet synthetic predictive maintenance

All readings, simulated maintenance cycles, failure events and training labels are **SYNTHETIC**. These are illustrative research models, not validated semiconductor tool specifications or industrial maintenance/safety evidence.

This batch trained the nine missing independent Random Forests and preserved CLN-101, CLN-102 and the latest seven-feature ETCH-101. The real MySQL inventory was read without writes; its twelve IDs/names/types match the source inventory. Existing per-machine prediction counts are unchanged.

## Twelve served models and synthetic held-out evaluation

Each row is the **actually served bundle**, not an unused candidate. All twelve `/models/{code}/schema` and `/predict/machine` requests returned HTTP 200 on the fresh integration service at port 18082. API status below describes that tested service; deployment of the existing port-8000 process still requires restart.

P = failure precision, R = recall, F1 = failure F1, PR-AUC = trapezoidal precision-recall area, AP = average precision. Thresholds are displayed rounded here; exact floats are saved and served. Every model has a 12-hour future target and seven required inputs.

| DB ID | Machine / type | Served version | P | R | F1 | PR-AUC | AP | Threshold | Approved model path | API schema/predict |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | CLN-101 / Cleaning | CLN101-SYN-RF-v1 | 0.3067 | 0.9194 | 0.4600 | 0.4460 | 0.4478 | 0.20 | `FabCore_CLN101_Training/cln101_model.joblib` | 200 / 200 |
| 2 | CLN-102 / Cleaning | CLN-102-SYN-RF-v2 | 0.4115 | 0.7821 | 0.5392 | 0.4506 | 0.4525 | 0.35 | `artifacts/CLN-102/model.joblib` | 200 / 200 |
| 3 | LTH-101 / Lithography | LTH-101-SYN-RF-v1 | 0.3409 | 0.7662 | 0.4718 | 0.4778 | 0.4789 | 0.35 | `artifacts/LTH-101/model.joblib` | 200 / 200 |
| 4 | LTH-102 / Lithography | LTH-102-SYN-RF-v1 | 0.3631 | 0.9213 | 0.5209 | 0.6147 | 0.6149 | 0.20 | `artifacts/LTH-102/model.joblib` | 200 / 200 |
| 5 | ETCH-101 / Etching | ETCH-101-SYN-RF-v2 | 0.3661 | 0.9118 | 0.5225 | 0.5226 | 0.5239 | 0.25 | `artifacts/ETCH-101/model.joblib` | 200 / 200 |
| 6 | ETCH-102 / Etching | ETCH-102-SYN-RF-v1 | 0.3235 | 0.6478 | 0.4315 | 0.4720 | 0.4726 | 0.20 | `artifacts/ETCH-102/model.joblib` | 200 / 200 |
| 7 | DEP-101 / Deposition | DEP-101-SYN-RF-v1 | 0.3626 | 0.8125 | 0.5014 | 0.5720 | 0.5726 | 0.25 | `artifacts/DEP-101/model.joblib` | 200 / 200 |
| 8 | DEP-102 / Deposition | DEP-102-SYN-RF-v1 | 0.3350 | 0.9259 | 0.4920 | 0.5342 | 0.5351 | 0.25 | `artifacts/DEP-102/model.joblib` | 200 / 200 |
| 9 | CMP-101 / Chemical Mechanical Polishing | CMP-101-SYN-RF-v1 | 0.4004 | 0.8241 | 0.5390 | 0.6035 | 0.6040 | 0.35 | `artifacts/CMP-101/model.joblib` | 200 / 200 |
| 10 | INSP-101 / Inspection | INSP-101-SYN-RF-v1 | 0.4143 | 0.9120 | 0.5698 | 0.5674 | 0.5684 | 0.30 | `artifacts/INSP-101/model.joblib` | 200 / 200 |
| 11 | MET-101 / Metrology | MET-101-SYN-RF-v1 | 0.3457 | 0.8866 | 0.4974 | 0.5086 | 0.5098 | 0.25 | `artifacts/MET-101/model.joblib` | 200 / 200 |
| 12 | PACK-101 / Packaging | PACK-101-SYN-RF-v1 | 0.3844 | 0.8889 | 0.5367 | 0.5825 | 0.5830 | 0.25 | `artifacts/PACK-101/model.joblib` | 200 / 200 |

CLN-101 deliberately retains `FabCore_CLN101_Training/cln101_model.joblib` (v1). The older `artifacts/CLN-101/model.joblib` v2 candidate and its evaluation were preserved but are **not** automatically deployed. `artifacts/CLN-101/serving_evaluation.json` evaluates the served v1 on its original synthetic split and matches its original report. CLN-102 and ETCH-101 retain their existing v2 bundles.

## Exact ordered inputs and units

| Machine | Required inputs in model order |
|---|---|
| CLN-101 | `chamber_temperature_c` (°C); `chamber_pressure_kpa` (kPa); `pump_vibration_mm_s` (mm/s); `chemical_flow_l_min` (L/min); `filter_diff_pressure_kpa` (kPa); `motor_current_a` (A); `hours_since_maintenance` (h) |
| CLN-102 | `chamber_temperature_c` (°C); `chamber_pressure_kpa` (kPa); `pump_vibration_mm_s` (mm/s); `chemical_flow_l_min` (L/min); `filter_diff_pressure_kpa` (kPa); `motor_current_a` (A); `hours_since_maintenance` (h) |
| LTH-101 | `exposure_dose_mj_cm2` (mJ/cm²); `focus_error_um` (µm); `stage_position_error_um` (µm); `illumination_intensity_mw_cm2` (mW/cm²); `lens_temperature_c` (°C); `stage_vibration_mm_s` (mm/s); `hours_since_maintenance` (h) |
| LTH-102 | `exposure_dose_mj_cm2` (mJ/cm²); `focus_error_um` (µm); `stage_position_error_um` (µm); `illumination_intensity_mw_cm2` (mW/cm²); `lens_temperature_c` (°C); `stage_vibration_mm_s` (mm/s); `hours_since_maintenance` (h) |
| ETCH-101 | `chamber_pressure_mtorr` (mTorr); `rf_forward_power_w` (W); `rf_reflected_power_w` (W); `process_gas_flow_sccm` (sccm); `electrode_temperature_c` (°C); `vacuum_pump_vibration_mm_s` (mm/s); `hours_since_maintenance` (h) |
| ETCH-102 | `chamber_pressure_mtorr` (mTorr); `rf_forward_power_w` (W); `rf_reflected_power_w` (W); `process_gas_flow_sccm` (sccm); `electrode_temperature_c` (°C); `vacuum_pump_vibration_mm_s` (mm/s); `hours_since_maintenance` (h) |
| DEP-101 | `chamber_pressure_pa` (Pa); `substrate_temperature_c` (°C); `precursor_flow_sccm` (sccm); `deposition_rate_nm_min` (nm/min); `rf_power_w` (W); `exhaust_pump_vibration_mm_s` (mm/s); `hours_since_maintenance` (h) |
| DEP-102 | `chamber_pressure_pa` (Pa); `substrate_temperature_c` (°C); `precursor_flow_sccm` (sccm); `deposition_rate_nm_min` (nm/min); `rf_power_w` (W); `exhaust_pump_vibration_mm_s` (mm/s); `hours_since_maintenance` (h) |
| CMP-101 | `platen_speed_rpm` (rpm); `downforce_kpa` (kPa); `slurry_flow_ml_min` (mL/min); `pad_temperature_c` (°C); `motor_current_a` (A); `platen_vibration_mm_s` (mm/s); `hours_since_maintenance` (h) |
| INSP-101 | `illumination_intensity_mw_cm2` (mW/cm²); `camera_temperature_c` (°C); `focus_error_um` (µm); `stage_position_error_um` (µm); `image_noise_dn` (DN); `stage_vibration_mm_s` (mm/s); `hours_since_maintenance` (h) |
| MET-101 | `laser_power_mw` (mW); `detector_noise_mv` (mV); `reference_drift_nm` (nm); `stage_repeatability_um` (µm); `enclosure_temperature_c` (°C); `interferometer_signal_db` (dB); `hours_since_maintenance` (h) |
| PACK-101 | `bond_head_temperature_c` (°C); `bonding_force_n` (N); `vacuum_pickup_kpa` (kPa); `alignment_error_um` (µm); `ultrasonic_power_w` (W); `head_vibration_mm_s` (mm/s); `hours_since_maintenance` (h) |

Units come from `training/sensors.py`, shared by generation/training/serving; no implicit Pa/kPa/mTorr conversions are made. `h` means elapsed hours since simulated maintenance. sccm is nominal standard cm³/min; DN is a simulated digital-number noise amplitude; dB is an illustrative relative signal level. No actual calibration, gas chemistry, standard-volume reference or sensor specification is claimed.

## Confusion matrices, class distributions and importance

Rows in each confusion matrix are actual class; columns are predicted class, order `[0,1]`. Counts are actual generated labels. All exact reports also contain both-class classification reports, cycle membership, units, dataset/model hashes and all feature importances.

| Machine | Confusion matrix `[[TN,FP],[FN,TP]]` | All labels (0 / 1) | Test labels (0 / 1) | Three largest feature importances |
|---|---|---:|---:|---|
| CLN-101 | `[[2192, 773], [30, 342]]` | 14398 / 1860 | 2965 / 372 | `pump_vibration_mm_s` 0.250; `filter_diff_pressure_kpa` 0.214; `hours_since_maintenance` 0.184 |
| CLN-102 | `[[3175, 349], [68, 244]]` | 19554 / 1452 | 3524 / 312 | `filter_diff_pressure_kpa` 0.308; `chemical_flow_l_min` 0.177; `pump_vibration_mm_s` 0.173 |
| LTH-101 | `[[2744, 640], [101, 331]]` | 16523 / 2124 | 3384 / 432 | `hours_since_maintenance` 0.286; `lens_temperature_c` 0.220; `focus_error_um` 0.144 |
| LTH-102 | `[[2392, 698], [34, 398]]` | 15657 / 2148 | 3090 / 432 | `hours_since_maintenance` 0.296; `lens_temperature_c` 0.206; `focus_error_um` 0.148 |
| ETCH-101 | `[[2921, 644], [36, 372]]` | 17834 / 2076 | 3565 / 408 | `rf_reflected_power_w` 0.302; `hours_since_maintenance` 0.267; `electrode_temperature_c` 0.168 |
| ETCH-102 | `[[3109, 504], [131, 241]]` | 18292 / 1944 | 3613 / 372 | `hours_since_maintenance` 0.310; `rf_reflected_power_w` 0.223; `electrode_temperature_c` 0.184 |
| DEP-101 | `[[2096, 617], [81, 351]]` | 14376 / 2148 | 2713 / 432 | `hours_since_maintenance` 0.278; `deposition_rate_nm_min` 0.189; `exhaust_pump_vibration_mm_s` 0.160 |
| DEP-102 | `[[2382, 794], [32, 400]]` | 14968 / 2124 | 3176 / 432 | `hours_since_maintenance` 0.232; `exhaust_pump_vibration_mm_s` 0.187; `chamber_pressure_pa` 0.166 |
| CMP-101 | `[[2401, 533], [76, 356]]` | 14267 / 2136 | 2934 / 432 | `hours_since_maintenance` 0.250; `pad_temperature_c` 0.216; `motor_current_a` 0.186 |
| INSP-101 | `[[2400, 557], [38, 394]]` | 14902 / 2160 | 2957 / 432 | `hours_since_maintenance` 0.285; `focus_error_um` 0.187; `stage_vibration_mm_s` 0.151 |
| MET-101 | `[[2835, 725], [49, 383]]` | 18035 / 2148 | 3560 / 432 | `reference_drift_nm` 0.222; `detector_noise_mv` 0.201; `hours_since_maintenance` 0.187 |
| PACK-101 | `[[2099, 615], [48, 384]]` | 13874 / 2160 | 2714 / 432 | `hours_since_maintenance` 0.255; `ultrasonic_power_w` 0.175; `head_vibration_mm_s` 0.161 |

Full per-feature scores, train/validation/test class counts and sample counts are saved in `artifacts/<CODE>/serving_evaluation.json` for all twelve, and `evaluation.json` for each newly trained model. No numbers were invented or selected to force different model predictions.

## Simulation families and assumptions

`training/equipment_profiles.py` contains every new family’s baselines, noise standard deviations, response vectors, drift distributions, budgets and failure-mode names. These are declared illustrative assumptions, fixed before seeing held-out test results. Same-type machines have distinct seeds, configurations and fitted forests.

For new non-etch families, each independent maintenance cycle samples planned duration, workload U(0.85,1.15), initial latent wear `a~U(0.03,0.12)` and `b~U(0.02,0.10)`, per-cycle drift, two Gamma stress budgets and small sensor offsets. Wear advances with workload and N(0,0.0015) noise. Thermal memory is `0.92*old + 0.08*(a+b) + N(0,0.005)`. Mean-one lognormal stress shocks have log standard deviation 0.30. A failure is the first cumulative stress to cross its independently sampled budget, not a direct threshold on the target or a future sensor reading.

| Family | Observable synthetic operating profile | First stress channel | Second stress channel |
|---|---|---|---|
| Lithography | Exposure dose, focus/stage errors, light intensity, lens temperature and vibration; optics wear reduces dose/intensity and increases drift | Illumination/optics: `0.10 + 3*a²` | Focus-stage: `0.05 + 1.8*b^1.5` |
| Deposition | Pressure, substrate temperature, precursor flow, deposition rate, RF power and pump vibration; coating buildup lowers flow/rate | Coating accumulation: `0.08 + 1.1*a` | Heater/exhaust: `0.06 + 4*b²` |
| Polishing | Platen speed, downforce, slurry flow, pad temperature, current and vibration; pad/slurry wear affects friction/load | Pad/slurry abrasion: `0.12 + 0.9*a*(1+b)` | Platen bearing: `0.07 + 3*b²` |
| Inspection | Light intensity, camera temperature, focus/stage errors, image noise and vibration | Camera thermal dose: `0.06 + 1.7*max(0,thermal)²` | Optical-stage: `0.04 + 2.4*b^1.5` |
| Metrology | Laser power, detector noise, reference drift, stage repeatability, enclosure temperature and signal level | Detector stress: `0.06 + 2.5*a²` | Reference instability: `0.05 + 3*a*b` |
| Packaging | Bond-head temperature, force, vacuum pickup differential, alignment error, ultrasonic power and vibration | Pickup seal fatigue: `0.08 + a*(1+b)` | Bond-head stress: `0.10 + 3.5*b²` |

These rates are multiplied by cycle workload and independent stress shocks before accumulation. Each sensor is its family baseline + response_a*a + response_b*b + cycle offset + hourly noise. Signed focus/reference errors are allowed; other simulated readings are floored above zero. This floor is not an equipment operating limit. The final feature is maintenance age, known at reading time, reset to zero in every maintenance/repair cycle.

ETCH-102 reuses the existing etching simulator, with its own seed, RF/pressure/gas setpoint ranges, wear drifts, budgets and independently trained forest. It does not reuse ETCH-101’s fitted model. The etching failure process integrates competing RF matching and vacuum-load stress; see `ETCH101.md` and the exact ETCH-102 JSON simulation metadata. The original cleaning and ETCH-101 simulations retain their random draws and behavior.

| New machine | Simulation seed | RF model seed | Simulation configuration |
|---|---:|---:|---|
| LTH-101 | 20101 | 20108 | `equipment-competing-stress-v1`, full parameters in `artifacts/LTH-101/synthetic_telemetry.metadata.json` |
| LTH-102 | 20102 | 20109 | `equipment-competing-stress-v1`, full parameters in `artifacts/LTH-102/synthetic_telemetry.metadata.json` |
| ETCH-102 | 10202 | 10209 | `etch-competing-stress-v2`, full parameters in `artifacts/ETCH-102/synthetic_telemetry.metadata.json` |
| DEP-101 | 30101 | 30108 | `equipment-competing-stress-v1`, full parameters in `artifacts/DEP-101/synthetic_telemetry.metadata.json` |
| DEP-102 | 30102 | 30109 | `equipment-competing-stress-v1`, full parameters in `artifacts/DEP-102/synthetic_telemetry.metadata.json` |
| CMP-101 | 40101 | 40108 | `equipment-competing-stress-v1`, full parameters in `artifacts/CMP-101/synthetic_telemetry.metadata.json` |
| INSP-101 | 50101 | 50108 | `equipment-competing-stress-v1`, full parameters in `artifacts/INSP-101/synthetic_telemetry.metadata.json` |
| MET-101 | 60101 | 60108 | `equipment-competing-stress-v1`, full parameters in `artifacts/MET-101/synthetic_telemetry.metadata.json` |
| PACK-101 | 70101 | 70108 | `equipment-competing-stress-v1`, full parameters in `artifacts/PACK-101/synthetic_telemetry.metadata.json` |

No telemetry is emitted at or after a simulated failure. A reading is positive only if the event is in `(t,t+12]`; maintenance-ended tails whose full future horizon is unknown are excluded for every newly generated model. Hidden wear, stress budgets, cumulative damage, event times/modes, planned duration, cycle IDs and raw hour indices are excluded from features. Only maintenance age (observable at reading time) is included. `synthetic_cycle_events.csv` is a separate audit, never merged into model inputs.

## Training and artifact preservation

The shared `generate_dataset.py` and `train_model.py` remain the only dataset/training entry points. `--machine CODE` selects one machine; `--all --skip-existing` safely handles the fleet without overwriting existing datasets/models. This pass used that safe batch command and trained only nine missing models. New models use 180 trees, minimum leaf size 5, balanced-subsample weights, per-machine model seeds and grouped 60/20/20 maintenance-cycle splits (108/36/36 cycles; split seeds 42 and 17). Validation alone selects maximum F2 over thresholds 0.10–0.80 in 0.05 steps, preferring the higher threshold on ties. No search was optimized against test metrics.

Each of the nine new `artifacts/<CODE>/` folders has `synthetic_telemetry.csv`, `synthetic_cycle_events.csv`, `synthetic_telemetry.metadata.json`, `model.joblib`, `model_metadata.json`, `evaluation.json`, `example_request.json`, `example_response.json`, and the extra `serving_evaluation.json`. New serving metadata/examples were added where missing for the three preserved models; none of their pre-existing artifacts were overwritten. Original legacy paths and reports remain intact. All twelve serving reports were recomputed from saved estimators, then checked against original saved precision/recall/AP values; no refitting occurred during export.

Backups are in `backups/full_fleet_before/`; its `manifest.json` records original hashes and locations for 50 files. `verification/full_fleet/preserved_artifacts.json` confirms all 24 pre-existing protected artifact/training-folder files unchanged, including the three working models and AI4I. Captured baseline predictions for those models match exactly. Database IDs, names, types and history counts match their read-only before/after inventory snapshots. Production MySQL rows were never inserted, updated or deleted.

## Dynamic integration and secure selection

`model_registry.py` accepts only canonical machine codes from the server-owned registry. Paths are explicit, approved project files; traversal and symlinks are rejected before deserialization. Bundle machine identity, exact feature order, model classes, simulation flag, label, threshold and available embedded metadata are validated. Missing/invalid models fail closed. Unknown valid code `UNKNOWN-999` returns “Custom model not available”; prediction never substitutes AI4I or another machine’s model.

FastAPI keeps `/predict` (explicit legacy AI4I), `/predict/cln101` and `/predict/cln102`. All twelve custom models use `GET /models/{code}/schema` and `POST /predict/machine`. Validation rejects missing/extra readings, booleans, numeric strings, nonfinite values and negative maintenance age. Invalid nonfinite inputs now return JSON-safe 422 responses.

C# continues to use JWT-protected `/api/machine-ai/{id}/schema` and `/api/machine-ai/{id}/predict/custom`. It resolves code from the actual database machine ID, fetches the schema, checks exact response metadata, and saves through the existing `MachineTelemetry` / `MachinePrediction` method. A small null-feature schema guard was added. No migration, machine renaming, JWT change or machine-specific route was needed.

React still generates every field and unit from the schema. `simulatedMachineSamples.json` supplies editable, explicitly simulated defaults for all twelve machines; the three previous defaults were retained. New defaults are actual first rows from the seeded synthetic datasets. Defaults are never described as live telemetry or adjusted to force different predictions. Test fixtures are generated from verified saved-model catalogs; they do not route production requests. Late schemas and histories from previous machine selections are ignored.

## Verification results and limits

- Python: **28 tests passed**, including all twelve schemas/predictions, exact saved-model probabilities, missing/invalid/nonfinite sensors, missing models, whitelisted paths, event audits/labels, resets, grouped isolation, validation-only thresholds, recomputed metrics, generator reproducibility, protected checksums and backward compatibility.
- Backend: **185 passed, 4 skipped**. The live test used real FastAPI models and a temporary C# HTTP/JWT server. It checked anonymous rejection for all twelve, logged in, submitted twelve requests, persisted 84 sensor rows and twelve new predictions, retained twelve pre-existing test-history rows, validated names/units/values/version/exact threshold/simulation/timestamps/links, and checked independent histories. A separate mock flow also submitted repeated predictions without history mixing.
- The four skipped tests require `FABCORE_TEST_MYSQL`. Persistence verification used isolated EF InMemory storage; **live MySQL prediction persistence was not tested**. MySQL was used only for the read-only real inventory/count check, with no production mutations.
- Targeted React: **61 tests passed** across all twelve dynamic forms, defaults, validation, stale responses, unknown models, retry and error handling.
- Production React build: **succeeded with the existing UsersPage.js missing-effect-dependency warning**. No dependencies were upgraded or lint rules disabled.
- Full React discovery: the machine-health suite passes; the pre-existing unrelated `App.test.js` still fails because Jest cannot resolve `react-router/dom`. This is an explicit remaining full-suite failure, not a claimed pass.
- Backend build emitted existing OpenAPI/MySqlConnector package advisories and existing nullable warnings in unrelated controllers. Raw warnings are retained in `backend_tests.log`.
- New models and CLN-102/ETCH-101 match runtime scikit-learn **1.6.1**. Legacy CLN-101 emits no mismatch warning. Preserved AI4I still reports **1.9.1 → 1.6.1** mismatch; it passed legacy prediction tests but compatibility is not guaranteed by that sample.
- Twelve independent network schema/prediction checks also returned **200/200**, with exact responses saved in `verification/full_fleet/http_results.json`.
- This is not a browser-driven or production-MySQL end-to-end deployment test; React component tests and real C#→FastAPI HTTP tests are the verified scopes.

## Exact commands executed and repeatable checks

Full command ledger: `verification/full_fleet/commands.txt`. The initial backups and read-only inventory probes used temporary scripts; their manifests/snapshots persist. Repeat the main pipeline from ManufacturingML:

```bash
.venv/bin/python generate_dataset.py --all --skip-existing
.venv/bin/python train_model.py --all --skip-existing
.venv/bin/python -m scripts.export_model_catalog
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 18082
```

From FabCore/backend while the test service is running:

```bash
FABCORE_TEST_AI_URL=http://127.0.0.1:18082 dotnet test FabCore.Tests/FabCore.Tests.csproj --no-restore --nologo
```

From FabCore/frontend/fabcore-web:

```bash
CI=true npm test -- --watchAll=false --runInBand --testPathPattern=MachineHealthPage
CI=false npm run build
CI=true npm test -- --watchAll=false --runInBand
```

The last command reports the known router discovery failure above. The production build runs in normal non-CI mode and reports its lint warning. Fixtures are stored in `backend/FabCore.Tests/AI/Fixtures/model_catalog.json` and `frontend/fabcore-web/src/Pages/__fixtures__/model_catalog.json`; refresh them from the exported catalog after any intentional model upgrade.

Example custom request (all other machines use the same route and their own example file):

```bash
curl --fail-with-body http://127.0.0.1:8000/models/PACK-101/schema
curl --fail-with-body -X POST http://127.0.0.1:8000/predict/machine \
  -H 'Content-Type: application/json' \
  --data-binary @artifacts/PACK-101/example_request.json
```

## Deployment / required restarts

The existing port-8000 process was intentionally left running; at verification it still returned `available:false` for LTH-101 because it has the old in-memory registry. **Restart Python FastAPI on port 8000** to load the twelve-model registry and validation handler. No generation/training is needed at restart:

```bash
# Stop the old uvicorn process in its owning terminal, then from ManufacturingML:
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

**Restart/redeploy the C# API only to load the added null-feature validation safeguard**. The existing dynamic controller already routes all models without hardcoded per-machine changes and fetches schemas on request. The active C# service itself was not restarted during tests.

The React development server hot-reloads the changed JSON defaults/import; refresh the browser after Python restart so it discards old schema state. For a static deployment, deploy the new successful `build/` output. Do not restart MySQL; no migration is needed. The temporary integration service at 18082 is stopped after verification.

## Changed files

ManufacturingML source: `api.py`, `generate_dataset.py`, `train_model.py`, `model_registry.py`, `training/config.py`, `training/sensors.py`, `training/etch_simulator.py`, new `training/equipment_profiles.py`, new `training/equipment_simulator.py`, `tests/test_model_registry.py`, new `tests/test_full_fleet.py`, new `scripts/export_model_catalog.py`, this guide and current-guide links in `PIPELINE.md` / `MACHINE_AI_INTEGRATION.md`.

FabCore backend: `FabCore.Api/Controllers/MachineAIController.Registry.cs` (null guard), `FabCore.Tests/AI/MachineAwareControllerTests.cs`, `MachineAwareHttpTests.cs`, new `ModelTestCatalog.cs`, new `AI/Fixtures/model_catalog.json`.

FabCore frontend: `src/Pages/MachineHealthPage.js`, `MachineHealthPage.test.js`, new `simulatedMachineSamples.json`, new `__fixtures__/model_catalog.json`. Production build output was regenerated.

Generated artifacts: the nine new machine folders, new serving evaluations for all twelve and missing metadata/examples for preserved machines. Backups and all raw logs/manifests/catalogs are under `backups/full_fleet_before/` and `verification/full_fleet/`. A machine-readable complete changed-file list is `verification/full_fleet/changed_files.json`.
