# ETCH-101 synthetic training and serving

**Every measurement, operating assumption, failure event and label here is synthetic.** No real semiconductor equipment data is used. These values are not validated ETCH-101 specifications, recipes, safety limits or predictive-maintenance evidence.

## Exact model input schema

`training/sensors.py` supplies the ordered schema to both training and the registry. ETCH-101 v2 has seven inputs: the six original etching measurements plus observable maintenance age. Cleaning-specific measurements are not used.

| Feature | Simulated unit | Illustrative generation assumption |
|---|---|---|
| `chamber_pressure_mtorr` | mTorr | Per-cycle setpoint U(30,60), multiplied by (1 + 0.4 × exhaust restriction), plus N(0,1.3) noise; floor 0.1 |
| `rf_forward_power_w` | W | Per-cycle setpoint U(1050,1550), plus N(0,25) hourly noise; floor 1 |
| `rf_reflected_power_w` | W | Forward power × (0.015 + 0.55 × matching wear²), plus N(0,4); clipped to [0, forward power] |
| `process_gas_flow_sccm` | sccm | Per-cycle setpoint U(70,125), multiplied by (1 − 0.18 × exhaust restriction), plus N(0,2); floor 0.1 |
| `electrode_temperature_c` | °C | Initial N(50,2); each hour 0.92 × previous + 0.08 × (45 + 12 × RF load + 30 × matching wear + 7 × restriction), plus N(0,0.5) |
| `vacuum_pump_vibration_mm_s` | mm/s | 0.7 + 3.6 × restriction + 0.15 × gas load, plus N(0,0.15); floor 0.1 |
| `hours_since_maintenance` | h (hours) | Elapsed hourly time in the current cycle; starts at 0 after each maintenance reset, then increases by 1 per reading |

N(mean, standard deviation) denotes Gaussian noise; U is uniform. sccm denotes nominal standard cm³/min in this illustrative simulator; no actual gas chemistry, calibration or plasma-physics model is asserted. Values are rounded to three decimal places. ETCH pressure is in **mTorr**, unlike the cleaning schema's **kPa**; no implicit conversion or shared field is used.

## Distinct etching failure process

The generator delegates ETCH-101 to `training/etch_simulator.py`, version `etch-competing-stress-v2`. Seed **10101** initializes a fresh NumPy generator on every run. The existing cleaning simulation is unchanged.

Each independent post-maintenance cycle has a planned duration sampled from integers 120–239 hours and its own recipe setpoints. Matching wear starts U(0.03,0.10); exhaust restriction starts U(0.02,0.12). Hourly matching drift is sampled once per cycle from U(0.0015,0.0045), with hourly N(0,0.002) noise. Restriction drift is U(0.0010,0.0030), with hourly N(0,0.0015) noise. Both states are floored at zero.

Two competing hidden damage processes accumulate:

- RF damage increment = RF load × (0.12 + 4 × matching wear²) × independent lognormal shock.
- Vacuum damage increment = gas load × (0.10 + 2.2 × restriction²) × independent lognormal shock.

RF load = RF setpoint / 1300; gas load = gas setpoint / 100. Shocks have log standard deviation 0.25 and log mean −0.25²/2, giving mean-one multiplicative noise. Each cycle samples independent RF and vacuum stress budgets from Gamma(shape=9, scale=5) and Gamma(shape=10, scale=4). The first accumulated damage to cross its budget triggers a simulated trip. If both cross in the same hour, the audit records RF first. This is a cumulative competing-stress process, **not the cleaning generator's degradation-dependent stochastic hazard**.

No telemetry is emitted at or after a trip. `failure_within_12h = 1` exactly when a future simulated event occurs in `(reading hour, reading hour + 12]`. Planned-maintenance cycles have no observed failure; readings whose future window reaches or exceeds maintenance are omitted, rather than falsely labelled negative. The dataset has 151 RF trips, 22 vacuum trips and seven maintenance-terminated cycles.

The separate `synthetic_cycle_events.csv` records event hour, mode and planned maintenance for label auditing only. Its SHA-256 is stored in dataset metadata. Event audit fields, cycle IDs, raw hour indices, latent wear, accumulated damage, budgets and targets are excluded. The explicitly named `hours_since_maintenance` feature equals elapsed cycle hours: it is known at reading time, resets after maintenance (including repair following a trip), and never uses future failure time. Grouping by cycle prevents overlapping future-label windows from crossing training/evaluation partitions. Adding age does not consume random draws: all six original sensor streams, cycle events and labels are unchanged. Tests reconstruct every label from the audit and verify censored tails and no readings after failure.

## Training and synthetic evaluation

The shared `train_model.py` now chooses its schema by machine. It uses 180 Random Forest trees, balanced-subsample weights, minimum leaf size five and model seed 42. Cycle-grouped splits use seeds 42 and 17: **108 training / 36 validation / 36 test cycles**. All readings from a cycle stay in one partition. Validation alone chooses the highest-F2 threshold from 0.10–0.80 in 0.05 increments, preferring the higher threshold on ties; the final test does not select parameters or threshold.

| Split | Negative labels | Positive labels | Total |
|---|---:|---:|---:|
| Training | 10,464 | 1,248 | 11,712 |
| Validation | 3,805 | 420 | 4,225 |
| Held-out test | 3,565 | 408 | 3,973 |
| All | 17,834 | 2,076 | 19,910 |

Held-out synthetic results at threshold **0.25** (exact saved float `0.25000000000000006`):

| Failure-class metric | Value |
|---|---:|
| Precision | 0.366142 |
| Recall | 0.911765 |
| F1 | 0.522472 |
| PR-AUC, trapezoidal | 0.522633 |
| Average precision, step-integrated PR summary | 0.523919 |

Confusion matrix, rows actual / columns predicted, label order `[0,1]`:

```text
[[2921, 644],
 [  36, 372]]
```

There are 644 false alarms and 36 missed simulated failures in this test split. Scores evaluate only held-out cycles from the same simulator; they establish no field performance or cross-machine generalization. Predictions are not altered to force machine differences.

`artifacts/ETCH-101/` contains:

- `synthetic_telemetry.csv`: hourly synthetic readings and retrospective labels.
- `synthetic_cycle_events.csv`: synthetic label-audit events, excluded from features.
- `synthetic_telemetry.metadata.json`: seed, simulation profile, horizon, synthetic origin and event-audit checksum.
- `model.joblib`: fitted estimator, exact feature order, exact threshold, synthetic flag, version `ETCH-101-SYN-RF-v2` and evaluation metadata.
- `evaluation.json`: classification report, both PR summaries, precision/recall/F1, confusion matrix, class counts, exact threshold, cycle membership, dataset SHA-256, units and package versions.
- `example_request.json` / `example_response.json`: the first synthetic dataset reading and its successful API result. The example is not intended to guarantee a failure prediction.

## Exact commands

From the ManufacturingML root, using the existing environment:

```bash
.venv/bin/python generate_dataset.py --machine ETCH-101
.venv/bin/python train_model.py --machine ETCH-101
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

From another terminal in the same folder:

```bash
curl --fail-with-body http://127.0.0.1:8000/models/ETCH-101/schema
curl --fail-with-body -X POST http://127.0.0.1:8000/predict/machine \
  -H 'Content-Type: application/json' \
  --data-binary @artifacts/ETCH-101/example_request.json
```

The example returns HTTP 200 with `machineCode: "ETCH-101"`, `failurePrediction: 0`, `failureProbability: 0.0`, `isSimulated: true`, horizon 12, the saved threshold and `ETCH-101-SYN-RF-v2` version. The schema route also returns HTTP 200, `available: true`, the seven ordered input definitions and their units. Both routes were exercised through the FastAPI ASGI application in tests. Restart a running FastAPI process after installing the updated registry; its artifacts are cached.

`--all` now includes CLN-101, CLN-102 and ETCH-101. The ETCH-only commands above leave the cleaning artifacts untouched. Generator and model predictions are reproducible for fixed package versions and seeds; training timestamps change.

## Preservation, tests and compatibility

**21 Python tests passed**. The suite checks all seven ordered features, maintenance age starting at zero and increasing consistently in every cycle, its required `h` unit and nonnegative minimum, missing/negative input rejection, exact saved-model prediction agreement, grouped membership, independently recomputed metrics, event labels, censoring, reproducibility, wrong-machine schemas and approved artifact paths. Cleaning simulations remain byte-identical for seed-controlled fixtures.

The original six ETCH sensor columns, cycle IDs, hour indices, labels and event audit were compared with the previous dataset and are unchanged. SHA-256 checks confirmed all **17 pre-existing** cleaning/AI4I model, data, report and training-folder files unchanged. Captured CLN-101, CLN-102 and generic AI4I predictions matched their baselines. No production history data or migrations were changed.

The existing `/predict/machine` route and dynamic registry now expose the seven-feature schema automatically; `api.py` and `model_registry.py` need no special ETCH route or code change. Missing maintenance age returns 422 from FastAPI; C# rejects it with 400 before prediction/persistence.

The FabCore simulated ETCH example now includes `hours_since_maintenance: 0`, matching the first synthetic dataset reading. React continues to render server-provided fields and uses the shared custom-prediction route. Its tests check all seven inputs, editable simulated defaults, hours submission and missing-value rejection.

The backend live test now checks ETCH v2 through real C# HTTP, JWT and FastAPI. It checks missing/negative age requests do not persist, seven telemetry values including `Hours since maintenance` in `h`, model version, exact threshold, linked samples, separate histories for all three custom machines, and retention of existing CLN and ETCH v1 history rows. **180 backend tests passed; four MySQL tests were skipped** because `FABCORE_TEST_MYSQL` is not configured. Persistence checks use isolated EF InMemory storage; this does not claim live MySQL verification.

The focused frontend page suite has **13 passing tests**. Full frontend discovery still fails in the unrelated `App.test.js` because Jest cannot resolve `react-router/dom`; the machine-health suite passes. The production build succeeds with the existing `UsersPage.js` effect-dependency warning (CI treats that warning as an error).

ETCH-101 and CLN-102 use scikit-learn **1.6.1**, matching runtime. Legacy CLN-101 emitted no mismatch warning. AI4I retains its existing **1.9.1 → 1.6.1** warning. Dependency versions were not changed.

### Exact FabCore verification commands

With a temporary FastAPI test service running:

```bash
# ManufacturingML terminal
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 18081

# FabCore/backend terminal
FABCORE_TEST_AI_URL=http://127.0.0.1:18081 dotnet test FabCore.Tests/FabCore.Tests.csproj --no-restore --nologo

# FabCore/frontend/fabcore-web terminal
CI=true npm test -- --watchAll=false --runInBand
CI=true npm test -- --watchAll=false --runInBand --testPathPattern=MachineHealthPage
CI=false npm run build
```

The full frontend command reports the existing router test failure described above; the focused command passes. The temporary test service is stopped after verification.

### Services and refresh

At verification, the existing FastAPI service on **127.0.0.1:8000** still returned **v1 / six inputs / threshold 0.35**. Restart it to load **v2 / seven inputs / threshold 0.25**. The new model files are already saved on disk; no retraining is needed at restart.

The C# service on **5087** requires no restart: production backend code is unchanged and it obtains schemas on request. The React development server on **3000** can hot-reload the changed sample; refresh the browser after restarting FastAPI to fetch its new schema and reset stale form state. If serving a static React deployment, rebuild/deploy the updated frontend bundle. MySQL requires no restart or migration.

### Changed files in this update

ManufacturingML: `training/etch_simulator.py`, `training/sensors.py`, `training/config.py`, `generate_dataset.py` (age metadata), `train_model.py` (new model version), `tests/test_etch_pipeline.py`, this document, and the model/dataset/metadata/report/examples under `artifacts/ETCH-101/`. The event audit was regenerated identically.

FabCore: `frontend/fabcore-web/src/Pages/MachineHealthPage.js`, `MachineHealthPage.test.js`, and `backend/FabCore.Tests/AI/MachineAwareHttpTests.cs`. Production C# controllers, JWT and persistence code remain unchanged.
