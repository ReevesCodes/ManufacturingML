> Current twelve-machine guide: [FLEET_MODELS.md](FLEET_MODELS.md). The material below records the earlier integration; use `--all --skip-existing` to preserve trained artifacts.

# Machine-aware predictive maintenance integration

The custom-model flow is React → JWT-protected FabCore C# controller → FastAPI registry → selected model → existing telemetry/prediction persistence. No database migrations, production data updates, model retraining or artifact overwrites are part of this change.

## Registry and compatibility

`model_registry.py` is the server-owned allowlist. CLN-101 retains the existing loader's ordered paths: root `cln101_model.joblib`, then `FabCore_CLN101_Training/cln101_model.joblib`, then its historical case variant. In this workspace it serves **CLN101-SYN-RF-v1**, threshold **0.20000000000000004**. It does not silently adopt `artifacts/CLN-101/model.joblib`.

CLN-102 uses only `artifacts/CLN-102/model.joblib`, version **CLN-102-SYN-RF-v2**, threshold **0.3500000000000001**. Feature order, thresholds and versions come from saved bundles; labels and units retain the existing persisted telemetry names. Identity, estimator features/classes, label, threshold and available embedded metadata are validated before use. Request data cannot select paths; symlinks and resolved paths outside the approved files are rejected before deserialization. Artifacts must remain trusted and operator-managed. Restart FastAPI after intentionally deploying a model because loaded bundles are cached.

The registry contains feature names, labels, measurement units and approved artifact paths; its loaded descriptors add the exact saved threshold and version. Add a supported machine with an explicit `ModelEntry`, approved paths and feature definitions; no machine-code branches are needed in C# or React. Units are simulation assumptions, not operating specifications. Training remains the existing seed-controlled grouped pipeline; model deployment remains explicit.

Legacy Python `/predict`, `/predict/cln101` and `/predict/cln102`, C# `/api/machine-ai/{id}/predict`, `/predict/cln101`, `/predict/cln102`, and per-machine `/predictions` remain available with their existing contracts. AI4I remains an explicit legacy endpoint and is never a fallback for a missing custom model.

## New contracts

FastAPI:

- `GET /models/CLN-101/schema` (or CLN-102): `machineCode`, `available`, `isSimulated`, `modelVersion`, `decisionThreshold`, `predictionHorizonHours`, and ordered `features` with `name`, `label`, `unit`, `type`, `required`, `minimum`.
- Unknown valid machine codes return `available: false`, `message: "Custom model not available"` and no features. Unavailable/corrupt registered artifacts fail closed; no other model is substituted.
- `POST /predict/machine`: canonical validated machine code and exactly its required finite numeric readings. Missing/extra readings, booleans, numeric strings and negative maintenance hours are rejected. Result identifies the machine, version, exact threshold, 12-hour horizon and `isSimulated: true`.

```json
{
  "machine_code": "CLN-102",
  "readings": {
    "chamber_temperature_c": 41,
    "chamber_pressure_kpa": 105,
    "pump_vibration_mm_s": 1.8,
    "chemical_flow_l_min": 19,
    "filter_diff_pressure_kpa": 6,
    "motor_current_a": 8,
    "hours_since_maintenance": 24
  }
}
```

C# adds authenticated `GET /api/machine-ai/{id}/schema` and `POST /api/machine-ai/{id}/predict/custom`. The latter accepts only `{ "readings": { ... } }`. C# gets the machine code from the persisted record, validates the fetched schema, uses `/predict/machine`, verifies returned identity/threshold/version/horizon/simulation metadata against that schema, then calls the existing save method. Failed requests do not save telemetry or predictions.

React fetches this schema for the selected database machine ID, generates inputs and units dynamically and starts inputs blank. It retains inputs separately by machine ID, disables unsupported predictions, labels simulated outputs and retains the history view even for machines without custom models. Stale schema/history responses from previous selections are ignored. The existing Axios JWT interceptor is unchanged.

## Run and verify

From ManufacturingML:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

From `../FabCore/backend` with FastAPI running:

```bash
FABCORE_TEST_AI_URL=http://127.0.0.1:8000 dotnet test FabCore.Tests/FabCore.Tests.csproj --no-restore --nologo
```

The opt-in live test starts a temporary C# HTTP server, logs in through the real JWT flow, rejects anonymous requests, obtains both real FastAPI model schemas, submits predictions, checks linked telemetry and independent histories, preserves an existing history record and rejects unsupported custom models. It uses isolated EF InMemory persistence, not production MySQL. Normal tests skip this test unless `FABCORE_TEST_AI_URL` is set. Existing MySQL concurrency tests require `FABCORE_TEST_MYSQL`; no live MySQL verification is claimed without that configuration.

From `../FabCore/frontend/fabcore-web`:

```bash
CI=true npm test -- --watchAll=false --runInBand --testPathPattern=MachineHealthPage
npm run build
```

## Validation and existing limitations

- Python registry tests cover both model contracts, exact saved feature order/threshold, legacy compatibility, wrong identities, schema mismatches, unsupported machines and pre-load symlink rejection.
- C# tests cover linked persistence, independent machine histories, sensor units, unsupported machines and invalid model responses; the opt-in live check exercises real Python predictions and C# JWT HTTP handling.
- React tests cover dynamic forms for both cleaning machines, submission shape, empty input validation, unavailable models and stale responses across machine selections.
- All model artifact SHA-256 values and pre-refactor sample predictions were checked unchanged.
- CLN-102 metadata records scikit-learn **1.6.1**, matching runtime. Legacy CLN-101 emitted no mismatch warning. Original AI4I loading emits a **1.9.1 → 1.6.1** mismatch warning; its model is preserved.
- Full React test discovery currently fails in the unrelated `App.test.js` because Jest cannot resolve `react-router/dom`. The production build succeeds with the existing `UsersPage.js` missing-effect-dependency lint warning; CI treats that warning as an error. These unrelated files/dependencies were not changed.

All cleaning-model data, labels and evaluations are synthetic. Results are experimental, not field-validated maintenance evidence. No test forces the machines' predictions to differ.

## Changed source files

ManufacturingML: `api.py`, new `model_registry.py`, `tests/test_cln102_api.py`, new `tests/test_model_registry.py`, and this document.

FabCore backend: `FabCore.Api/Controllers/MachineAIController.cs` (partial declaration only), new `MachineAIController.Registry.cs`, new `FabCore.Tests/AI/MachineAwareControllerTests.cs`, new `MachineAwareHttpTests.cs`.

FabCore frontend: `src/Pages/MachineHealthPage.js`, `src/Pages/MachineHealthPage.test.js`.

Final run: Python **12 passed**; C# with live FastAPI **180 passed, 4 MySQL tests skipped**; targeted React **6 passed**. Non-CI production build succeeded with the existing UsersPage lint warning. All four saved model checksums matched their pre-change values. Both saved pre-refactor sample prediction responses matched exactly after the refactor. The temporary FastAPI test service was stopped after verification.
