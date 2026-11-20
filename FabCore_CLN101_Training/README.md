# FabCore CLN-101: train your first machine-specific model

**Purpose:** Learn the full data-generation → training → evaluation → inference workflow before wiring a custom model to your FabCore MES.

**Critical limitation:** This project generates *synthetic* observations and *synthetic failure labels* for an illustrative cleaning-machine process. It does not use telemetry or verified failures from CLN-101. Scores on held-out synthetic cycles are *not* real-world predictive-maintenance validation. The generated temperatures, pressures, flows and hazard dynamics are placeholders, not validated operating specifications.

## Run on Mac or Windows in VS Code

Open a terminal **in this folder** (or `cd` into it). Use the Python environment where your scikit-learn packages are installed.

```bash
python -m pip install -r requirements.txt
python 01_generate_dataset.py
python 02_train_model.py
python 03_try_prediction.py
```

On some Macs, use `python3` instead of `python`.

## What you created

- `cln101_synthetic_telemetry.csv` — labelled hourly samples from separate simulated maintenance cycles.
- `cln101_model.joblib` — your trained Random Forest, seven input features, and chosen decision threshold.
- `cln101_report.json` — held-out test metrics and feature importances.

The **target** is `failure_within_12h`. We train only with seven sensor/maintenance features; `machine_code`, `cycle_id`, `hour_in_cycle` and target are not accidentally passed to the model. The split is by *maintenance cycle*, not random row, so readings from one cycle cannot appear in both train and test. The decision threshold comes from validation data, before the final test.

## Feature definitions (simulated)

| Feature | Unit | Meaning |
|---|---|---|
| chamber_temperature_c | °C | Chamber/process temperature |
| chamber_pressure_kpa | kPa | Chamber pressure |
| pump_vibration_mm_s | mm/s | Pump vibration |
| chemical_flow_l_min | L/min | Chemical flow |
| filter_diff_pressure_kpa | kPa | Pressure difference over filter |
| motor_current_a | A | Motor electrical current |
| hours_since_maintenance | h | Equipment time since service |

## Important for FabCore integration

Your **current** Python FastAPI `/predict` accepts 5 *AI4I* inputs (air/process temperature, speed, torque, wear). This custom CLN-101 model uses **7 different** input features and predicts *failure within 12 hours*. It cannot simply replace the old `.joblib` file without changing the Python API request schema, the C# payload, and the React form.

After your local train/evaluate works, integrate it as a **separate endpoint** (`/predict/cleaning`) or with a model registry keyed by `machineCode`. Keep the original AI4I endpoint working until the new end-to-end path has been tested. Do not treat synthetic model results as real equipment maintenance advice.

## Next steps toward research-quality evidence

Get real historical telemetry and time-stamped maintenance/failure events from a cleaning tool. Define the prediction horizon, check missing readings, split train/test chronologically by maintenance cycle or equipment, and test whether the model generalises to data outside the training period. Evaluate false alarms, missed failures, precision, recall, and calibration. Only then make claims about genuine predictive performance.
