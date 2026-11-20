"""Try your trained CLN-101 model on two new simulated readings."""
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / 'cln101_model.joblib'

SCENARIOS = {
    'Routine simulated reading': {
        'chamber_temperature_c': 37.5,
        'chamber_pressure_kpa': 102.0,
        'pump_vibration_mm_s': 1.3,
        'chemical_flow_l_min': 18.0,
        'filter_diff_pressure_kpa': 5.2,
        'motor_current_a': 7.0,
        'hours_since_maintenance': 18,
    },
    'Degraded simulated reading': {
        'chamber_temperature_c': 46.0,
        'chamber_pressure_kpa': 108.5,
        'pump_vibration_mm_s': 5.0,
        'chemical_flow_l_min': 12.5,
        'filter_diff_pressure_kpa': 13.2,
        'motor_current_a': 11.3,
        'hours_since_maintenance': 128,
    },
}


def main():
    if not MODEL_PATH.exists():
        raise SystemExit('Model missing. Run: python 02_train_model.py')
    bundle = joblib.load(MODEL_PATH)  # Load only files you created/trust.
    for name, sensors in SCENARIOS.items():
        x = pd.DataFrame([sensors], columns=bundle['features'])
        p = float(bundle['model'].predict_proba(x)[0, 1])
        decision = int(p >= bundle['decision_threshold'])
        print(f'{name}: probability={p:.3f}, failureWithin12h={decision}')
    print('These are synthetic demonstration outputs, not field-validated warnings.')


if __name__ == '__main__':
    main()
