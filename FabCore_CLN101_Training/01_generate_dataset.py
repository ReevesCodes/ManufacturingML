"""Generate a SYNTHETIC, labelled history for FabCore machine CLN-101.

This is an educational wet-cleaning-equipment simulation, NOT data from
an actual semiconductor tool. Generated labels are simulated failures
within the next 12 hours, not observed real-world failures.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / 'cln101_synthetic_telemetry.csv'
RNG = np.random.default_rng(42)
N_CYCLES = 180
LOOKAHEAD_HOURS = 12


def generate():
    rows = []
    failures = 0
    for cycle_id in range(1, N_CYCLES + 1):
        # A cycle begins after simulated maintenance and ends at maintenance
        # or a simulated equipment failure. Nothing here is a real telemetry feed.
        planned_hours = int(RNG.integers(95, 170))
        degradation = float(RNG.uniform(0.025, 0.18))
        drift = float(RNG.uniform(0.0032, 0.0095))
        tool_offset = float(RNG.normal(0, 0.7))
        observed = []
        failure_hour = None

        for hour in range(planned_hours):
            degradation = max(0, degradation + drift + float(RNG.normal(0, 0.002)))

            # Hidden stochastic failure process: a failure event is sampled,
            # rather than setting the label using a simple sensor threshold.
            hazard = min(0.16, max(0.0, (degradation - 0.48) * 0.24))
            if RNG.random() < hazard:
                failure_hour = hour
                break  # No telemetry is collected after a failure.

            observed.append({
                'machine_code': 'CLN-101',
                'cycle_id': cycle_id,
                'hour_in_cycle': hour,
                'chamber_temperature_c': round(35.5 + 10.0 * degradation + tool_offset + RNG.normal(0, 1.4), 3),
                'chamber_pressure_kpa': round(100.5 + 7.2 * degradation + RNG.normal(0, 1.7), 3),
                'pump_vibration_mm_s': round(max(0.1, 0.9 + 4.0 * degradation + RNG.normal(0, 0.38)), 3),
                'chemical_flow_l_min': round(max(0.1, 18.5 - 6.0 * degradation + RNG.normal(0, 0.9)), 3),
                'filter_diff_pressure_kpa': round(max(0.1, 4.0 + 9.0 * degradation + RNG.normal(0, 0.85)), 3),
                'motor_current_a': round(max(0.1, 6.6 + 4.3 * degradation + RNG.normal(0, 0.6)), 3),
                'hours_since_maintenance': hour,
            })

        if failure_hour is not None:
            failures += 1
        for reading in observed:
            # Ground truth in this simulation: event occurs in the *future*.
            # The event itself and cycle_id are NOT model input features.
            t = reading['hour_in_cycle']
            reading['failure_within_12h'] = int(
                failure_hour is not None and 0 < failure_hour - t <= LOOKAHEAD_HOURS
            )
            rows.append(reading)

    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    print(f'Generated {len(df):,} hourly simulated readings across {N_CYCLES} independent maintenance cycles.')
    print(f'Simulated cycles ending in equipment failure: {failures}/{N_CYCLES}')
    print(f'Failure-within-12h labelled readings: {int(df.failure_within_12h.sum()):,} ({df.failure_within_12h.mean():.1%})')
    print(f'Dataset: {DATA_PATH}')
    print('IMPORTANT: This is synthetic data; scores are not real-world evidence.')


if __name__ == '__main__':
    generate()
