"""Generate a SYNTHETIC, labelled history for configured FabCore machines.

These are educational cleaning/etching simulations, NOT data from
an actual semiconductor tool. Generated labels are simulated failures
within the next 12 hours, not observed real-world failures.
"""
import json
from training.config import ROOT, PROFILES, arguments
from training.etch_simulator import simulate_etch_cycles
from training.equipment_simulator import simulate_equipment_cycles

import numpy as np
import pandas as pd

LOOKAHEAD_HOURS = 12


def generate_cleaning(machine, n_cycles=180):
    profile = PROFILES[machine]
    RNG = np.random.default_rng(profile['seed'])
    N_CYCLES = n_cycles
    DATA_PATH = ROOT / 'artifacts' / machine / 'synthetic_telemetry.csv'
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = 0
    for cycle_id in range(1, N_CYCLES + 1):
        # A cycle begins after simulated maintenance and ends at maintenance
        # or a simulated equipment failure. Nothing here is a real telemetry feed.
        planned_hours = int(RNG.integers(*profile['planned']))
        degradation = float(RNG.uniform(0.025, 0.18))
        drift = float(RNG.uniform(*profile['drift']))
        tool_offset = float(RNG.normal(0, 0.7))
        observed = []
        failure_hour = None

        for hour in range(planned_hours):
            degradation = max(0, degradation + drift + float(RNG.normal(0, 0.002)))

            # Hidden stochastic failure process: a failure event is sampled,
            # rather than setting the label using a simple sensor threshold.
            hazard = min(0.16, max(0.0, (degradation - profile['hazard_onset']) * profile['hazard_slope']))
            if RNG.random() < hazard:
                failure_hour = hour
                break  # No telemetry is collected after a failure.

            observed.append({
                'machine_code': machine,
                'cycle_id': cycle_id,
                'hour_in_cycle': hour,
                'chamber_temperature_c': round(profile['baseline'][0] + profile['sensitivity'][0] * degradation + tool_offset + RNG.normal(0, 1.4), 3),
                'chamber_pressure_kpa': round(profile['baseline'][1] + profile['sensitivity'][1] * degradation + RNG.normal(0, 1.7), 3),
                'pump_vibration_mm_s': round(max(0.1, profile['baseline'][2] + profile['sensitivity'][2] * degradation + RNG.normal(0, 0.38)), 3),
                'chemical_flow_l_min': round(max(0.1, profile['baseline'][3] + profile['sensitivity'][3] * degradation + RNG.normal(0, 0.9)), 3),
                'filter_diff_pressure_kpa': round(max(0.1, profile['baseline'][4] + profile['sensitivity'][4] * degradation + RNG.normal(0, 0.85)), 3),
                'motor_current_a': round(max(0.1, profile['baseline'][5] + profile['sensitivity'][5] * degradation + RNG.normal(0, 0.6)), 3),
                'hours_since_maintenance': hour,
            })

        if failure_hour is not None:
            failures += 1
        for reading in observed:
            # Ground truth in this simulation: event occurs in the *future*.
            # The event itself and cycle_id are NOT model input features.
            t = reading['hour_in_cycle']
            if failure_hour is None and t + LOOKAHEAD_HOURS >= planned_hours:
                continue  # Right-censored: full future horizon is unknown.
            reading['failure_within_12h'] = int(
                failure_hour is not None and 0 < failure_hour - t <= LOOKAHEAD_HOURS
            )
            rows.append(reading)

    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    metadata = dict(machine_code=machine, is_simulated=True, profile=profile,
                    cycles=N_CYCLES, seed=profile['seed'], horizon_hours=12,
                    label_rule='Future sampled failure in (t, t+12]; censored tails excluded')
    DATA_PATH.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Generated {len(df):,} hourly simulated readings across {N_CYCLES} independent maintenance cycles.')
    print(f'Simulated cycles ending in equipment failure: {failures}/{N_CYCLES}')
    print(f'Failure-within-12h labelled readings: {int(df.failure_within_12h.sum()):,} ({df.failure_within_12h.mean():.1%})')
    print(f'Dataset: {DATA_PATH}')
    print('IMPORTANT: This is synthetic data; scores are not real-world evidence.')


def generate(machine, n_cycles=180):
    if machine.startswith('CLN-'):
        return generate_cleaning(machine, n_cycles)
    profile = PROFILES[machine]
    if machine.startswith('ETCH-'):
        rows, events = simulate_etch_cycles(profile, n_cycles, machine)
    else:
        rows, events = simulate_equipment_cycles(machine, profile, n_cycles)
    directory = ROOT / 'artifacts' / machine
    directory.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    data_path = directory / 'synthetic_telemetry.csv'
    event_path = directory / 'synthetic_cycle_events.csv'
    df.to_csv(data_path, index=False)
    pd.DataFrame(events).to_csv(event_path, index=False)
    import hashlib
    metadata = dict(
        machine_code=machine, is_simulated=True, data_origin='Synthetic equipment simulation; not real machine data',
        profile=profile, cycles=n_cycles, seed=profile['seed'], horizon_hours=LOOKAHEAD_HOURS,
        label_rule='First equipment-specific stress budget crossing in (t, t+12]; censored maintenance tails excluded',
        event_audit_file=event_path.name, event_audit_sha256=hashlib.sha256(event_path.read_bytes()).hexdigest(),
        event_audit_is_model_input=False,
        maintenance_age_rule='Elapsed hourly cycle time, starting at zero after each maintenance reset',
    )
    data_path.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'{machine}: {len(df):,} SYNTHETIC hourly readings, {n_cycles} independent cycles')
    print(f'Simulated failures: {sum(e["failure_hour"] is not None for e in events)} cycles')
    print(f'12-hour positive labels: {int(df.failure_within_12h.sum())} ({df.failure_within_12h.mean():.1%})')
    print(f'Dataset: {data_path}')
    print('SYNTHETIC simulation only; no real semiconductor machine measurements or failures.')


if __name__ == '__main__':
    args, machines = arguments('Generate SYNTHETIC machine telemetry', generator=True)
    for machine in machines:
        if args.skip_existing and (ROOT / 'artifacts' / machine / 'synthetic_telemetry.csv').exists():
            print(f'{machine}: existing dataset preserved')
            continue
        generate(machine, args.cycles)
