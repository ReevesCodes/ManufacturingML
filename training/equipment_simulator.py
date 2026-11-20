"""Equipment-specific SYNTHETIC cycles; no field measurements or validated physics.

The common engine models independent maintenance resets and two competing
cumulative stress channels; family-specific equations and sensor responses are
explicit below and in equipment_profiles.py. Budgets/events/latent states are
never exported as model inputs.
"""
import numpy as np
from training.sensors import features_for


def stress_rates(family, a, b, thermal):
    if family == 'lithography':
        return 0.10 + 3 * a ** 2, 0.05 + 1.8 * b ** 1.5
    if family == 'deposition':
        return 0.08 + 1.1 * a, 0.06 + 4 * b ** 2
    if family == 'polishing':
        return 0.12 + 0.9 * a * (1 + b), 0.07 + 3 * b ** 2
    if family == 'inspection':
        return 0.06 + 1.7 * max(0, thermal) ** 2, 0.04 + 2.4 * b ** 1.5
    if family == 'metrology':
        return 0.06 + 2.5 * a ** 2, 0.05 + 3 * a * b
    if family == 'packaging':
        return 0.08 + a * (1 + b), 0.10 + 3.5 * b ** 2
    raise ValueError(f'Unknown simulation family: {family}')


def simulate_equipment_cycles(machine, profile, n_cycles):
    rng = np.random.default_rng(profile['seed'])
    sensors = features_for(machine)[:-1]
    rows, events = [], []
    for cycle in range(1, n_cycles + 1):
        planned = int(rng.integers(*profile['planned']))
        workload = rng.uniform(0.85, 1.15)
        a, b = rng.uniform(0.03, 0.12), rng.uniform(0.02, 0.10)
        drift_a, drift_b = rng.uniform(*profile['drift_a']), rng.uniform(*profile['drift_b'])
        budget_a, budget_b = rng.gamma(*profile['budget_a']), rng.gamma(*profile['budget_b'])
        damage_a = damage_b = thermal = 0.0
        offsets = rng.normal(0, np.array(profile['noise']) * 0.4)
        observed, failure_hour, mode = [], None, 'planned_maintenance'
        for hour in range(planned):
            a = max(0, a + drift_a * workload + rng.normal(0, 0.0015))
            b = max(0, b + drift_b * workload + rng.normal(0, 0.0015))
            thermal = 0.92 * thermal + 0.08 * (a + b) + rng.normal(0, 0.005)
            rate_a, rate_b = stress_rates(profile['family'], a, b, thermal)
            sigma = profile['stress_noise_sigma']
            damage_a += workload * rate_a * rng.lognormal(-sigma ** 2 / 2, sigma)
            damage_b += workload * rate_b * rng.lognormal(-sigma ** 2 / 2, sigma)
            if damage_a >= budget_a or damage_b >= budget_b:
                failure_hour = hour
                mode = profile['failure_modes'][0 if damage_a >= budget_a else 1]
                break
            values = (np.array(profile['baseline']) + np.array(profile['response_a']) * a
                      + np.array(profile['response_b']) * b + offsets + rng.normal(0, profile['noise']))
            # Signed optical errors/drifts are meaningful. Other simulated sensors
            # have a zero floor; this is a simulator constraint, not an operating limit.
            reading = dict(machine_code=machine, cycle_id=cycle, hour_in_cycle=hour,
                           hours_since_maintenance=hour)
            for name, value in zip(sensors, values):
                signed = name in {'focus_error_um', 'reference_drift_nm'}
                reading[name] = round(float(value if signed else max(0.001, value)), 3)
            observed.append(reading)
        for reading in observed:
            hour = reading['hour_in_cycle']
            if failure_hour is None and hour + 12 >= planned:
                continue
            reading['failure_within_12h'] = int(failure_hour is not None and 0 < failure_hour - hour <= 12)
            rows.append(reading)
        events.append(dict(machine_code=machine, cycle_id=cycle, planned_hours=planned,
                           failure_hour=failure_hour, failure_mode=mode, is_simulated=True))
    return rows, events
