"""SYNTHETIC etching cycles, not observed semiconductor equipment data.

Competing cumulative RF/exhaust stress first-passage processes deliberately
replace the cleaning simulator's scalar degradation-dependent failure hazard.
Latent states, stress budgets and future event times are never input features.
"""
import numpy as np

HORIZON_HOURS = 12


def simulate_etch_cycles(profile, n_cycles, machine="ETCH-101"):
    rng = np.random.default_rng(profile['seed'])
    rows, events = [], []
    for cycle_id in range(1, n_cycles + 1):
        planned_hours = int(rng.integers(*profile['planned']))
        pressure_base = rng.uniform(*profile['pressure_setpoint_mtorr'])
        rf_base = rng.uniform(*profile['rf_setpoint_w'])
        gas_base = rng.uniform(*profile['gas_setpoint_sccm'])
        matching_drift = rng.uniform(*profile['matching_drift'])
        exhaust_drift = rng.uniform(*profile['exhaust_drift'])
        matching = rng.uniform(0.03, 0.10)
        restriction = rng.uniform(0.02, 0.12)
        temperature = rng.normal(50.0, 2.0)
        rf_budget = rng.gamma(*profile['rf_budget_gamma'])
        vacuum_budget = rng.gamma(*profile['vacuum_budget_gamma'])
        rf_damage = vacuum_damage = 0.0
        failure_hour = None
        failure_mode = 'planned_maintenance'
        observed = []
        for hour in range(planned_hours):
            matching = max(0, matching + matching_drift + rng.normal(0, 0.002))
            restriction = max(0, restriction + exhaust_drift + rng.normal(0, 0.0015))
            rf_load, gas_load = rf_base / 1300, gas_base / 100
            # Independent, multiplicative stress shocks; no cleaning hazard rule.
            sigma = profile['stress_noise_sigma']
            rf_damage += rf_load * (0.12 + 4 * matching ** 2) * rng.lognormal(-sigma ** 2 / 2, sigma)
            vacuum_damage += gas_load * (0.10 + 2.2 * restriction ** 2) * rng.lognormal(-sigma ** 2 / 2, sigma)
            if rf_damage >= rf_budget or vacuum_damage >= vacuum_budget:
                failure_hour = hour
                failure_mode = 'rf_matching_trip' if rf_damage >= rf_budget else 'vacuum_load_trip'
                break  # No reading at or after the simulated event.
            temperature = 0.92 * temperature + 0.08 * (45 + 12 * rf_load + 30 * matching + 7 * restriction) + rng.normal(0, 0.5)
            forward = max(1.0, rf_base + rng.normal(0, 25))
            reflected = np.clip(forward * (0.015 + 0.55 * matching ** 2) + rng.normal(0, 4), 0, forward)
            observed.append({
                'machine_code': machine, 'cycle_id': cycle_id, 'hour_in_cycle': hour,
                # Observable age since this cycle's maintenance, never time to failure.
                'hours_since_maintenance': hour,
                'chamber_pressure_mtorr': round(max(0.1, pressure_base * (1 + 0.4 * restriction) + rng.normal(0, 1.3)), 3),
                'rf_forward_power_w': round(forward, 3),
                'rf_reflected_power_w': round(float(reflected), 3),
                'process_gas_flow_sccm': round(max(0.1, gas_base * (1 - 0.18 * restriction) + rng.normal(0, 2)), 3),
                'electrode_temperature_c': round(temperature, 3),
                'vacuum_pump_vibration_mm_s': round(max(0.1, 0.7 + 3.6 * restriction + 0.15 * gas_load + rng.normal(0, 0.15)), 3),
            })
        for reading in observed:
            hour = reading['hour_in_cycle']
            # For maintenance-terminated cycles, the full future window must be observed.
            if failure_hour is None and hour + HORIZON_HOURS >= planned_hours:
                continue
            reading['failure_within_12h'] = int(failure_hour is not None and 0 < failure_hour - hour <= HORIZON_HOURS)
            rows.append(reading)
        events.append(dict(machine_code=machine, cycle_id=cycle_id, planned_hours=planned_hours,
                           failure_hour=failure_hour, failure_mode=failure_mode, is_simulated=True))
    return rows, events
