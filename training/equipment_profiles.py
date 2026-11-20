"""Invented equipment-specific profiles, NOT measured semiconductor specifications.

Each family uses two hidden wear channels, different observable responses and
failure-stress equations (equipment_simulator.py). No hidden state is a feature.
All tuples are deterministic configuration, never fitted to held-out outcomes.
"""
FAMILIES = {
    'lithography': dict(
        planned=(140, 230), drift_a=(0.0020, 0.0050), drift_b=(0.0015, 0.0040),
        baseline=(120, 0, 0.03, 20, 22, 0.15), noise=(2.5, 0.05, 0.02, 0.5, 0.3, 0.04),
        response_a=(-15, 0.6, 0.04, -5, 3, 0.1), response_b=(3, 0.15, 0.35, -1, 2, 0.7),
        failure_modes=('illumination_optics_trip', 'focus_stage_trip')),
    'deposition': dict(
        planned=(130, 220), drift_a=(0.0025, 0.0055), drift_b=(0.0015, 0.0045),
        baseline=(120, 350, 80, 25, 600, 0.8), noise=(3, 3, 1.5, 0.6, 12, 0.12),
        response_a=(45, 8, -20, -10, 30, 0.5), response_b=(12, 35, -5, -3, 50, 3),
        failure_modes=('chamber_coating_trip', 'heater_exhaust_trip')),
    'polishing': dict(
        planned=(120, 210), drift_a=(0.0025, 0.0055), drift_b=(0.0018, 0.0045),
        baseline=(60, 25, 200, 30, 8, 0.5), noise=(0.8, 0.5, 4, 0.6, 0.25, 0.12),
        response_a=(-3, 8, -70, 12, 5, 0.5), response_b=(5, 2, -10, 5, 2, 3),
        failure_modes=('pad_slurry_trip', 'platen_bearing_trip')),
    'inspection': dict(
        planned=(150, 240), drift_a=(0.0018, 0.0045), drift_b=(0.0018, 0.0045),
        baseline=(8, 30, 0, 0.03, 2, 0.1), noise=(0.2, 0.5, 0.03, 0.015, 0.25, 0.03),
        response_a=(-3, 15, 0.15, 0.04, 6, 0.1), response_b=(-0.5, 2, 0.6, 0.3, 1, 0.8),
        failure_modes=('camera_thermal_trip', 'optical_stage_trip')),
    'metrology': dict(
        planned=(160, 250), drift_a=(0.0020, 0.0048), drift_b=(0.0015, 0.0045),
        baseline=(5, 0.4, 0, 0.015, 22, 40), noise=(0.1, 0.08, 0.5, 0.008, 0.2, 0.6),
        response_a=(-2, 2.5, 12, 0.03, 2, -10), response_b=(-0.4, 0.8, 6, 0.12, 3, -3),
        failure_modes=('detector_noise_trip', 'reference_stability_trip')),
    'packaging': dict(
        planned=(130, 220), drift_a=(0.0020, 0.0050), drift_b=(0.0020, 0.0050),
        baseline=(150, 8, 65, 1, 25, 0.2), noise=(2, 0.25, 1.3, 0.3, 0.8, 0.06),
        response_a=(8, 3, -25, 2, 5, 0.4), response_b=(40, 1, -4, 9, 15, 1.8),
        failure_modes=('pickup_seal_trip', 'bond_head_trip')),
}


def equipment_profile(family, seed, *, scale=1.0, offset=0.0):
    base = FAMILIES[family]
    return dict(base, family=family, simulator='equipment-competing-stress-v1', seed=seed,
                model_seed=seed + 7, model_version_suffix='SYN-RF-v1',
                baseline=tuple(x * scale + offset for x in base['baseline']),
                drift_a=tuple(x * scale for x in base['drift_a']),
                drift_b=tuple(x / scale for x in base['drift_b']),
                budget_a=(9.0, 4.5 * scale), budget_b=(7.0, 5.0 / scale), stress_noise_sigma=0.30)
