"""Illustrative simulation parameters, not equipment specifications."""
from training.sensors import CLEANING_SENSORS
from training.equipment_profiles import equipment_profile

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "CLN-101": dict(seed=42, planned=(95, 170), drift=(0.0032, 0.0095),
                    baseline=(35.5, 100.5, 0.9, 18.5, 4.0, 6.6),
                    sensitivity=(10.0, 7.2, 4.0, -6.0, 9.0, 4.3),
                    hazard_onset=0.48, hazard_slope=0.24),
    "CLN-102": dict(seed=102, planned=(110, 190), drift=(0.0025, 0.0075),
                    baseline=(39.0, 104.0, 1.1, 21.0, 3.5, 7.4),
                    sensitivity=(8.0, 5.5, 3.2, -8.0, 12.0, 3.6),
                    hazard_onset=0.55, hazard_slope=0.30),
    "ETCH-101": dict(
        seed=10101, simulator="etch-competing-stress-v2", planned=(120, 240),
        pressure_setpoint_mtorr=(30.0, 60.0), rf_setpoint_w=(1050.0, 1550.0),
        gas_setpoint_sccm=(70.0, 125.0), matching_drift=(0.0015, 0.0045),
        exhaust_drift=(0.0010, 0.0030), rf_budget_gamma=(9.0, 5.0),
        vacuum_budget_gamma=(10.0, 4.0), stress_noise_sigma=0.25,
    ),
}
# Each remaining machine has an independent seed, configuration and fitted model.
PROFILES.update({
    'ETCH-102': dict(PROFILES['ETCH-101'], seed=10202, model_seed=10209,
        model_version_suffix='SYN-RF-v1', pressure_setpoint_mtorr=(36.0, 66.0),
        rf_setpoint_w=(1150.0, 1650.0), gas_setpoint_sccm=(80.0, 135.0),
        matching_drift=(0.0013, 0.0041), exhaust_drift=(0.0012, 0.0034),
        rf_budget_gamma=(9.0, 5.4), vacuum_budget_gamma=(10.0, 3.8)),
    'LTH-101': equipment_profile('lithography', 20101),
    'LTH-102': equipment_profile('lithography', 20102, scale=1.08, offset=0.01),
    'DEP-101': equipment_profile('deposition', 30101),
    'DEP-102': equipment_profile('deposition', 30102, scale=1.10, offset=0.02),
    'CMP-101': equipment_profile('polishing', 40101),
    'INSP-101': equipment_profile('inspection', 50101),
    'MET-101': equipment_profile('metrology', 60101),
    'PACK-101': equipment_profile('packaging', 70101),
})
PRESERVED_MACHINES = ('CLN-101', 'CLN-102', 'ETCH-101')

FEATURES = list(CLEANING_SENSORS)  # Legacy cleaning-schema import compatibility.

def arguments(description, generator=False):
    parser = argparse.ArgumentParser(description=description)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--machine', choices=PROFILES)
    selection.add_argument('--all', action='store_true')
    if generator:
        parser.add_argument('--cycles', type=int, default=180)
    parser.add_argument('--skip-existing', action='store_true', help='Leave existing datasets/models untouched in a batch')
    args = parser.parse_args()
    if generator and args.cycles < 20:
        parser.error('--cycles must be at least 20 for grouped evaluation')
    return args, list(PROFILES) if args.all else [args.machine]
