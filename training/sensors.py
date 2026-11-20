"""Synthetic measurement schemas shared by training and serving.

Units and labels describe illustrative simulations, not real tool specifications.
"""
CLEANING_SENSORS = {
    'chamber_temperature_c': ('Chamber temperature', '°C'),
    'chamber_pressure_kpa': ('Chamber pressure', 'kPa'),
    'pump_vibration_mm_s': ('Pump vibration', 'mm/s'),
    'chemical_flow_l_min': ('Chemical flow', 'L/min'),
    'filter_diff_pressure_kpa': ('Filter differential pressure', 'kPa'),
    'motor_current_a': ('Motor current', 'A'),
    'hours_since_maintenance': ('Hours since maintenance', 'h'),
}
ETCHING_SENSORS = {
    'chamber_pressure_mtorr': ('Chamber pressure', 'mTorr'),
    'rf_forward_power_w': ('RF forward power', 'W'),
    'rf_reflected_power_w': ('RF reflected power', 'W'),
    'process_gas_flow_sccm': ('Process gas flow', 'sccm'),
    'electrode_temperature_c': ('Electrode temperature', '°C'),
    'vacuum_pump_vibration_mm_s': ('Vacuum pump vibration', 'mm/s'),
    'hours_since_maintenance': ('Hours since maintenance', 'h'),
}
def equipment_sensors(rows):
    return dict([*rows, ('hours_since_maintenance', ('Hours since maintenance', 'h'))])


LITHOGRAPHY_SENSORS = equipment_sensors([
    ('exposure_dose_mj_cm2', ('Exposure dose', 'mJ/cm²')),
    ('focus_error_um', ('Focus error', 'µm')),
    ('stage_position_error_um', ('Stage position error', 'µm')),
    ('illumination_intensity_mw_cm2', ('Illumination intensity', 'mW/cm²')),
    ('lens_temperature_c', ('Lens temperature', '°C')),
    ('stage_vibration_mm_s', ('Stage vibration', 'mm/s')),
])
DEPOSITION_SENSORS = equipment_sensors([
    ('chamber_pressure_pa', ('Chamber pressure', 'Pa')),
    ('substrate_temperature_c', ('Substrate temperature', '°C')),
    ('precursor_flow_sccm', ('Precursor flow', 'sccm')),
    ('deposition_rate_nm_min', ('Deposition rate', 'nm/min')),
    ('rf_power_w', ('RF power', 'W')),
    ('exhaust_pump_vibration_mm_s', ('Exhaust pump vibration', 'mm/s')),
])
POLISHING_SENSORS = equipment_sensors([
    ('platen_speed_rpm', ('Platen speed', 'rpm')),
    ('downforce_kpa', ('Downforce', 'kPa')),
    ('slurry_flow_ml_min', ('Slurry flow', 'mL/min')),
    ('pad_temperature_c', ('Pad temperature', '°C')),
    ('motor_current_a', ('Motor current', 'A')),
    ('platen_vibration_mm_s', ('Platen vibration', 'mm/s')),
])
INSPECTION_SENSORS = equipment_sensors([
    ('illumination_intensity_mw_cm2', ('Illumination intensity', 'mW/cm²')),
    ('camera_temperature_c', ('Camera temperature', '°C')),
    ('focus_error_um', ('Focus error', 'µm')),
    ('stage_position_error_um', ('Stage position error', 'µm')),
    ('image_noise_dn', ('Image noise', 'DN')),
    ('stage_vibration_mm_s', ('Stage vibration', 'mm/s')),
])
METROLOGY_SENSORS = equipment_sensors([
    ('laser_power_mw', ('Laser power', 'mW')),
    ('detector_noise_mv', ('Detector noise', 'mV')),
    ('reference_drift_nm', ('Reference drift', 'nm')),
    ('stage_repeatability_um', ('Stage repeatability', 'µm')),
    ('enclosure_temperature_c', ('Enclosure temperature', '°C')),
    ('interferometer_signal_db', ('Interferometer signal', 'dB')),
])
PACKAGING_SENSORS = equipment_sensors([
    ('bond_head_temperature_c', ('Bond head temperature', '°C')),
    ('bonding_force_n', ('Bonding force', 'N')),
    ('vacuum_pickup_kpa', ('Vacuum pickup differential', 'kPa')),
    ('alignment_error_um', ('Alignment error', 'µm')),
    ('ultrasonic_power_w', ('Ultrasonic power', 'W')),
    ('head_vibration_mm_s', ('Head vibration', 'mm/s')),
])
MACHINE_SENSORS = {
    'CLN-101': CLEANING_SENSORS, 'CLN-102': CLEANING_SENSORS,
    'ETCH-101': ETCHING_SENSORS, 'ETCH-102': ETCHING_SENSORS,
    'LTH-101': LITHOGRAPHY_SENSORS, 'LTH-102': LITHOGRAPHY_SENSORS,
    'DEP-101': DEPOSITION_SENSORS, 'DEP-102': DEPOSITION_SENSORS,
    'CMP-101': POLISHING_SENSORS, 'INSP-101': INSPECTION_SENSORS,
    'MET-101': METROLOGY_SENSORS, 'PACK-101': PACKAGING_SENSORS,
}


def features_for(machine):
    return list(MACHINE_SENSORS[machine])
