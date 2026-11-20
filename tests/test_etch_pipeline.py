"""Tests of SYNTHETIC etching labels, evaluation and machine-specific serving."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)

import api
import model_registry
from generate_dataset import generate
from training.config import ROOT
from training.sensors import ETCHING_SENSORS
from train_model import split_by_cycle
from test_cln102_api import REQUEST, post

DIRECTORY = ROOT / 'artifacts' / 'ETCH-101'


class EtchPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(DIRECTORY / 'synthetic_telemetry.csv')
        cls.events = pd.read_csv(DIRECTORY / 'synthetic_cycle_events.csv')
        cls.bundle = joblib.load(DIRECTORY / 'model.joblib')
        cls.report = json.loads((DIRECTORY / 'evaluation.json').read_text())

    def test_feature_schema_has_only_current_etch_telemetry(self):
        features = list(ETCHING_SENSORS)
        self.assertEqual(self.bundle['features'], features)
        self.assertEqual(set(self.data.columns), set(features + ['machine_code', 'cycle_id', 'hour_in_cycle', 'failure_within_12h']))
        self.assertEqual(set(self.data.machine_code), {'ETCH-101'})
        self.assertTrue(np.isfinite(self.data[features]).all().all())
        self.assertTrue((self.data.rf_reflected_power_w <= self.data.rf_forward_power_w).all())
        self.assertEqual(self.report['measurement_units']['chamber_pressure_mtorr'], 'mTorr')
        self.assertEqual(self.report['measurement_units']['process_gas_flow_sccm'], 'sccm')

    def test_maintenance_age_resets_and_tracks_elapsed_hours(self):
        self.assertEqual(len(self.bundle['features']), 7)
        self.assertEqual(self.bundle['model_version'], 'ETCH-101-SYN-RF-v2')
        self.assertEqual(self.report['measurement_units']['hours_since_maintenance'], 'h')
        for _, cycle in self.data.groupby('cycle_id'):
            self.assertEqual(cycle.hours_since_maintenance.iloc[0], 0)
            np.testing.assert_array_equal(cycle.hours_since_maintenance, cycle.hour_in_cycle)
            np.testing.assert_array_equal(cycle.hours_since_maintenance, np.arange(len(cycle)))

    def test_future_labels_and_censored_windows_against_event_audit(self):
        self.assertEqual(len(self.events), 180)
        self.assertTrue(self.events.is_simulated.all())
        self.assertEqual(self.report['generator_metadata']['event_audit_sha256'], hashlib.sha256((DIRECTORY / 'synthetic_cycle_events.csv').read_bytes()).hexdigest())
        self.assertFalse(self.report['generator_metadata']['event_audit_is_model_input'])
        # Includes maintenance-terminated and both competing failure mechanisms.
        self.assertEqual(set(self.events.failure_mode), {'planned_maintenance', 'rf_matching_trip', 'vacuum_load_trip'})
        for event in self.events.itertuples():
            readings = self.data[self.data.cycle_id == event.cycle_id]
            if pd.isna(event.failure_hour):
                self.assertTrue((readings.hour_in_cycle + 12 < event.planned_hours).all())
                self.assertEqual(int(readings.failure_within_12h.sum()), 0)
                self.assertEqual(len(readings), event.planned_hours - 12)
            else:
                self.assertTrue((readings.hour_in_cycle < event.failure_hour).all())
                expected = ((event.failure_hour - readings.hour_in_cycle) > 0) & ((event.failure_hour - readings.hour_in_cycle) <= 12)
                np.testing.assert_array_equal(expected.astype(int), readings.failure_within_12h)
                self.assertEqual(len(readings), int(event.failure_hour))

    def test_generator_event_audit_and_labels_are_reproducible(self):
        with TemporaryDirectory() as directory, patch('generate_dataset.ROOT', Path(directory)):
            generate('ETCH-101', 20)
            files = list((Path(directory) / 'artifacts' / 'ETCH-101').glob('*'))
            original = {p.name: p.read_bytes() for p in files}
            generate('ETCH-101', 20)
            self.assertEqual(original, {p.name: p.read_bytes() for p in files})

    def test_cleaning_simulation_remains_byte_identical(self):
        expected = {
            'CLN-101': 'b0d66a1246454d898ba732b43a8bfbe0546836e4188c012afc779688aea6e2f5',
            'CLN-102': '4e7a1c0afc0aea1a53d67521b81ca1bdf615c7215d5d71395d76e387c63be621',
        }
        with TemporaryDirectory() as directory, patch('generate_dataset.ROOT', Path(directory)):
            for machine, digest in expected.items():
                generate(machine, 20)
                path = Path(directory) / 'artifacts' / machine / 'synthetic_telemetry.csv'
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_group_isolation_and_saved_metrics_recompute(self):
        train, validation, test = split_by_cycle(self.data)
        frames = {'train': train, 'validation': validation, 'test': test}
        for name, frame in frames.items():
            self.assertEqual(sorted(frame.cycle_id.unique()), self.report['split_cycle_ids'][name])
            for other_name, other in frames.items():
                if name != other_name:
                    self.assertFalse(set(frame.cycle_id) & set(other.cycle_id))
        features = self.bundle['features']
        y = test.failure_within_12h
        probabilities = self.bundle['model'].predict_proba(test[features])[:, list(self.bundle['model'].classes_).index(1)]
        predictions = (probabilities >= self.bundle['decision_threshold']).astype(int)
        for key, metric in [('test_precision', precision_score), ('test_recall', recall_score), ('test_f1', f1_score)]:
            self.assertAlmostEqual(self.report[key], metric(y, predictions, zero_division=0))
        self.assertEqual(self.report['confusion_matrix_0_1'], confusion_matrix(y, predictions, labels=[0, 1]).tolist())
        precision, recall, _ = precision_recall_curve(y, probabilities)
        self.assertAlmostEqual(self.report['test_pr_auc_trapezoidal'], auc(recall, precision))
        self.assertAlmostEqual(self.report['test_average_precision'], average_precision_score(y, probabilities))
        self.assertEqual(self.report['decision_threshold'], self.bundle['decision_threshold'])
        for name, frame in [('all', self.data), *frames.items()]:
            self.assertEqual(self.report['class_distribution'][name], {str(label): int((frame.failure_within_12h == label).sum()) for label in [0, 1]})

    def test_schema_and_predictions_use_exact_etch_model(self):
        status, schema = post('/models/ETCH-101/schema', {}, method='GET')
        self.assertEqual(status, 200)
        self.assertTrue(schema['available'])
        self.assertTrue(schema['isSimulated'])
        self.assertEqual([f['name'] for f in schema['features']], list(ETCHING_SENSORS))
        age = next(f for f in schema['features'] if f['name'] == 'hours_since_maintenance')
        self.assertEqual(age['unit'], 'h')
        self.assertEqual(age['minimum'], 0)
        self.assertTrue(age['required'])
        self.assertEqual(len(schema['features']), 7)
        features = self.bundle['features']
        for index in [0, len(self.data) // 2, len(self.data) - 1]:
            sensors = self.data.iloc[index][features].astype(float).to_dict()
            status, response = post('/predict/machine', dict(machine_code='ETCH-101', readings=sensors))
            self.assertEqual(status, 200)
            expected = float(self.bundle['model'].predict_proba(pd.DataFrame([sensors], columns=features))[0, 1])
            self.assertEqual(response['machineCode'], 'ETCH-101')
            self.assertEqual(response['modelVersion'], self.bundle['model_version'])
            self.assertEqual(response['failureProbability'], round(expected, 4))
            self.assertEqual(response['failurePrediction'], int(expected >= self.bundle['decision_threshold']))
            self.assertEqual(response['decisionThreshold'], self.bundle['decision_threshold'])
            self.assertEqual(response['predictionHorizonHours'], 12)
            self.assertTrue(response['isSimulated'])

    def test_cross_machine_schemas_and_invalid_readings_rejected(self):
        sensors = self.data.iloc[0][list(ETCHING_SENSORS)].astype(float).to_dict()
        cleaning = {k: v for k, v in REQUEST.items() if k != 'machine_id'}
        for machine, readings in [('ETCH-101', cleaning), ('CLN-101', sensors), ('CLN-102', sensors)]:
            self.assertEqual(post('/predict/machine', dict(machine_code=machine, readings=readings))[0], 422)
        for field in ETCHING_SENSORS:
            invalid = {k: v for k, v in sensors.items() if k != field}
            self.assertEqual(post('/predict/machine', dict(machine_code='ETCH-101', readings=invalid))[0], 422)
        for readings in [dict(sensors, hidden_rf_damage=100), dict(sensors, rf_forward_power_w=True), dict(sensors, rf_forward_power_w='1200'), dict(sensors, hours_since_maintenance=-1)]:
            self.assertEqual(post('/predict/machine', dict(machine_code='ETCH-101', readings=readings))[0], 422)

    def test_etch_path_is_explicit_and_wrong_machine_artifacts_fail_closed(self):
        self.assertEqual(model_registry.model_path('ETCH-101'), DIRECTORY / 'model.joblib')
        model_registry.load_model.cache_clear()
        try:
            with patch('model_registry.joblib.load', return_value=api.joblib.load(ROOT / 'artifacts/CLN-102/model.joblib')):
                self.assertEqual(post('/models/ETCH-101/schema', {}, method='GET')[0], 503)
        finally:
            model_registry.load_model.cache_clear()
