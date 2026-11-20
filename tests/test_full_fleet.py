"""All twelve saved models, synthetic labels, compatibility and rejection paths."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import (auc, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, fbeta_score)

import api
import model_registry as registry
from training.config import ROOT, PROFILES, PRESERVED_MACHINES
from training.sensors import MACHINE_SENSORS
from train_model import split_by_cycle
from test_cln102_api import post

CATALOG = json.loads((ROOT / 'verification/full_fleet/model_catalog.json').read_text())


class FleetTests(unittest.TestCase):
    def tearDown(self):
        registry.load_model.cache_clear()

    def test_actual_inventory_has_twelve_registered_independent_models(self):
        self.assertEqual(len(CATALOG), 12)
        self.assertEqual({x['machineCode'] for x in CATALOG}, set(PROFILES))
        self.assertEqual(set(PROFILES), set(registry.REGISTRY))
        self.assertEqual(len({str(registry.model_path(x['machineCode'])) for x in CATALOG}), 12)
        self.assertEqual(len({x['evaluation']['model_sha256'] for x in CATALOG}), 12)
        for x in CATALOG:
            with self.subTest(machine=x['machineCode']):
                code = x['machineCode']
                bundle = registry.load_model(code)
                self.assertEqual(bundle['machine_code'], code)
                self.assertEqual(bundle['features'], list(MACHINE_SENSORS[code]))
                if code not in PRESERVED_MACHINES:
                    self.assertEqual(bundle['model'].random_state, PROFILES[code]['model_seed'])
                    self.assertTrue((ROOT / 'artifacts' / code / 'model_metadata.json').is_file())

    def test_all_twelve_asgi_schemas_and_exact_saved_model_predictions(self):
        for item in CATALOG:
            code = item['machineCode']
            with self.subTest(machine=code):
                status, schema = post(f'/models/{code}/schema', {}, method='GET')
                self.assertEqual((status, schema), (200, item['schema']))
                status, response = post('/predict/machine', item['request'])
                self.assertEqual((status, response), (200, item['response']))
                self.assertEqual(response['machineCode'], code)
                self.assertTrue(response['isSimulated'])
                self.assertEqual(response['predictionHorizonHours'], 12)
                bundle = registry.load_model(code)
                frame = pd.DataFrame([item['request']['readings']], columns=bundle['features'])
                probability = float(bundle['model'].predict_proba(frame)[0, list(bundle['model'].classes_).index(1)])
                self.assertEqual(response['failureProbability'], round(probability, 4))
                self.assertEqual(response['failurePrediction'], int(probability >= bundle['decision_threshold']))
                self.assertEqual(response['decisionThreshold'], bundle['decision_threshold'])

    def test_every_required_sensor_invalid_numeric_and_extra_inputs_rejected(self):
        for item in CATALOG:
            code, sample = item['machineCode'], item['request']['readings']
            with self.subTest(machine=code):
                for field in sample:
                    missing = {k: v for k, v in sample.items() if k != field}
                    self.assertEqual(post('/predict/machine', dict(machine_code=code, readings=missing))[0], 422)
                first = next(iter(sample))
                for invalid in [True, '12', None, float('nan'), float('inf'), float('-inf')]:
                    self.assertEqual(post('/predict/machine', dict(machine_code=code, readings=dict(sample, **{first: invalid})))[0], 422)
                for readings in [dict(sample, future_failure_hour=1), dict(sample, hours_since_maintenance=-1)]:
                    self.assertEqual(post('/predict/machine', dict(machine_code=code, readings=readings))[0], 422)

    def test_missing_models_fail_closed_without_any_other_load(self):
        for item in CATALOG:
            registry.load_model.cache_clear()
            with self.subTest(machine=item['machineCode']), patch('model_registry.Path.is_file', return_value=False), patch('model_registry.joblib.load') as loader:
                self.assertEqual(post('/predict/machine', item['request'])[0], 503)
                loader.assert_not_called()

    def test_new_cycle_audits_labels_and_maintenance_resets(self):
        for item in CATALOG:
            code = item['machineCode']
            if code in PRESERVED_MACHINES:
                continue
            with self.subTest(machine=code):
                directory = ROOT / 'artifacts' / code
                df = pd.read_csv(directory / 'synthetic_telemetry.csv')
                events = pd.read_csv(directory / 'synthetic_cycle_events.csv')
                metadata = json.loads((directory / 'synthetic_telemetry.metadata.json').read_text())
                self.assertTrue(metadata['is_simulated'])
                self.assertTrue(events.is_simulated.all())
                self.assertEqual(len(events), 180)
                self.assertEqual(metadata['event_audit_sha256'], hashlib.sha256((directory / 'synthetic_cycle_events.csv').read_bytes()).hexdigest())
                np.testing.assert_array_equal(df.hours_since_maintenance, df.hour_in_cycle)
                self.assertTrue((df.groupby('cycle_id').hours_since_maintenance.min() == 0).all())
                merged = df.merge(events, on=['machine_code', 'cycle_id'], validate='many_to_one')
                expected = merged.failure_hour.notna() & (merged.failure_hour > merged.hour_in_cycle) & (merged.failure_hour <= merged.hour_in_cycle + 12)
                np.testing.assert_array_equal(expected.astype(int), merged.failure_within_12h)
                failed = merged.failure_hour.notna()
                self.assertTrue((merged.loc[failed, 'hour_in_cycle'] < merged.loc[failed, 'failure_hour']).all())
                self.assertTrue((merged.loc[~failed, 'hour_in_cycle'] + 12 < merged.loc[~failed, 'planned_hours']).all())
                self.assertEqual(set(df.columns), set(MACHINE_SENSORS[code]) | {'machine_code', 'cycle_id', 'hour_in_cycle', 'failure_within_12h'})

    def test_new_evaluations_and_validation_only_thresholds_recompute(self):
        for item in CATALOG:
            code = item['machineCode']
            if code in PRESERVED_MACHINES:
                continue
            with self.subTest(machine=code):
                directory = ROOT / 'artifacts' / code
                df = pd.read_csv(directory / 'synthetic_telemetry.csv')
                report = json.loads((directory / 'evaluation.json').read_text())
                bundle = registry.load_model(code)
                train, val, test = split_by_cycle(df)
                groups = [set(frame.cycle_id) for frame in [train, val, test]]
                self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
                for name, frame in [('train', train), ('validation', val), ('test', test)]:
                    self.assertEqual(sorted(frame.cycle_id.unique()), report['split_cycle_ids'][name])
                val_p = bundle['model'].predict_proba(val[bundle['features']])[:, 1]
                selected = float(max(np.arange(0.10, 0.81, 0.05), key=lambda t: (fbeta_score(val.failure_within_12h, val_p >= t, beta=2, zero_division=0), t)))
                self.assertEqual(selected, bundle['decision_threshold'])
                probability = bundle['model'].predict_proba(test[bundle['features']])[:, 1]
                predicted = probability >= selected
                y = test.failure_within_12h
                for key, metric in [('test_precision', precision_score), ('test_recall', recall_score), ('test_f1', f1_score)]:
                    self.assertAlmostEqual(report[key], metric(y, predicted, zero_division=0))
                p, r, _ = precision_recall_curve(y, probability)
                self.assertAlmostEqual(report['test_pr_auc_trapezoidal'], auc(r, p))
                self.assertAlmostEqual(report['test_average_precision'], average_precision_score(y, probability))
                self.assertEqual(report['confusion_matrix_0_1'], confusion_matrix(y, predicted, labels=[0, 1]).tolist())
                self.assertEqual(report['class_distribution']['all'], {str(label): int((df.failure_within_12h == label).sum()) for label in [0, 1]})

    def test_preserved_artifact_checksums_and_baseline_predictions(self):
        manifest = json.loads((ROOT / 'backups/full_fleet_before/manifest.json').read_text())
        for name, entry in manifest.items():
            path = Path(name)
            relative = name.removeprefix(str(ROOT) + '/')
            if relative.startswith(('artifacts/CLN-', 'artifacts/ETCH-101/', 'FabCore_CLN101_Training/')) or relative == 'manufacturing_model.joblib':
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry['sha256'], relative)
        baseline = json.loads((ROOT / 'verification/full_fleet/baseline_predictions.json').read_text())
        for code, readings in baseline['requests'].items():
            self.assertEqual(post('/predict/machine', dict(machine_code=code, readings=readings)), (200, baseline['responses'][code]))
        status, response = post('/predict', dict(machine_id='DEMO', air_temperature=298.1, process_temperature=308.6, rotational_speed=1551, torque=42.8, tool_wear=25))
        self.assertEqual((status, response), (200, baseline['responses']['AI4I']))
