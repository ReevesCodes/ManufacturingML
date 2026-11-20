import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import api
import model_registry as registry
from test_cln102_api import REQUEST, post


class RegistryTests(unittest.TestCase):
    def tearDown(self):
        registry.load_model.cache_clear()

    def test_schema_and_generic_match_existing_cleaning_routes(self):
        for code, legacy in [('CLN-101', '/predict/cln101'), ('CLN-102', '/predict/cln102')]:
            status, schema = post(f'/models/{code}/schema', {}, method='GET')
            self.assertEqual(status, 200)
            bundle = registry.load_model(code)
            self.assertEqual([f['name'] for f in schema['features']], bundle['features'])
            self.assertEqual(schema['decisionThreshold'], bundle['decision_threshold'])
            self.assertTrue(all(f['unit'] and f['required'] for f in schema['features']))
            payload = dict(REQUEST, machine_id=code)
            _, expected = post(legacy, payload)
            status, actual = post('/predict/machine', dict(machine_code=code,
                                  readings={k: v for k, v in payload.items() if k != 'machine_id'}))
            self.assertEqual(status, 200)
            self.assertEqual(actual, expected)

    def test_unknown_model_never_falls_back(self):
        status, schema = post('/models/UNKNOWN-999/schema', {}, method='GET')
        self.assertEqual(status, 200)
        self.assertFalse(schema['available'])
        self.assertEqual(schema['message'], 'Custom model not available')
        status, body = post('/predict/machine', dict(machine_code='UNKNOWN-999', readings={}))
        self.assertEqual(status, 404)
        self.assertEqual(body['detail'], 'Custom model not available')

    def test_invalid_identity_and_exact_sensor_schema(self):
        sensors = {k: v for k, v in REQUEST.items() if k != 'machine_id'}
        for code in ['cln-102', 'CLN-101/../CLN-102', '../../model.joblib', '2', '']:
            self.assertEqual(post('/predict/machine', dict(machine_code=code, readings=sensors))[0], 422)
        for readings in [{}, dict(sensors, unexpected=1), dict(sensors, motor_current_a=True),
                         dict(sensors, motor_current_a='8'), dict(sensors, hours_since_maintenance=-1)]:
            self.assertEqual(post('/predict/machine', dict(machine_code='CLN-102', readings=readings))[0], 422)

    def test_symlink_rejected_before_deserialization(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'cln101_model.joblib').symlink_to(registry.ROOT / 'FabCore_CLN101_Training/cln101_model.joblib')
            with patch('model_registry.ROOT', root), patch('model_registry.joblib.load') as loader:
                with self.assertRaises(api.HTTPException) as error:
                    registry.model_path('CLN-101')
                self.assertEqual(error.exception.status_code, 503)
                loader.assert_not_called()

    def test_machine_identity_in_bundle_is_checked(self):
        registry.load_model.cache_clear()
        real = api.joblib.load(api.ROOT / 'artifacts/CLN-102/model.joblib')
        with patch('model_registry.joblib.load', return_value=dict(real, machine_code='CLN-101')):
            self.assertEqual(post('/predict/machine', dict(machine_code='CLN-102', readings={}))[0], 503)
