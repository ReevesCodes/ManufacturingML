import hashlib
import importlib.util
import unittest
from pathlib import Path

import joblib
import pandas as pd

from generate_dataset import generate
from train_model import split_by_cycle
from training.config import ROOT, PROFILES


class PipelineTests(unittest.TestCase):
    def test_artifact_contract_and_group_isolation(self):
        for machine in PROFILES:
            directory = ROOT / 'artifacts' / machine
            df = pd.read_csv(directory / 'synthetic_telemetry.csv')
            train, validation, test = split_by_cycle(df)
            groups = [set(x.cycle_id) for x in (train, validation, test)]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            self.assertEqual(len(train) + len(validation) + len(test), len(df))
            bundle = joblib.load(directory / 'model.joblib')
            self.assertEqual(bundle['machine_code'], machine)
            self.assertTrue(bundle['is_simulated'])
            self.assertEqual(bundle['metadata']['dataset_sha256'], hashlib.sha256((directory / 'synthetic_telemetry.csv').read_bytes()).hexdigest())
            self.assertEqual(bundle['model'].predict_proba(test[bundle['features']]).shape, (len(test), 2))

    def test_generation_reproducible(self):
        # Generate in a temporary root so existing artifacts cannot be changed.
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory, patch('generate_dataset.ROOT', Path(directory)):
            for machine in PROFILES:
                generate(machine, 20)
                path = Path(directory) / 'artifacts' / machine / 'synthetic_telemetry.csv'
                first = path.read_bytes()
                generate(machine, 20)
                self.assertEqual(first, path.read_bytes())

    def test_legacy_api_contract(self):
        import api
        self.assertTrue({'/', '/predict', '/predict/cln101'} <= {route.path for route in api.app.routes})
        self.assertEqual(api.home()['message'], 'Manufacturing AI API is running')
        response = api.predict_failure(api.MachineReading(machine_id='DEMO-001', air_temperature=298.1,
                    process_temperature=308.6, rotational_speed=1551, torque=42.8, tool_wear=25))
        self.assertEqual(set(response), {'machineId', 'failurePrediction', 'failureProbability', 'riskLevel'})
        spec = importlib.util.spec_from_file_location('legacy_demo', ROOT / 'FabCore_CLN101_Training/03_try_prediction.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for sensors in module.SCENARIOS.values():
            response = api.predict_cln101(api.CLN101Reading(**sensors))
            self.assertEqual(response['machineCode'], 'CLN-101')
            self.assertTrue(response['isSimulated'])
        with self.assertRaises(api.HTTPException) as error:
            api.predict_cln101(api.CLN101Reading(**dict(sensors, machine_id='CLN-102')))
        self.assertEqual(error.exception.status_code, 422)



if __name__ == '__main__':
    unittest.main()
