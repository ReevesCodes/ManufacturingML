"""Exercise the FastAPI ASGI app, including HTTP request validation."""
import asyncio
import json
import unittest
from unittest.mock import patch

import pandas as pd

import api

REQUEST = {
    'machine_id': 'CLN-102',
    'chamber_temperature_c': 41.0,
    'chamber_pressure_kpa': 105.0,
    'pump_vibration_mm_s': 1.8,
    'chemical_flow_l_min': 19.0,
    'filter_diff_pressure_kpa': 6.0,
    'motor_current_a': 8.0,
    'hours_since_maintenance': 24,
}


def post(path, payload, method="POST"):
    async def request():
        messages = []
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': json.dumps(payload).encode(), 'more_body': False}
            return {'type': 'http.disconnect'}

        async def send(message):
            messages.append(message)

        await api.app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                       'method': method, 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
                       'query_string': b'', 'root_path': '', 'headers': [(b'content-type', b'application/json')],
                       'server': ('testserver', 80), 'client': ('127.0.0.1', 1234)}, receive, send)
        status = next(m['status'] for m in messages if m['type'] == 'http.response.start')
        body = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
        return status, json.loads(body)
    return asyncio.run(request())


class CLN102EndpointTests(unittest.TestCase):
    def test_saved_model_prediction(self):
        bundle = api.load_cln102_model()
        status, response = post('/predict/cln102', REQUEST)
        self.assertEqual(status, 200)
        frame = pd.DataFrame([REQUEST])[bundle['features']]
        probability = float(bundle['model'].predict_proba(frame)[0, list(bundle['model'].classes_).index(1)])
        self.assertEqual(response['failureProbability'], round(probability, 4))
        self.assertEqual(response['failurePrediction'], int(probability >= bundle['decision_threshold']))
        self.assertEqual(response['decisionThreshold'], bundle['decision_threshold'])
        self.assertEqual(response['machineCode'], 'CLN-102')
        self.assertTrue(response['isSimulated'])

    def test_identity_and_input_validation(self):
        for identity in ['CLN-101', '1', '2', 'unknown', None]:
            self.assertEqual(post('/predict/cln102', dict(REQUEST, machine_id=identity))[0], 422)
        for field in REQUEST:
            payload = dict(REQUEST)
            del payload[field]
            self.assertEqual(post('/predict/cln102', payload)[0], 422)
        self.assertEqual(post('/predict/cln102', dict(REQUEST, extra_sensor=3))[0], 422)
        self.assertEqual(post('/predict/cln102', dict(REQUEST, motor_current_a='invalid'))[0], 422)

    def test_missing_and_wrong_machine_artifact(self):
        api.load_cln102_model.cache_clear()
        try:
            with patch('api.Path.is_file', return_value=False):
                self.assertEqual(post('/predict/cln102', REQUEST)[0], 503)
            real = api.joblib.load(api.ROOT / 'artifacts/CLN-102/model.joblib')
            with patch('api.joblib.load', return_value=dict(real, machine_code='CLN-101')):
                self.assertEqual(post('/predict/cln102', REQUEST)[0], 503)
        finally:
            api.load_cln102_model.cache_clear()

    def test_existing_endpoints_over_http(self):
        self.assertEqual(post('/predict', dict(machine_id='DEMO', air_temperature=298.1,
                             process_temperature=308.6, rotational_speed=1551, torque=42.8, tool_wear=25))[0], 200)
        self.assertEqual(post('/predict/cln101', dict(REQUEST, machine_id='CLN-101'))[0], 200)


if __name__ == '__main__':
    unittest.main()
