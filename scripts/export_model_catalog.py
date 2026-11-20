"""Export real saved-model schemas/examples/evaluations; never fit or alter models.

Run from the project root: .venv/bin/python -m scripts.export_model_catalog
The catalog is a verification fixture, not an application routing table.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (auc, average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score)

import api
from model_registry import load_model, model_path, sensor_schema
from train_model import split_by_cycle
from training.config import ROOT


def export():
    inventory = json.loads((ROOT / 'verification/full_fleet/inventory.json').read_text())
    catalog = []
    for machine in inventory:
        code = machine['machineCode']
        bundle = load_model(code)
        directory = ROOT / 'artifacts' / code
        data_path = (ROOT / 'FabCore_CLN101_Training/cln101_synthetic_telemetry.csv') if code == 'CLN-101' else directory / 'synthetic_telemetry.csv'
        df = pd.read_csv(data_path)
        train, validation, test = split_by_cycle(df)
        features = bundle['features']
        y = test.failure_within_12h
        probabilities = bundle['model'].predict_proba(test[features])[:, list(bundle['model'].classes_).index(1)]
        threshold = float(bundle['decision_threshold'])
        predictions = (probabilities >= threshold).astype(int)
        precision, recall, _ = precision_recall_curve(y, probabilities)
        report = dict(
            machine_code=code, model_version=bundle['model_version'], is_simulated=True,
            data_origin='Synthetic simulator evaluation; not industrial validation',
            model_path=str(model_path(code).relative_to(ROOT)), dataset_path=str(data_path.relative_to(ROOT)),
            dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
            model_sha256=hashlib.sha256(model_path(code).read_bytes()).hexdigest(),
            decision_threshold=threshold, features=features,
            feature_importance=dict(zip(features, map(float, bundle['model'].feature_importances_))),
            test_precision=float(precision_score(y, predictions, zero_division=0)),
            test_recall=float(recall_score(y, predictions, zero_division=0)),
            test_f1=float(f1_score(y, predictions, zero_division=0)),
            test_pr_auc_trapezoidal=float(auc(recall, precision)),
            test_average_precision=float(average_precision_score(y, probabilities)),
            classification_report=classification_report(y, predictions, output_dict=True, zero_division=0),
            confusion_matrix_0_1=confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
            class_distribution={name: {str(label): int((frame.failure_within_12h == label).sum()) for label in [0, 1]}
                                for name, frame in [('all', df), ('train', train), ('validation', validation), ('test', test)]},
            split_cycle_ids={name: sorted(map(int, frame.cycle_id.unique())) for name, frame in [('train', train), ('validation', validation), ('test', test)]},
            n_samples={name: len(frame) for name, frame in [('train', train), ('validation', validation), ('test', test)]},
            threshold_selection='Existing validation-selected saved threshold; not changed during export',
        )
        original_path = (ROOT / 'FabCore_CLN101_Training/cln101_report.json') if code == 'CLN-101' else directory / 'evaluation.json'
        original = json.loads(original_path.read_text())
        for metric in ['test_precision', 'test_recall', 'test_average_precision']:
            if abs(original[metric] - report[metric]) > 1e-10:
                raise ValueError(f'{code}: reconstructed evaluation differs from original {metric}')
        # Extra serving report distinguishes CLN-101's preserved v1 from its
        # newer, unused artifacts/CLN-101 v2 candidate. No original is overwritten.
        (directory / 'serving_evaluation.json').write_text(json.dumps(report, indent=2))
        schema = sensor_schema(code)
        sample = dict(machine_code=code, readings=df.iloc[0][features].astype(float).to_dict())
        response = api.predict_machine(api.MachineModelReading(**sample))
        for filename, content in [('example_request.json', sample), ('example_response.json', response),
                                  ('model_metadata.json', dict(report, schema=schema))]:
            if not (directory / filename).exists():
                (directory / filename).write_text(json.dumps(content, indent=2))
        catalog.append(dict(machine, schema=schema, request=sample, response=response, evaluation=report))
    destination = ROOT / 'verification/full_fleet/model_catalog.json'
    destination.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    print(f'Exported {len(catalog)} saved-model schemas, requests, responses and independently checked reports to {destination}')
    return catalog


if __name__ == '__main__':
    export()
