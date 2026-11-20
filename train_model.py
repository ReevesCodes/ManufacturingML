"""Train, validate and evaluate a configured machine-specific synthetic model.

Splits by maintenance cycle to avoid the same cycle appearing in train
and test sets. Decision threshold is selected on validation data only.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import hashlib
import platform
import sklearn
from training.config import ROOT, PROFILES, arguments
from training.sensors import CLEANING_SENSORS, MACHINE_SENSORS, features_for

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
)
from sklearn.model_selection import GroupShuffleSplit

FEATURES = list(CLEANING_SENSORS)  # Legacy import; each training run selects its own schema.
TARGET = 'failure_within_12h'


def split_by_cycle(df):
    # 20% held out for final test, 20% held out for validation, 60% train.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_val_index, test_index = next(
        splitter.split(df, df[TARGET], groups=df['cycle_id'])
    )
    train_val = df.iloc[train_val_index].reset_index(drop=True)
    test = df.iloc[test_index].reset_index(drop=True)

    val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=17)
    train_index, val_index = next(
        val_splitter.split(train_val, train_val[TARGET], groups=train_val['cycle_id'])
    )
    train = train_val.iloc[train_index]
    validation = train_val.iloc[val_index]
    assert not (set(train.cycle_id) & set(validation.cycle_id))
    assert not (set(train.cycle_id) & set(test.cycle_id))
    assert not (set(validation.cycle_id) & set(test.cycle_id))
    return train, validation, test


def train_machine(machine):
    FEATURES = features_for(machine)
    directory = ROOT / 'artifacts' / machine
    DATA_PATH = directory / 'synthetic_telemetry.csv'
    MODEL_PATH = directory / 'model.joblib'
    REPORT_PATH = directory / 'evaluation.json'
    VERSION = machine + '-' + PROFILES[machine].get('model_version_suffix', 'SYN-RF-v2')

    if not DATA_PATH.exists():
        raise SystemExit('Dataset missing. Run: python generate_dataset.py --all')

    df = pd.read_csv(DATA_PATH)
    for col in FEATURES + [TARGET, 'cycle_id', 'hour_in_cycle', 'machine_code']:
        if col not in df.columns:
            raise SystemExit(f'Missing expected CSV column: {col}')
    if set(df.machine_code) != {machine}:
        raise SystemExit('Dataset machine_code does not match requested machine')
    if df.duplicated(['cycle_id', 'hour_in_cycle']).any():
        raise SystemExit('Duplicate cycle/hour readings')
    if not df[TARGET].isin([0, 1]).all() or not np.isfinite(df[FEATURES]).all().all():
        raise SystemExit('Invalid labels or non-finite features')
    train, val, test = split_by_cycle(df)
    for name, frame in [('training', train), ('validation', val), ('test', test)]:
        if frame[TARGET].nunique() != 2:
            raise SystemExit(f'{name} split has only one class; regenerate the dataset.')

    rf = RandomForestClassifier(
        n_estimators=180,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        random_state=PROFILES[machine].get('model_seed', 42),
        n_jobs=-1,
    )
    rf.fit(train[FEATURES], train[TARGET])

    val_probability = rf.predict_proba(val[FEATURES])[:, 1]
    threshold_candidates = np.arange(0.10, 0.81, 0.05)
    # F2 favours recall, useful when missing failures is costly.
    best_threshold = float(max(
        threshold_candidates,
        key=lambda threshold: (
            fbeta_score(val[TARGET], val_probability >= threshold, beta=2, zero_division=0),
            threshold,
        )
    ))

    test_probability = rf.predict_proba(test[FEATURES])[:, 1]
    test_prediction = (test_probability >= best_threshold).astype(int)
    conf = confusion_matrix(test[TARGET], test_prediction, labels=[0, 1]).tolist()

    pr_precision, pr_recall, _ = precision_recall_curve(test[TARGET], test_probability)
    report = {
        'machine_code': machine,
        'model_version': VERSION,
        'is_simulated': True,
        'features': FEATURES,
        'measurement_units': {name: MACHINE_SENSORS[machine][name][1] for name in FEATURES},
        'dataset_sha256': hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        'generator_metadata': json.loads(DATA_PATH.with_suffix('.metadata.json').read_text()),
        'software': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__},
        'split_method': 'GroupShuffleSplit by maintenance cycle; 60/20/20',
        'split_seeds': [42, 17],
        'model_parameters': rf.get_params(),
        'threshold_selection': 'Maximum validation F2; ties prefer higher threshold',
        'split_cycle_ids': {name: sorted(int(x) for x in frame.cycle_id.unique())
                            for name, frame in [('train', train), ('validation', val), ('test', test)]},
        'classification_report': classification_report(test[TARGET], test_prediction,
                                  labels=[0, 1], output_dict=True, zero_division=0),
        'data_origin': 'Synthetic simulation, not actual equipment readings',
        'prediction_target': 'Simulated equipment failure within next 12 hours',
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'n_samples': {
            'train': len(train), 'validation': len(val), 'test': len(test),
        },
        'cycles': {
            'train': int(train.cycle_id.nunique()),
            'validation': int(val.cycle_id.nunique()),
            'test': int(test.cycle_id.nunique()),
        },
        'class_distribution': {name: {str(label): int((frame[TARGET] == label).sum()) for label in [0, 1]}
                               for name, frame in [('all', df), ('train', train), ('validation', val), ('test', test)]},
        'test_f1': float(f1_score(test[TARGET], test_prediction, zero_division=0)),
        'test_pr_auc_trapezoidal': float(auc(pr_recall, pr_precision)),
        'test_positive_rate': float(test[TARGET].mean()),
        'decision_threshold': best_threshold,
        'test_precision': float(precision_score(test[TARGET], test_prediction, zero_division=0)),
        'test_recall': float(recall_score(test[TARGET], test_prediction, zero_division=0)),
        'test_average_precision': float(average_precision_score(test[TARGET], test_probability)),
        'confusion_matrix_0_1': conf,
        'feature_importance': {
            feature: float(score)
            for feature, score in zip(FEATURES, rf.feature_importances_)
        },
    }
    joblib.dump({
        'metadata': report,
        'model': rf,
        'features': FEATURES,
        'decision_threshold': best_threshold,
        'machine_code': machine,
        'model_version': VERSION,
        'label': 'failure_within_12h',
        'is_simulated': True,
    }, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
    (directory / 'model_metadata.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    sample = df.iloc[0][FEATURES].astype(float).to_dict()
    probability = float(rf.predict_proba(pd.DataFrame([sample], columns=FEATURES))[0, 1])
    (directory / 'example_request.json').write_text(json.dumps(dict(machine_code=machine, readings=sample), indent=2))
    (directory / 'example_response.json').write_text(json.dumps(dict(
        machineId=machine, machineCode=machine, failurePrediction=int(probability >= best_threshold),
        failureProbability=round(probability, 4), riskLevel='HIGH' if probability >= 0.70 else 'MODERATE' if probability >= best_threshold else 'LOW',
        predictionHorizonHours=12, decisionThreshold=best_threshold, modelVersion=VERSION, isSimulated=True), indent=2))

    print(f'\n{machine} Random Forest - SYNTHETIC HELD-OUT test')
    print('Training / validation / test observations:', len(train), len(val), len(test))
    print('Independent cycles:', train.cycle_id.nunique(), val.cycle_id.nunique(), test.cycle_id.nunique())
    print('Threshold selected using validation data:', round(best_threshold, 3))
    print(classification_report(test[TARGET], test_prediction, target_names=['No failure in 12h', 'Failure within 12h'], zero_division=0))
    print('Confusion matrix [[true 0 predicted 0, true 0 predicted 1],')
    print('                  [true 1 predicted 0, true 1 predicted 1]]:')
    print(conf)
    print('Failure precision / recall / F1:', report['test_precision'], report['test_recall'], report['test_f1'])
    print('PR-AUC (trapezoidal):', report['test_pr_auc_trapezoidal'])
    print('Average precision (step-integrated PR summary):', report['test_average_precision'])
    print('Class distribution (0 / 1 counts):', report['class_distribution'])
    print('Most influential features:')
    for k, v in sorted(report['feature_importance'].items(), key=lambda pair: -pair[1]):
        print(f'  {k:28} {v:.3f}')
    print(f'\nSaved model: {MODEL_PATH}')
    print(f'Saved report: {REPORT_PATH}')
    print('WARNING: Evaluation is on synthetic data from the same simulator, not real machinery.')


if __name__ == '__main__':
    args, machines = arguments('Train and evaluate SYNTHETIC machine models')
    for machine in machines:
        if args.skip_existing and (ROOT / 'artifacts' / machine / 'model.joblib').exists():
            print(f'{machine}: existing model preserved')
            continue
        train_machine(machine)
