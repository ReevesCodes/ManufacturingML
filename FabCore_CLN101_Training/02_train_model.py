"""Train, validate and evaluate a machine-specific CLN-101 model.

Splits by maintenance cycle to avoid the same cycle appearing in train
and test sets. Decision threshold is selected on validation data only.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

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
)
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / 'cln101_synthetic_telemetry.csv'
MODEL_PATH = HERE / 'cln101_model.joblib'
REPORT_PATH = HERE / 'cln101_report.json'

FEATURES = [
    'chamber_temperature_c',
    'chamber_pressure_kpa',
    'pump_vibration_mm_s',
    'chemical_flow_l_min',
    'filter_diff_pressure_kpa',
    'motor_current_a',
    'hours_since_maintenance',
]
TARGET = 'failure_within_12h'
VERSION = 'CLN101-SYN-RF-v1'


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


def main():
    if not DATA_PATH.exists():
        raise SystemExit('Dataset missing. Run: python 01_generate_dataset.py')

    df = pd.read_csv(DATA_PATH)
    for col in FEATURES + [TARGET, 'cycle_id']:
        if col not in df.columns:
            raise SystemExit(f'Missing expected CSV column: {col}')
    train, val, test = split_by_cycle(df)
    for name, frame in [('training', train), ('validation', val), ('test', test)]:
        if frame[TARGET].nunique() != 2:
            raise SystemExit(f'{name} split has only one class; regenerate the dataset.')

    rf = RandomForestClassifier(
        n_estimators=180,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        random_state=42,
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

    report = {
        'machine_code': 'CLN-101',
        'model_version': VERSION,
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
        'test_positive_rate': float(test[TARGET].mean()),
        'decision_threshold': round(best_threshold, 4),
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
        'model': rf,
        'features': FEATURES,
        'decision_threshold': best_threshold,
        'machine_code': 'CLN-101',
        'model_version': VERSION,
        'label': 'failure_within_12h',
        'is_simulated': True,
    }, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print('\nCLN-101 Random Forest - final HELD-OUT test')
    print('Training / validation / test observations:', len(train), len(val), len(test))
    print('Independent cycles:', train.cycle_id.nunique(), val.cycle_id.nunique(), test.cycle_id.nunique())
    print('Threshold selected using validation data:', round(best_threshold, 3))
    print(classification_report(test[TARGET], test_prediction, target_names=['No failure in 12h', 'Failure within 12h'], zero_division=0))
    print('Confusion matrix [[true 0 predicted 0, true 0 predicted 1],')
    print('                  [true 1 predicted 0, true 1 predicted 1]]:')
    print(conf)
    print('Test average precision:', round(report['test_average_precision'], 4))
    print('Most influential features:')
    for k, v in sorted(report['feature_importance'].items(), key=lambda pair: -pair[1]):
        print(f'  {k:28} {v:.3f}')
    print(f'\nSaved model: {MODEL_PATH}')
    print(f'Saved report: {REPORT_PATH}')
    print('WARNING: Evaluation is on synthetic data from the same simulator, not real machinery.')


if __name__ == '__main__':
    main()
