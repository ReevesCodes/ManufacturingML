"""Server-owned model allowlist. Request data can never specify artifact paths.

Thresholds, feature order and versions come from the selected saved bundle.
CLN-101 deliberately keeps its currently served v1 model, not the new v2 model.
"""
import logging
import math
import re
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import sklearn
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent

from training.sensors import CLEANING_SENSORS, ETCHING_SENSORS, MACHINE_SENSORS

# Preserve the old cleaning-catalog import for existing callers.
SENSORS = CLEANING_SENSORS

@dataclass(frozen=True)
class ModelEntry:
    paths: tuple
    feature_names: tuple = tuple(SENSORS)
    label: str = 'failure_within_12h'
    horizon_hours: int = 12
    is_simulated: bool = True


REGISTRY = {
    'CLN-101': ModelEntry((
        'cln101_model.joblib',
        'FabCore_CLN101_Training/cln101_model.joblib',
        'Fabcore_CLN101_Training/cln101_model.joblib',
    )),
    'CLN-102': ModelEntry(('artifacts/CLN-102/model.joblib',)),
    'ETCH-101': ModelEntry(('artifacts/ETCH-101/model.joblib',), tuple(ETCHING_SENSORS)),
}
# Keys come only from the server-owned schema catalog, never from a request path.
for code, sensors in MACHINE_SENSORS.items():
    if code not in REGISTRY:
        REGISTRY[code] = ModelEntry((f'artifacts/{code}/model.joblib',), tuple(sensors))


def validate_machine_code(code):
    # Use one canonical spelling; no path separators or numeric aliases here.
    if not isinstance(code, str) or not re.fullmatch(r'[A-Z][A-Z0-9]{1,15}-[0-9]{3}', code):
        raise HTTPException(status_code=422, detail='Invalid machine code.')
    return code


def model_path(code):
    validate_machine_code(code)
    if code not in REGISTRY:
        raise HTTPException(status_code=404, detail='Custom model not available')
    for relative in REGISTRY[code].paths:
        path = ROOT / relative
        # Reject symlinks, including parent directories, before deserializing.
        if path.resolve() != path.absolute() or ROOT not in path.resolve().parents:
            raise HTTPException(status_code=503, detail='Model path is not whitelisted.')
        if path.is_file():
            return path
    raise HTTPException(status_code=503, detail='Custom model not available')


@lru_cache(maxsize=32)
def load_model(code):
    path = model_path(code)
    try:
        bundle = joblib.load(path)  # Only operator-managed, explicitly allowed artifacts.
        features = bundle['features']
        model = bundle['model']
        threshold = float(bundle['decision_threshold'])
        if (bundle.get('machine_code') != code
                or bundle.get('is_simulated') is not REGISTRY[code].is_simulated
                or bundle.get('label') != REGISTRY[code].label
                or not isinstance(bundle.get('model_version'), str)
                or not bundle['model_version'].strip()
                or features != list(REGISTRY[code].feature_names)
                or list(model.feature_names_in_) != features
                or set(model.classes_) != {0, 1}
                or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError('Bundle does not match registered machine/schema')
        metadata = bundle.get('metadata')
        if metadata is not None:
            if (metadata.get('machine_code') != code or metadata.get('features') != features
                    or metadata.get('model_version') != bundle['model_version']
                    or metadata.get('is_simulated') is not True
                    or metadata.get('generator_metadata', {}).get('horizon_hours') != REGISTRY[code].horizon_hours
                    or not math.isclose(threshold, metadata.get('decision_threshold', -1), abs_tol=1e-12)):
                raise ValueError('Inconsistent saved metadata')
            trained = metadata.get('software', {}).get('sklearn')
            if trained != sklearn.__version__:
                logging.getLogger(__name__).warning('%s sklearn trained=%s runtime=%s', code, trained, sklearn.__version__)
        return bundle
    except (KeyError, TypeError, ValueError, AttributeError, OSError, EOFError, pickle.UnpicklingError) as error:
        logging.getLogger(__name__).error('Invalid %s artifact: %s', code, error)
        raise HTTPException(status_code=503, detail='Custom model metadata is incompatible.') from error


def sensor_schema(code):
    validate_machine_code(code)
    if code not in REGISTRY:
        return dict(machineCode=code, available=False, message='Custom model not available', features=[])
    bundle = load_model(code)
    entry = REGISTRY[code]
    sensors = MACHINE_SENSORS[code]
    return dict(
        machineCode=code, available=True, isSimulated=entry.is_simulated,
        modelVersion=bundle['model_version'], decisionThreshold=float(bundle['decision_threshold']),
        predictionHorizonHours=entry.horizon_hours,
        features=[dict(name=name, label=sensors[name][0], unit=sensors[name][1],
                       type='number', required=True,
                       minimum=0 if name == 'hours_since_maintenance' else None)
                  for name in bundle['features']],
    )
