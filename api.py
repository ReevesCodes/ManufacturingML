"""FabCore Manufacturing AI service.

Keeps the original AI4I /predict endpoint and adds a machine-specific
CLN-101 and CLN-102 endpoints for synthetically trained cleaning models.
"""
from __future__ import annotations
from typing import Literal, Union

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import math
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt
from model_registry import load_model, sensor_schema, validate_machine_code

app = FastAPI(title="FabCore Manufacturing AI API")


@app.exception_handler(RequestValidationError)
async def validation_error(request, error):
    # Invalid JSON numeric values can reach validation as NaN/Infinity. Keep
    # validation responses JSON-safe instead of failing to serialize a 422.
    def safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        return value
    return JSONResponse(status_code=422, content={"detail": safe(jsonable_encoder(error.errors()))})

ROOT = Path(__file__).resolve().parent

# Original AI4I model: leave its filename and schema unchanged.
original_bundle = joblib.load(ROOT / "manufacturing_model.joblib") 
original_model = original_bundle["model"]
original_features = original_bundle["features"]
original_threshold = float(original_bundle.get("threshold", 0.30))


class MachineReading(BaseModel):
    machine_id: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


class CLN101Reading(BaseModel):
    machine_id: str = "1"  # FabCore machine ID for CLN-101
    chamber_temperature_c: float
    chamber_pressure_kpa: float
    pump_vibration_mm_s: float
    chemical_flow_l_min: float
    filter_diff_pressure_kpa: float
    motor_current_a: float
    hours_since_maintenance: float


class CLN102Reading(BaseModel):
    """Synthetic CLN-102 inputs; no numeric MES ID has been established."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    machine_id: Literal["CLN-102"]
    chamber_temperature_c: float
    chamber_pressure_kpa: float
    pump_vibration_mm_s: float
    chemical_flow_l_min: float
    filter_diff_pressure_kpa: float
    motor_current_a: float
    hours_since_maintenance: float


def load_cln102_model():
    return load_model("CLN-102")


def load_cln101_model():
    return load_model("CLN-101")


# Preserve the cache-clearing hooks used by existing checks and operators.
load_cln101_model.cache_clear = load_model.cache_clear
load_cln102_model.cache_clear = load_model.cache_clear


class MachineModelReading(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    machine_code: str
    readings: dict[str, Union[StrictFloat, StrictInt]]


@app.get("/models/{machine_code}/schema")
def get_model_schema(machine_code: str):
    return sensor_schema(machine_code)


@app.post("/predict/machine")
def predict_machine(reading: MachineModelReading):
    code = validate_machine_code(reading.machine_code)
    bundle = load_model(code)
    if set(reading.readings) != set(bundle["features"]):
        raise HTTPException(status_code=422, detail="Readings must match the machine sensor schema exactly.")
    if reading.readings.get("hours_since_maintenance", 0) < 0:
        raise HTTPException(status_code=422, detail="Hours since maintenance must be nonnegative.")
    return custom_prediction(code, reading.readings, bundle, code)


def custom_prediction(code, inputs, bundle, machine_id):
    probability = get_probability(bundle["model"], bundle["features"], inputs)
    threshold = float(bundle["decision_threshold"])
    return {
        "machineId": machine_id,
        "machineCode": code,
        "failurePrediction": int(probability >= threshold),
        "failureProbability": round(probability, 4),
        "riskLevel": risk_level(probability, threshold),
        "predictionHorizonHours": sensor_schema(code)["predictionHorizonHours"],
        "decisionThreshold": threshold,
        "modelVersion": bundle["model_version"],
        "isSimulated": bundle["is_simulated"],
    }


def get_probability(model, features, readings):
    frame = pd.DataFrame([readings])[features]
    positive_class_index = list(model.classes_).index(1)
    return float(model.predict_proba(frame)[0][positive_class_index])


def risk_level(probability, threshold):
    if probability >= 0.70:
        return "HIGH"
    if probability >= threshold:
        return "MODERATE"
    return "LOW"


@app.get("/")
def home():
    return {
        "message": "Manufacturing AI API is running",
        "endpoints": ["/predict", "/predict/cln101", "/predict/cln102", "/predict/machine", "/models/{machine_code}/schema"],
    }


@app.post("/predict")
def predict_failure(reading: MachineReading):
    """Existing five-input AI4I endpoint; compatible with today's C# API."""
    inputs = {
        "Air temperature": reading.air_temperature,
        "Process temperature": reading.process_temperature,
        "Rotational speed": reading.rotational_speed,
        "Torque": reading.torque,
        "Tool wear": reading.tool_wear,
    }
    probability = get_probability(original_model, original_features, inputs)
    prediction = int(probability >= original_threshold)
    return {
        "machineId": reading.machine_id,
        "failurePrediction": prediction,
        "failureProbability": round(probability, 4),
        "riskLevel": risk_level(probability, original_threshold),
    }


@app.post("/predict/cln101")
def predict_cln101(reading: CLN101Reading):
    """Seven-input CLN-101 model; prototype only, trained on synthetic data."""
    if reading.machine_id not in {"1", "CLN-101"}:
        raise HTTPException(
            status_code=422,
            detail="This endpoint is for CLN-101 (FabCore machine ID 1) only.",
        )

    return custom_prediction("CLN-101", reading.model_dump(exclude={"machine_id"}),
                             load_cln101_model(), reading.machine_id)


@app.post("/predict/cln102")
def predict_cln102(reading: CLN102Reading):
    """Simulated failure risk from CLN-102's own saved model and threshold."""
    return custom_prediction("CLN-102", reading.model_dump(exclude={"machine_id"}),
                             load_cln102_model(), reading.machine_id)
