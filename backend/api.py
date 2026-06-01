from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).resolve().parent / "model" / "heart_failure_model.joblib"


class PatientInput(BaseModel):
    age: float = Field(..., examples=[65])
    anaemia: int = Field(..., ge=0, le=1, examples=[0])
    creatinine_phosphokinase: float = Field(..., examples=[250])
    diabetes: int = Field(..., ge=0, le=1, examples=[1])
    ejection_fraction: float = Field(..., examples=[35])
    high_blood_pressure: int = Field(..., ge=0, le=1, examples=[1])
    platelets: float = Field(..., examples=[263000])
    serum_creatinine: float = Field(..., examples=[1.3])
    serum_sodium: float = Field(..., examples=[136])
    sex: int = Field(..., ge=0, le=1, examples=[1])
    smoking: int = Field(..., ge=0, le=1, examples=[0])


class PredictionResponse(BaseModel):
    mortality_risk_probability: float
    predicted_death_event: int
    threshold: float
    risk_level: str


def load_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run backend/train_model.py first."
        )
    return joblib.load(MODEL_PATH)


app = FastAPI(
    title="Heart Failure Mortality Risk Prediction API",
    description="Predicts heart failure death event risk using a saved stacking ensemble model.",
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    app.state.artifact = load_artifact()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Heart Failure Mortality Risk Prediction API"}


@app.get("/health")
def health() -> dict[str, bool]:
    return {"model_loaded": hasattr(app.state, "artifact")}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    artifact = app.state.artifact
    return {
        "target": artifact["target"],
        "features": artifact["features"],
        "threshold": artifact["threshold"],
        "test_metrics": artifact.get("test_metrics", {}),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput) -> PredictionResponse:
    artifact = app.state.artifact
    features = artifact["features"]
    input_df = pd.DataFrame([patient.model_dump()])

    missing_features = [feature for feature in features if feature not in input_df.columns]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {', '.join(missing_features)}",
        )

    probability = float(artifact["model"].predict_proba(input_df[features])[:, 1][0])
    threshold = float(artifact["threshold"])
    predicted_death_event = int(probability >= threshold)

    if probability >= 0.7:
        risk_level = "high"
    elif probability >= 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"

    return PredictionResponse(
        mortality_risk_probability=round(probability, 4),
        predicted_death_event=predicted_death_event,
        threshold=round(threshold, 4),
        risk_level=risk_level,
    )
