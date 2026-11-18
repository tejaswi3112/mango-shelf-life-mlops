import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from src.config import ARTIFACT_DIR
from src.features import engineer_features

app = FastAPI(title="Mango Remaining Shelf-Life Predictor", version="1.0.0",
              description="Predicts remaining marketable mango shelf life in days.")
MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"

class ShipmentInput(BaseModel):
    harvest_maturity_index: float = Field(ge=0.0, le=1.0)
    pulp_temperature_c: float = Field(ge=5.0, le=45.0)
    brix: float = Field(ge=5.0, le=25.0)
    bruising_percentage: float = Field(ge=0.0, le=100.0)
    transport_duration_hours: float = Field(ge=0.0, le=120.0)
    cooling_delay_hours: float = Field(ge=0.0, le=48.0)
    packaging_type: str

    @field_validator("packaging_type")
    @classmethod
    def valid_packaging(cls, value):
        allowed = {"ventilated_crate", "corrugated_box", "plastic_crate"}
        if value not in allowed:
            raise ValueError(f"packaging_type must be one of {sorted(allowed)}")
        return value

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))

@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}

@app.post("/predict")
def predict(payload: ShipmentInput):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained. Run python -m src.generate_data, python -m src.features and python -m src.train.")
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([payload.model_dump()])
    engineered = engineer_features(row)
    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prediction = float(model.predict(engineered[metadata["features"]])[0])
    prediction = max(0.0, min(30.0, prediction))
    return {
        "estimated_remaining_shelf_life_days": round(prediction, 2),
        "unit": "days", "model": metadata["selected_model"],
        "note": "Synthetic-data demonstration; validate with real shipment observations."
    }
