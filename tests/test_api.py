from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)
VALID = {
    "harvest_maturity_index": 0.72, "pulp_temperature_c": 22.5,
    "brix": 14.2, "bruising_percentage": 3.5,
    "transport_duration_hours": 12, "cooling_delay_hours": 2,
    "packaging_type": "ventilated_crate"
}

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Mango Shelf-Life Predictor" in response.text

def test_health():
    assert client.get("/health").json()["status"] == "ok"

def test_rejects_invalid_ranges():
    assert client.post("/predict", json=VALID | {"harvest_maturity_index": 1.8}).status_code == 422

def test_rejects_invalid_packaging():
    assert client.post("/predict", json=VALID | {"packaging_type": "open_bag"}).status_code == 422

def test_prediction_execution():
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    value = response.json()["estimated_remaining_shelf_life_days"]
    assert 0 <= value <= 30
