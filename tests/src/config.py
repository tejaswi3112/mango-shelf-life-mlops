from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
FEATURE_DIR = DATA_DIR / "features"
for folder in (DATA_DIR, ARTIFACT_DIR, FEATURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "harvest_maturity_index", "pulp_temperature_c", "brix",
    "bruising_percentage", "transport_duration_hours",
    "cooling_delay_hours", "packaging_type",
    "heat_exposure_index", "handling_damage_index"
]
TARGET = "remaining_shelf_life_days"
