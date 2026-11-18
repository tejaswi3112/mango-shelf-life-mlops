import pandas as pd
from config import FEATURES, TARGET, DATA_DIR, FEATURE_DIR

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["heat_exposure_index"] = (
        (out["pulp_temperature_c"] - 12).clip(lower=0)
        * out["transport_duration_hours"] / 24
    )
    out["handling_damage_index"] = (
        out["bruising_percentage"] * 0.08
        + out["cooling_delay_hours"] * 0.55
    )
    out["packaging_type"] = out["packaging_type"].map({
        "ventilated_crate": 0, "corrugated_box": 1, "plastic_crate": 2
    })
    return out

def build_feature_files():
    for version in ("v1", "v2"):
        raw = pd.read_csv(DATA_DIR / f"mango_shipments_{version}.csv")
        features = engineer_features(raw)
        features.to_csv(FEATURE_DIR / f"mango_features_{version}.csv", index=False)
    print("Reusable feature files created in data/features/")

if __name__ == "__main__":
    build_feature_files()
