import numpy as np
import pandas as pd
from src.config import DATA_DIR

RNG = np.random.default_rng(42)
PACKAGING = ["ventilated_crate", "corrugated_box", "plastic_crate"]

def make_shipments(n: int, version: int) -> pd.DataFrame:
    """Create plausible synthetic shipment records; V2 has longer transport/cooling delays."""
    maturity = RNG.uniform(0.45, 0.95, n)
    pulp_temp = RNG.normal(24 if version == 1 else 27, 3.5, n).clip(15, 38)
    brix = RNG.normal(12.5 + 2.0 * (maturity - 0.65), 1.2, n).clip(8, 20)
    bruising = RNG.beta(1.8, 10, n) * 35
    transport = RNG.gamma(2.8, 5.0 if version == 1 else 7.0, n).clip(2, 72)
    cooling = RNG.gamma(2.0, 2.0 if version == 1 else 3.2, n).clip(0.2, 24)
    packaging = RNG.choice(PACKAGING, n, p=[0.4, 0.35, 0.25])

    heat = np.maximum(pulp_temp - 12, 0) * transport / 24
    damage = bruising * 0.08 + cooling * 0.55
    pack_penalty = pd.Series(packaging).map({
        "ventilated_crate": 0.0, "corrugated_box": 0.35, "plastic_crate": 0.15
    }).to_numpy()

    # Synthetic target equation: higher maturity, heat, bruising and delays reduce shelf life.
    shelf = (
        27.0 - 13.0 * maturity
        + 0.85 * (brix - 12)
        - 0.19 * (pulp_temp - 20)
        - 0.075 * transport
        - 0.42 * bruising
        - 0.52 * cooling
        - 0.42 * heat
        - 0.50 * damage
        - pack_penalty
        + RNG.normal(0, 1.8, n)
    )
    shelf = np.clip(shelf, 1.0, 30.0)
    df = pd.DataFrame({
        "shipment_id": [f"V{version}-{i+1:04d}" for i in range(n)],
        "harvest_maturity_index": maturity.round(3),
        "pulp_temperature_c": pulp_temp.round(2),
        "brix": brix.round(2),
        "bruising_percentage": bruising.round(2),
        "transport_duration_hours": transport.round(2),
        "cooling_delay_hours": cooling.round(2),
        "packaging_type": packaging,
        "remaining_shelf_life_days": shelf.round(2)
    })
    return df

if __name__ == "__main__":
    v1 = make_shipments(300, 1)
    v2 = make_shipments(300, 2)
    v1.to_csv(DATA_DIR / "mango_shipments_v1.csv", index=False)
    v2.to_csv(DATA_DIR / "mango_shipments_v2.csv", index=False)
    print(f"Created V1: {len(v1)} rows; V2: {len(v2)} rows")
    print("V2 transport/cooling means:",
          round(v2.transport_duration_hours.mean(), 2),
          round(v2.cooling_delay_hours.mean(), 2))
