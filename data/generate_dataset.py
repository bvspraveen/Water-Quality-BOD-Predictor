"""
Generate synthetic but scientifically realistic river water quality dataset.

Features chosen based on standard river monitoring parameters used in
BOD estimation literature (consistent with BVS Praveen et al., 2025).

BOD (Biochemical Oxygen Demand) is the target variable — it measures
how much oxygen microbes consume to decompose organic matter, a key
water pollution indicator.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 800  # number of samples


def generate_dataset():
    # ── Input features (observable water quality parameters) ──────────────────
    pH             = np.random.normal(7.4, 0.6, N).clip(6.0, 9.5)
    temperature    = np.random.normal(26.0, 4.5, N).clip(10.0, 40.0)   # °C
    dissolved_O2   = np.random.normal(7.0, 1.8, N).clip(1.0, 14.0)     # mg/L
    turbidity      = np.random.exponential(scale=12, size=N).clip(0.5, 80.0)  # NTU
    conductivity   = np.random.normal(450, 120, N).clip(100.0, 900.0)   # µS/cm
    total_solids   = np.random.normal(320, 90, N).clip(80.0, 700.0)     # mg/L
    nitrates       = np.random.exponential(scale=4, size=N).clip(0.1, 25.0)   # mg/L
    phosphates     = np.random.exponential(scale=1.5, size=N).clip(0.01, 10.0)# mg/L
    ammonia        = np.random.exponential(scale=2.0, size=N).clip(0.05, 15.0)# mg/L
    cod            = np.random.normal(90, 30, N).clip(10.0, 300.0)      # mg/L (Chemical O.D.)

    # ── BOD formula (domain-grounded synthetic relationship) ──────────────────
    # Higher COD, ammonia, phosphates → higher BOD
    # Higher dissolved O2, higher pH → lower BOD
    # Noise term reflects real-world measurement variance
    bod = (
        0.45 * cod
        + 3.2 * ammonia
        + 2.8 * phosphates
        + 0.12 * turbidity
        + 0.04 * total_solids
        + 0.6  * nitrates
        - 2.5 * dissolved_O2
        - 1.8 * (pH - 7.0)
        + 0.2 * temperature
        + np.random.normal(0, 4, N)      # measurement noise
    ).clip(1.0, 250.0)

    df = pd.DataFrame({
        "pH":            pH,
        "Temperature_C": temperature,
        "Dissolved_O2":  dissolved_O2,
        "Turbidity_NTU": turbidity,
        "Conductivity":  conductivity,
        "Total_Solids":  total_solids,
        "Nitrates":      nitrates,
        "Phosphates":    phosphates,
        "Ammonia":       ammonia,
        "COD":           cod,
        "BOD":           bod          # ← target
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("river_water_quality.csv", index=False)
    print(f"Dataset saved: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.describe().round(2))
