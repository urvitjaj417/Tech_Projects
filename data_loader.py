# ml/data_loader.py
# Loads the sensor CSV produced by sensor_sim.c and engineers features for ML.

import pandas as pd
import numpy as np
from pathlib import Path

FEATURE_COLS = [
    "temperature_c", "vibration_mms", "pressure_bar",
    "current_a", "cycle_time_s", "output_rate_uph",
    # engineered
    "temp_vib_ratio", "power_approx", "efficiency_index",
    "temp_zscore", "vib_zscore",
]

TARGET_COL = "fault_flag"


def load_data(csv_path: str = "data/sensor_data.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found at '{csv_path}'.\n"
            "Fix: cd c_module && make run"
        )
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[data_loader] {len(df)} records | "
          f"faults: {df['fault_flag'].sum()} ({df['fault_flag'].mean()*100:.1f}%)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["temp_vib_ratio"]   = df["temperature_c"] / (df["vibration_mms"] + 1e-6)
    df["power_approx"]     = df["current_a"] * df["pressure_bar"]
    df["efficiency_index"] = df["output_rate_uph"] / (df["cycle_time_s"] + 1e-6)

    df["temp_zscore"] = 0.0
    df["vib_zscore"]  = 0.0
    for mid in df["machine_id"].unique():
        mask = df["machine_id"] == mid
        for col, out in [("temperature_c", "temp_zscore"),
                         ("vibration_mms",  "vib_zscore")]:
            rm = df.loc[mask, col].rolling(50, min_periods=5).mean()
            rs = df.loc[mask, col].rolling(50, min_periods=5).std().fillna(1.0)
            df.loc[mask, out] = ((df.loc[mask, col] - rm) / rs).fillna(0)

    return df


def get_xy(df: pd.DataFrame):
    df = engineer_features(df)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)
    return X, y, df


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("machine_name").agg(
        total_records  = ("fault_flag", "count"),
        fault_count    = ("fault_flag", "sum"),
        fault_rate_pct = ("fault_flag", lambda x: round(x.mean() * 100, 2)),
        avg_temp       = ("temperature_c",   "mean"),
        avg_vibration  = ("vibration_mms",   "mean"),
        avg_output     = ("output_rate_uph", "mean"),
    ).reset_index().round(2)


if __name__ == "__main__":
    df = load_data()
    X, y, _ = get_xy(df)
    print(f"Feature matrix: {X.shape}")
    print(summary_stats(df).to_string())
