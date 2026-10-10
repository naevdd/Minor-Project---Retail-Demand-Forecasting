"""Create open-day lag, rolling-window, calendar and store features."""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # Every forecasting model predicts demand only on days when a store is open.
    # Filtering first makes training lags consistent with recursive inference,
    # which appends one observed/predicted open-day sale at a time.
    if "Open" in df.columns:
        df = df[df["Open"].eq(1)].copy()
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    for lag in (1, 7, 14, 30):
        df[f"Sales_Lag_{lag}"] = df.groupby("Store", sort=False)["Sales"].shift(lag)

    for window in (7, 14):
        prior_sales = df.groupby("Store", sort=False)["Sales"].shift(1)
        grouped = prior_sales.groupby(df["Store"], sort=False)
        rolling = grouped.rolling(window=window, min_periods=1)
        df[f"Sales_RollMean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        df[f"Sales_RollStd_{window}"] = rolling.std().reset_index(level=0, drop=True)
        df[f"Sales_RollMin_{window}"] = rolling.min().reset_index(level=0, drop=True)
        df[f"Sales_RollMax_{window}"] = rolling.max().reset_index(level=0, drop=True)

    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].astype(str)
    cat_cols = [c for c in ("StoreType", "Assortment", "StateHoliday", "PromoInterval") if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # Remaining NaNs occur at the start of each store's history or for rolling
    # standard deviation with one prior observation; zero is the neutral fill.
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


if __name__ == "__main__":
    raw_path = PROJECT_ROOT / "data" / "processed" / "train_clean.csv"
    output_path = PROJECT_ROOT / "data" / "features" / "train_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading cleaned dataset: {raw_path}")
    frame = pd.read_csv(raw_path, parse_dates=["Date"], low_memory=False)
    features = create_features(frame)
    features.to_csv(output_path, index=False)
    print(f"Open-day feature dataset saved: {output_path} ({len(features):,} rows)")
