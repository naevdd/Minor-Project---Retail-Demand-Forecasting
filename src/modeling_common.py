"""Shared utilities for leakage-safe retail forecasting experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LAG_COLUMNS = {f"Sales_Lag_{lag}": lag for lag in (1, 7, 14, 30)}
ROLLING_COLUMNS = {
    f"Sales_Roll{stat}_{window}": (window, stat.lower())
    for window in (7, 14)
    for stat in ("Mean", "Std", "Min", "Max")
}
SALES_DERIVED_COLUMNS = set(LAG_COLUMNS) | set(ROLLING_COLUMNS)
DROP_FEATURE_COLUMNS = {"Date", "Sales", "Customers", "Open"}


def rmse(actual, predicted) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    valid = np.isfinite(a) & np.isfinite(p)
    if not valid.any():
        return float("inf")
    return float(np.sqrt(np.mean((a[valid] - p[valid]) ** 2)))


def calculate_metrics(actual, predicted) -> tuple[float, float, float]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    valid = np.isfinite(a) & np.isfinite(p)
    a, p = a[valid], p[valid]
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    errors = np.abs(a - p)
    rmse_value = float(np.sqrt(np.mean((a - p) ** 2)))
    mae_value = float(np.mean(errors))
    positive = a > 0
    mape_value = float(np.mean(errors[positive] / a[positive]) * 100) if positive.any() else np.nan
    return rmse_value, mae_value, mape_value


def get_model_features(frame: pd.DataFrame) -> list[str]:
    """Select model inputs while excluding target, date, closed flag and leakage-prone customers."""
    return [column for column in frame.columns if column not in DROP_FEATURE_COLUMNS]


def update_sales_features(row: pd.Series, history: Iterable[float]) -> pd.Series:
    """Regenerate the sales-derived features from the same open-day history used at inference."""
    row = row.copy()
    values = np.asarray(list(history), dtype=float)
    for column, lag in LAG_COLUMNS.items():
        if column in row.index:
            row[column] = float(values[-lag]) if len(values) >= lag else 0.0
    for column, (window, statistic) in ROLLING_COLUMNS.items():
        if column not in row.index:
            continue
        recent = values[-window:]
        if len(recent) == 0:
            value = 0.0
        elif statistic == "mean":
            value = float(np.mean(recent))
        elif statistic == "std":
            value = float(np.std(recent, ddof=1)) if len(recent) >= 2 else 0.0
        elif statistic == "min":
            value = float(np.min(recent))
        elif statistic == "max":
            value = float(np.max(recent))
        else:
            raise ValueError(f"Unsupported rolling statistic: {statistic}")
        row[column] = value if np.isfinite(value) else 0.0
    return row


def recursive_forecast(model, store_frame: pd.DataFrame, split_date, features: list[str], end_date=None):
    """Blind, recursive forecast on open days; predicted values replace future actuals in history."""
    frame = store_frame.sort_values("Date").copy()
    frame = frame[frame["Open"].eq(1)] if "Open" in frame.columns else frame
    if end_date is not None:
        frame = frame[frame["Date"] <= pd.Timestamp(end_date)]
    train_store = frame[frame["Date"] <= pd.Timestamp(split_date)]
    val_store = frame[frame["Date"] > pd.Timestamp(split_date)].copy()
    history = train_store["Sales"].astype(float).tolist()
    predictions = []
    for _, row in val_store.iterrows():
        feature_row = update_sales_features(row[features], history)
        X = pd.DataFrame([feature_row], columns=features)
        prediction = float(np.asarray(model.predict(X)).reshape(-1)[0])
        prediction = max(0.0, prediction) if np.isfinite(prediction) else 0.0
        predictions.append(prediction)
        history.append(prediction)
    return val_store, predictions


def get_sample_stores(project_root: Path, store_metadata: pd.DataFrame | None = None, n_per_type: int = 5) -> list[int]:
    """Reuse the project's persisted classical sample so comparisons stay reproducible."""
    sample_path = project_root / "data" / "features" / "classical_sample_stores.json"
    if sample_path.exists():
        try:
            stores = json.loads(sample_path.read_text(encoding="utf-8"))
            if stores:
                return sorted({int(store) for store in stores})
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    if store_metadata is not None and {"Store", "StoreType"}.issubset(store_metadata.columns):
        stores = (
            store_metadata.groupby("StoreType")["Store"]
            .apply(lambda values: values.sample(min(n_per_type, len(values)), random_state=42))
            .reset_index(drop=True)
            .astype(int)
            .tolist()
        )
        return sorted(stores)
    return []


def atomic_json_dump(data, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary_path.replace(path)
