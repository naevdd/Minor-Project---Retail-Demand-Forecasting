"""Tuned per-store ARIMA and Prophet baselines with 95% forecast intervals."""
from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

from modeling_common import calculate_metrics, get_sample_stores, rmse

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_DAYS = 42
ARIMA_ORDERS = [(1, 1, 0), (1, 1, 1), (3, 1, 0), (5, 1, 0)]
PROPHET_CONFIGS = [
    {"changepoint_prior_scale": 0.05, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.10, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.05, "seasonality_mode": "multiplicative"},
]


def _artifact_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _fit_prophet(train: pd.DataFrame, config: dict) -> Prophet:
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        uncertainty_samples=1000,
        **config,
    )
    model.fit(train[["Date", "Sales"]].rename(columns={"Date": "ds", "Sales": "y"}))
    return model


def tune_arima(train: pd.DataFrame, store_id: int):
    rows = []
    if len(train) < 70:
        return (1, 1, 0), rows
    fit_part, tune_part = train.iloc[:-28], train.iloc[-28:]
    best_order, best_score = ARIMA_ORDERS[0], float("inf")
    for order in ARIMA_ORDERS:
        started = time.perf_counter()
        try:
            fitted = ARIMA(fit_part["Sales"].astype(float).to_numpy(), order=order).fit()
            prediction = np.asarray(fitted.forecast(steps=len(tune_part)), dtype=float)
            score = rmse(tune_part["Sales"], prediction)
        except Exception as exc:
            score = float("inf")
            error = str(exc)
        else:
            error = ""
        rows.append({
            "Store": int(store_id), "Model": "ARIMA", "Parameters": json.dumps({"order": order}),
            "TuningRMSE": score, "TuningSeconds": time.perf_counter() - started, "Error": error,
        })
        if score < best_score:
            best_order, best_score = order, score
    return best_order, rows


def tune_prophet(train: pd.DataFrame, store_id: int):
    rows = []
    if len(train) < 70:
        return PROPHET_CONFIGS[0], rows
    fit_part, tune_part = train.iloc[:-28], train.iloc[-28:]
    best_config, best_score = PROPHET_CONFIGS[0], float("inf")
    for config in PROPHET_CONFIGS:
        started = time.perf_counter()
        try:
            model = _fit_prophet(fit_part, config)
            forecast = model.predict(pd.DataFrame({"ds": tune_part["Date"].to_numpy()}))
            score = rmse(tune_part["Sales"], forecast["yhat"].to_numpy(dtype=float))
            error = ""
        except Exception as exc:
            score = float("inf")
            error = str(exc)
        rows.append({
            "Store": int(store_id), "Model": "PROPHET", "Parameters": json.dumps(config, sort_keys=True),
            "TuningRMSE": score, "TuningSeconds": time.perf_counter() - started, "Error": error,
        })
        if score < best_score:
            best_config, best_score = config, score
    return best_config, rows


def train_eval_store(df_store: pd.DataFrame, split_date, store_id: int, model_dir: Path):
    frame = df_store.sort_values("Date").copy()
    if "Open" in frame.columns:
        frame = frame[frame["Open"].eq(1)]
    train = frame[frame["Date"] <= split_date].copy()
    val = frame[frame["Date"] > split_date].copy()
    if len(train) < 60 or val.empty:
        return None, [], []

    results = {
        "dates": val["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "actual": val["Sales"].astype(float).tolist(),
    }
    tuning_rows, runtime_rows = [], []
    store_dir = model_dir / f"store_{int(store_id):04d}"
    store_dir.mkdir(parents=True, exist_ok=True)

    # ARIMA: tune on the final 28 open training observations, then refit on all
    # pre-holdout observations and produce native prediction intervals.
    tuning_start = time.perf_counter()
    best_order, rows = tune_arima(train, store_id)
    tuning_rows.extend(rows)
    tuning_seconds = time.perf_counter() - tuning_start
    fit_start = time.perf_counter()
    try:
        arima_fit = ARIMA(train["Sales"].astype(float).to_numpy(), order=best_order).fit()
        fit_seconds = time.perf_counter() - fit_start
        forecast_start = time.perf_counter()
        arima_forecast = arima_fit.get_forecast(steps=len(val))
        arima_pred = np.nan_to_num(np.asarray(arima_forecast.predicted_mean, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        arima_ci = np.nan_to_num(np.asarray(arima_forecast.conf_int(alpha=0.05), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        forecast_seconds = time.perf_counter() - forecast_start
        lower = np.maximum(0, arima_ci[:, 0])
        upper = np.maximum(lower, arima_ci[:, 1])
        results.update({
            "arima": np.maximum(0, arima_pred).tolist(),
            "arima_lower": lower.tolist(), "arima_upper": upper.tolist(),
            "arima_order": list(best_order),
        })
        artifact = store_dir / "arima_results.pkl"
        arima_fit.save(str(artifact))
        runtime_rows.append({
            "Model": "ARIMA", "Store": int(store_id), "FitSeconds": fit_seconds,
            "ForecastSeconds": forecast_seconds, "TuningSeconds": tuning_seconds,
            "Artifact": _artifact_display_path(artifact),
        })
    except Exception as exc:
        print(f"  Store {store_id}: ARIMA failed ({exc})")
        results.update({"arima": [0.0] * len(val), "arima_lower": [0.0] * len(val),
                        "arima_upper": [0.0] * len(val), "arima_order": list(best_order)})

    # Prophet: tune trend flexibility and seasonal mode, then save native 95% intervals.
    tuning_start = time.perf_counter()
    best_config, rows = tune_prophet(train, store_id)
    tuning_rows.extend(rows)
    tuning_seconds = time.perf_counter() - tuning_start
    fit_start = time.perf_counter()
    try:
        prophet_model = _fit_prophet(train, best_config)
        fit_seconds = time.perf_counter() - fit_start
        forecast_start = time.perf_counter()
        forecast = prophet_model.predict(pd.DataFrame({"ds": val["Date"].to_numpy()}))
        prophet_pred = np.nan_to_num(forecast["yhat"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        lower = np.maximum(0, np.nan_to_num(forecast["yhat_lower"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
        upper = np.maximum(lower, np.nan_to_num(forecast["yhat_upper"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
        forecast_seconds = time.perf_counter() - forecast_start
        results.update({
            "prophet": np.maximum(0, prophet_pred).tolist(),
            "prophet_lower": lower.tolist(), "prophet_upper": upper.tolist(),
            "prophet_config": best_config,
        })
        artifact = store_dir / "prophet_model.joblib"
        joblib.dump(prophet_model, artifact)
        runtime_rows.append({
            "Model": "PROPHET", "Store": int(store_id), "FitSeconds": fit_seconds,
            "ForecastSeconds": forecast_seconds, "TuningSeconds": tuning_seconds,
            "Artifact": _artifact_display_path(artifact),
        })
    except Exception as exc:
        print(f"  Store {store_id}: Prophet failed ({exc})")
        results.update({"prophet": [0.0] * len(val), "prophet_lower": [0.0] * len(val),
                        "prophet_upper": [0.0] * len(val), "prophet_config": best_config})
    return results, tuning_rows, runtime_rows


def main():
    features_path = PROJECT_ROOT / "data" / "features" / "train_features.csv"
    frame = pd.read_csv(features_path, parse_dates=["Date"], low_memory=False)
    global_max_date = frame["Date"].max()
    split_date = global_max_date - pd.Timedelta(days=VAL_DAYS)
    raw_store_path = PROJECT_ROOT / "data" / "raw" / "store.csv"
    store_meta = pd.read_csv(raw_store_path) if raw_store_path.exists() else None
    stores = get_sample_stores(PROJECT_ROOT, store_meta)
    if not stores:
        raise RuntimeError("No classical sample stores found. Run feature engineering and provide data/raw/store.csv.")

    model_dir = PROJECT_ROOT / "models" / "classical"
    model_dir.mkdir(parents=True, exist_ok=True)
    all_results, tuning_rows, runtime_rows = {}, [], []
    started_total = time.perf_counter()
    print(f"Classical benchmark: {len(stores)} stratified stores; cutoff={split_date.date()}")
    for index, store_id in enumerate(stores, start=1):
        print(f"[{index}/{len(stores)}] Fitting Store {store_id}")
        store_frame = frame[frame["Store"].astype(int).eq(int(store_id))]
        result, tune, runtime = train_eval_store(store_frame, split_date, int(store_id), model_dir)
        if result is not None:
            all_results[str(int(store_id))] = result
        tuning_rows.extend(tune)
        runtime_rows.extend(runtime)

    feature_dir = PROJECT_ROOT / "data" / "features"
    (feature_dir / "classical_results.json").write_text(json.dumps(all_results, allow_nan=False), encoding="utf-8")
    (feature_dir / "classical_sample_stores.json").write_text(json.dumps(sorted(stores), indent=2), encoding="utf-8")
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tuning_rows).to_csv(processed_dir / "classical_tuning_results.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(processed_dir / "classical_runtime.csv", index=False)
    metadata = {
        "model_version": 2,
        "global_max_date": global_max_date.strftime("%Y-%m-%d"),
        "split_date": split_date.strftime("%Y-%m-%d"),
        "sample_stores": sorted(stores),
        "completed_stores": sorted(int(store) for store in all_results),
        "models": ["ARIMA", "PROPHET"],
    }
    (feature_dir / "classical_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Classical modeling complete in {time.perf_counter() - started_total:.1f}s. Saved {len(all_results)} stores and model artifacts under models/classical.")


if __name__ == "__main__":
    main()
