"""Three-fold expanding-window backtest and paired significance evaluation data.

The final 42-calendar-day holdout remains untouched. Rolling folds are fully
inside the pre-holdout period; hyperparameter selection for each fold uses only
that fold's training history.
"""
from pathlib import Path
import json
import re
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

from modeling_common import get_model_features, get_sample_stores, recursive_forecast, rmse
from importlib.util import spec_from_file_location, module_from_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQ_LENGTH = 30
CV_HORIZON_DAYS = 14
N_FOLDS = 3
ARIMA_ORDERS = [(1, 1, 0), (1, 1, 1), (3, 1, 0), (5, 1, 0)]
PROPHET_CONFIGS = [
    {"changepoint_prior_scale": 0.05, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.10, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.05, "seasonality_mode": "multiplicative"},
]

# Reuse the exact ML tuning grid and LSTM trainer defined for the main pipeline.
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import importlib
ml_module = importlib.import_module("04_ml_modeling")
dl_module = importlib.import_module("05_dl_modeling")


def _rmse(actual, predicted):
    return rmse(actual, predicted)


def _prophet_fit(train, config):
    model = Prophet(
        daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
        uncertainty_samples=0, **config,
    )
    model.fit(train[["Date", "Sales"]].rename(columns={"Date": "ds", "Sales": "y"}))
    return model


def _tune_arima(train):
    if len(train) < 70:
        return ARIMA_ORDERS[0]
    fit_part, validation = train.iloc[:-21], train.iloc[-21:]
    best, best_score = ARIMA_ORDERS[0], float("inf")
    for order in ARIMA_ORDERS:
        try:
            fitted = ARIMA(fit_part["Sales"].astype(float).to_numpy(), order=order).fit()
            prediction = np.asarray(fitted.forecast(steps=len(validation)), dtype=float)
            score = _rmse(validation["Sales"], prediction)
            if score < best_score:
                best, best_score = order, score
        except Exception:
            continue
    return best


def _tune_prophet(train):
    if len(train) < 70:
        return PROPHET_CONFIGS[0]
    fit_part, validation = train.iloc[:-21], train.iloc[-21:]
    best, best_score = PROPHET_CONFIGS[0], float("inf")
    for config in PROPHET_CONFIGS:
        try:
            model = _prophet_fit(fit_part, config)
            pred = model.predict(pd.DataFrame({"ds": validation["Date"].to_numpy()}))["yhat"].to_numpy()
            score = _rmse(validation["Sales"], pred)
            if score < best_score:
                best, best_score = config, score
        except Exception:
            continue
    return best


def _fit_tree(model_name, config):
    if model_name == "XGBOOST":
        return xgb.XGBRegressor(
            objective="reg:squarederror", tree_method="hist", n_jobs=-1, random_state=42, **config
        )
    return lgb.LGBMRegressor(
        objective="regression", n_jobs=-1, random_state=42, verbosity=-1, **config
    )


def _metric_rows(rows, fold, cutoff, store_id, dates, actual, predicted):
    if len(dates) != len(actual) or len(actual) != len(predicted):
        return
    for date, y, pred in zip(dates, actual, predicted):
        if np.isfinite(y) and np.isfinite(pred):
            rows.append({
                "Fold": fold, "Cutoff": cutoff.strftime("%Y-%m-%d"), "Store": int(store_id),
                "Model": None, "Date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "Actual": float(y), "Prediction": float(pred),
            })


def main():
    warnings.filterwarnings("ignore")
    feature_path = PROJECT_ROOT / "data" / "features" / "train_features.csv"
    frame = pd.read_csv(feature_path, parse_dates=["Date"], low_memory=False)
    # Match the feature-name normalization used by src/04_ml_modeling.py.
    # LightGBM rejects JSON-special characters (e.g. commas/brackets) in names
    # created by one-hot encoding values such as PromoInterval.
    frame = frame.rename(columns=lambda name: re.sub(r"[^A-Za-z0-9_]+", "_", str(name)))
    frame = frame.sort_values(["Store", "Date"])
    if "Open" in frame.columns:
        frame = frame[frame["Open"].eq(1)].copy()
    final_split = frame["Date"].max() - pd.Timedelta(days=42)
    stores = get_sample_stores(PROJECT_ROOT, pd.read_csv(PROJECT_ROOT / "data/raw/store.csv") if (PROJECT_ROOT / "data/raw/store.csv").exists() else None)
    stores = [store for store in stores if store in set(frame["Store"].astype(int))]
    if not stores:
        raise RuntimeError("Rolling-origin validation requires the classical sample-store list.")
    features = get_model_features(frame)
    records, runtime_rows = [], []

    # Historical folds are strictly before the hyperparameter-tuning window used
    # for the final fixed holdout: the last validation fold ends 28 days before it.
    cutoffs = [final_split - pd.Timedelta(days=70 - 14 * i) for i in range(N_FOLDS)]
    for fold_number, cutoff in enumerate(cutoffs, start=1):
        fold_end = cutoff + pd.Timedelta(days=CV_HORIZON_DAYS)
        fold_train = frame[frame["Date"] <= cutoff].copy()
        fold_validation = frame[(frame["Date"] > cutoff) & (frame["Date"] <= fold_end)].copy()
        if fold_train.empty or fold_validation.empty:
            print(f"Fold {fold_number} has insufficient data; skipping.")
            continue
        print(f"Rolling fold {fold_number}/{N_FOLDS}: train <= {cutoff.date()}, validate <= {fold_end.date()}")

        # Store-level ARIMA and Prophet baselines (tuned using that fold's own past).
        for store_id in stores:
            store_train = fold_train[fold_train["Store"].astype(int).eq(int(store_id))].sort_values("Date")
            store_val = fold_validation[fold_validation["Store"].astype(int).eq(int(store_id))].sort_values("Date")
            if len(store_train) < 60 or store_val.empty:
                continue
            started = time.perf_counter()
            arima_order = _tune_arima(store_train)
            try:
                fitted = ARIMA(store_train["Sales"].astype(float).to_numpy(), order=arima_order).fit()
                predictions = np.maximum(0.0, np.asarray(fitted.forecast(steps=len(store_val)), dtype=float))
                _metric_rows(records, fold_number, cutoff, store_id, store_val["Date"], store_val["Sales"], predictions)
                for record in records[-len(store_val):]:
                    record["Model"] = "ARIMA"
            except Exception as exc:
                print(f"  Fold {fold_number} Store {store_id} ARIMA skipped: {exc}")
            runtime_rows.append({"Fold": fold_number, "Model": "ARIMA", "Store": int(store_id), "Seconds": time.perf_counter() - started})

            started = time.perf_counter()
            config = _tune_prophet(store_train)
            try:
                fitted = _prophet_fit(store_train, config)
                forecast = fitted.predict(pd.DataFrame({"ds": store_val["Date"].to_numpy()}))
                predictions = np.maximum(0.0, forecast["yhat"].to_numpy(dtype=float))
                _metric_rows(records, fold_number, cutoff, store_id, store_val["Date"], store_val["Sales"], predictions)
                for record in records[-len(store_val):]:
                    record["Model"] = "PROPHET"
            except Exception as exc:
                print(f"  Fold {fold_number} Store {store_id} Prophet skipped: {exc}")
            runtime_rows.append({"Fold": fold_number, "Model": "PROPHET", "Store": int(store_id), "Seconds": time.perf_counter() - started})

        # Pooled ML models: tune each fold only on a trailing block within the
        # fold's own training data, refit on that full training window, then recurse.
        inner_cutoff = cutoff - pd.Timedelta(days=21)
        for model_name, grid in (("XGBOOST", ml_module.XGB_GRID), ("LIGHTGBM", ml_module.LGB_GRID)):
            started = time.perf_counter()
            best_config, best_score = grid[0], float("inf")
            tune_train = fold_train[fold_train["Date"] <= inner_cutoff]
            if tune_train.empty:
                tune_train = fold_train
            for config in grid:
                try:
                    candidate = _fit_tree(model_name, config)
                    candidate.fit(tune_train[features], tune_train["Sales"].astype(float))
                    actuals, predictions = [], []
                    for store_id in stores:
                        sframe = frame[(frame["Store"].astype(int).eq(int(store_id))) & (frame["Date"] <= cutoff)]
                        val, pred = recursive_forecast(candidate, sframe, inner_cutoff, features, end_date=cutoff)
                        if len(val) == len(pred) and len(val):
                            actuals.extend(val["Sales"].astype(float).tolist())
                            predictions.extend(pred)
                    score = _rmse(actuals, predictions)
                    if score < best_score:
                        best_config, best_score = config, score
                except Exception as exc:
                    print(f"  Fold {fold_number} {model_name} tuning candidate skipped: {exc}")
            fit_model = _fit_tree(model_name, best_config)
            fit_model.fit(fold_train[features], fold_train["Sales"].astype(float))
            for store_id in stores:
                sframe = frame[frame["Store"].astype(int).eq(int(store_id))]
                val, predictions = recursive_forecast(fit_model, sframe, cutoff, features, end_date=fold_end)
                _metric_rows(records, fold_number, cutoff, store_id, val["Date"], val["Sales"], predictions)
                for record in records[-len(val):] if len(val) else []:
                    record["Model"] = model_name
            runtime_rows.append({"Fold": fold_number, "Model": model_name, "Store": "ALL", "Seconds": time.perf_counter() - started})
            print(f"  {model_name} fold {fold_number} selected {best_config}")

        # LSTM on the same representative stores and dates. Early stopping and
        # scaling are recomputed within each fold; no final-test labels are used.
        for store_id in stores:
            store_frame = frame[frame["Store"].astype(int).eq(int(store_id))].copy()
            started = time.perf_counter()
            result = dl_module.train_store(
                store_frame, features, cutoff, forecast_end=fold_end,
                seq_length=dl_module.SEQ_LENGTH, max_epochs=30, patience=4,
                artifact_dir=None,
            )
            if result is None:
                continue
            _metric_rows(records, fold_number, cutoff, store_id, result["dates"], result["actual"], result["lstm"])
            for record in records[-len(result["dates"]):]:
                record["Model"] = "LSTM"
            runtime_rows.append({"Fold": fold_number, "Model": "LSTM", "Store": int(store_id), "Seconds": time.perf_counter() - started})

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(records)
    if result_frame.empty:
        raise RuntimeError("No rolling-origin forecasts were produced.")
    result_frame = result_frame.dropna(subset=["Model", "Prediction", "Actual"])
    result_frame.to_csv(output_dir / "rolling_origin_results.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(output_dir / "rolling_origin_runtime.csv", index=False)
    metadata = {
        "model_version": 2,
        "final_holdout_cutoff": final_split.strftime("%Y-%m-%d"),
        "folds": [
            {"fold": i + 1, "cutoff": cutoff.strftime("%Y-%m-%d"),
             "validation_end": (cutoff + pd.Timedelta(days=CV_HORIZON_DAYS)).strftime("%Y-%m-%d")}
            for i, cutoff in enumerate(cutoffs)
        ],
        "stores": stores,
        "models": sorted(result_frame["Model"].unique().tolist()),
    }
    (output_dir / "rolling_origin_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Rolling-origin predictions saved: {len(result_frame):,} rows across {result_frame['Fold'].nunique()} folds.")


if __name__ == "__main__":
    main()
