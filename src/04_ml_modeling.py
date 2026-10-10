"""Tune pooled XGBoost/LightGBM models and evaluate recursive open-day forecasts."""
from pathlib import Path
import json
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from modeling_common import get_model_features, get_sample_stores, recursive_forecast, rmse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_DAYS = 42
XGB_GRID = [
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.10, "subsample": 0.9, "colsample_bytree": 0.9},
    {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
    {"n_estimators": 100, "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
]
LGB_GRID = [
    {"n_estimators": 100, "max_depth": 6, "num_leaves": 31, "learning_rate": 0.10, "subsample": 0.9, "colsample_bytree": 0.9},
    {"n_estimators": 150, "max_depth": 8, "num_leaves": 31, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
    {"n_estimators": 100, "max_depth": -1, "num_leaves": 63, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
]


def make_xgb(config: dict):
    return xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist", n_jobs=-1,
        random_state=42, **config,
    )


def make_lgb(config: dict):
    return lgb.LGBMRegressor(
        objective="regression", n_jobs=-1, random_state=42, verbosity=-1,
        **config,
    )


def save_feature_importance(model, feature_names, path: Path, model_name: str):
    if model_name == "XGBOOST":
        importance = np.asarray(model.feature_importances_, dtype=float)
    else:
        importance = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    frame = pd.DataFrame({"Feature": list(feature_names), "Importance": importance})
    total = frame["Importance"].sum()
    frame["ImportancePct"] = frame["Importance"] / total * 100.0 if total > 0 else 0.0
    frame["Model"] = model_name
    frame.sort_values("Importance", ascending=False).to_csv(path, index=False)


def tune_model(model_name, grid, train_df, all_open_df, features, inner_cutoff, final_cutoff, sample_stores):
    factory = make_xgb if model_name == "XGBOOST" else make_lgb
    tuning_rows = []
    fit_candidates = train_df[train_df["Date"] <= inner_cutoff]
    if fit_candidates.empty:
        fit_candidates = train_df
    X_fit = fit_candidates[features]
    y_fit = fit_candidates["Sales"].astype(float)
    best_config = grid[0]
    best_score = float("inf")

    for index, config in enumerate(grid, start=1):
        started = time.perf_counter()
        model = factory(config)
        model.fit(X_fit, y_fit)
        actuals, predictions = [], []
        for store_id in sample_stores:
            store_frame = all_open_df[all_open_df["Store"].astype(int).eq(int(store_id))]
            val, preds = recursive_forecast(
                model, store_frame, inner_cutoff, features, end_date=final_cutoff
            )
            if len(val) and len(val) == len(preds):
                actuals.extend(val["Sales"].astype(float).tolist())
                predictions.extend(preds)
        score = rmse(actuals, predictions)
        tuning_rows.append({
            "Store": "ALL", "Model": model_name, "Candidate": index,
            "Parameters": json.dumps(config, sort_keys=True), "TuningRMSE": score,
            "TuningSeconds": time.perf_counter() - started, "Error": "",
        })
        if score < best_score:
            best_score, best_config = score, config
        print(f"  {model_name} tuning candidate {index}/{len(grid)}: RMSE={score:.2f}")
    return best_config, tuning_rows


def train_ml_models(frame: pd.DataFrame, val_days: int = VAL_DAYS):
    frame = frame.rename(columns=lambda name: re.sub(r"[^A-Za-z0-9_]+", "_", name))
    frame = frame.sort_values(["Store", "Date"]).copy()
    if "Open" in frame.columns:
        frame = frame[frame["Open"].eq(1)].copy()

    global_max_date = frame["Date"].max()
    split_date = global_max_date - pd.Timedelta(days=val_days)
    inner_cutoff = split_date - pd.Timedelta(days=21)
    final_train = frame[frame["Date"] <= split_date].copy()
    features = get_model_features(final_train)

    store_meta_path = PROJECT_ROOT / "data" / "raw" / "store.csv"
    store_meta = pd.read_csv(store_meta_path) if store_meta_path.exists() else None
    sample_stores = get_sample_stores(PROJECT_ROOT, store_meta)
    if not sample_stores:
        sample_stores = sorted(frame["Store"].astype(int).unique())[:20]

    print(f"Pooled ML training rows: {len(final_train):,}; common final cutoff={split_date.date()}")
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir = PROJECT_ROOT / "models" / "ml"
    model_dir.mkdir(parents=True, exist_ok=True)

    tuning_rows, runtime_rows, models, configurations = [], [], {}, {}
    for model_name, grid in (("XGBOOST", XGB_GRID), ("LIGHTGBM", LGB_GRID)):
        tuning_start = time.perf_counter()
        best_config, rows = tune_model(
            model_name, grid, final_train, frame, features, inner_cutoff, split_date, sample_stores
        )
        tuning_seconds = time.perf_counter() - tuning_start
        tuning_rows.extend(rows)
        model = (make_xgb if model_name == "XGBOOST" else make_lgb)(best_config)
        fit_start = time.perf_counter()
        model.fit(final_train[features], final_train["Sales"].astype(float))
        fit_seconds = time.perf_counter() - fit_start
        models[model_name] = model
        configurations[model_name] = best_config
        if model_name == "XGBOOST":
            model.save_model(str(model_dir / "xgboost_model.json"))
            save_feature_importance(model, features, processed_dir / "xgboost_feature_importance.csv", model_name)
        else:
            model.booster_.save_model(str(model_dir / "lightgbm_model.txt"))
            save_feature_importance(model, features, processed_dir / "lightgbm_feature_importance.csv", model_name)
        runtime_rows.append({
            "Model": model_name, "Scope": "Pooled across all stores", "FitSeconds": fit_seconds,
            "ForecastSeconds": 0.0, "TuningSeconds": tuning_seconds,
            "Artifact": str((model_dir / ("xgboost_model.json" if model_name == "XGBOOST" else "lightgbm_model.txt")).relative_to(PROJECT_ROOT)),
        })
        print(f"{model_name} final fit complete in {fit_seconds:.1f}s; selected {best_config}")

    # Persist feature order and configuration so saved native artifacts can be reused.
    (model_dir / "feature_columns.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (model_dir / "model_config.json").write_text(json.dumps({
        "version": 2, "split_date": split_date.strftime("%Y-%m-%d"),
        "inner_tuning_cutoff": inner_cutoff.strftime("%Y-%m-%d"),
        "features": features, "configurations": configurations,
    }, indent=2), encoding="utf-8")

    store_results = {}
    forecast_seconds = {"XGBOOST": 0.0, "LIGHTGBM": 0.0}
    for store_id, store_frame in frame.groupby("Store", sort=True):
        row_started = time.perf_counter()
        val, xgb_preds = recursive_forecast(models["XGBOOST"], store_frame, split_date, features)
        forecast_seconds["XGBOOST"] += time.perf_counter() - row_started
        row_started = time.perf_counter()
        lgb_val, lgb_preds = recursive_forecast(models["LIGHTGBM"], store_frame, split_date, features)
        forecast_seconds["LIGHTGBM"] += time.perf_counter() - row_started
        if val.empty:
            continue
        actual = val["Sales"].astype(float).tolist()
        dates = val["Date"].dt.strftime("%Y-%m-%d").tolist()
        store_results[str(int(store_id))] = {
            "dates": dates, "actual": actual,
            "xgboost": xgb_preds, "lightgbm": lgb_preds,
        }

    for runtime in runtime_rows:
        runtime["ForecastSeconds"] = forecast_seconds[runtime["Model"]]
    feature_dir = PROJECT_ROOT / "data" / "features"
    (feature_dir / "ml_results.json").write_text(json.dumps(store_results, allow_nan=False), encoding="utf-8")
    pd.DataFrame(tuning_rows).to_csv(processed_dir / "ml_tuning_results.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(processed_dir / "ml_runtime.csv", index=False)
    metadata = {
        "model_version": 2,
        "global_max_date": global_max_date.strftime("%Y-%m-%d"),
        "split_date": split_date.strftime("%Y-%m-%d"),
        "inner_tuning_cutoff": inner_cutoff.strftime("%Y-%m-%d"),
        "feature_count": len(features), "features": features,
        "configurations": configurations,
        "completed_stores": len(store_results),
        "models": ["XGBOOST", "LIGHTGBM"],
    }
    (feature_dir / "ml_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"ML modeling complete: {len(store_results)} stores. Artifacts saved under models/ml.")
    return store_results


if __name__ == "__main__":
    source = PROJECT_ROOT / "data" / "features" / "train_features.csv"
    data = pd.read_csv(source, parse_dates=["Date"], low_memory=False)
    train_ml_models(data)
