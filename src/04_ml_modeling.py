import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
import json
import re

VAL_DAYS = 42
LAG_COLUMNS = {
    "Sales_Lag_1": 1,
    "Sales_Lag_7": 7,
    "Sales_Lag_14": 14,
    "Sales_Lag_30": 30,
}
ROLLING_COLUMNS = {
    "Sales_RollMean_7": (7, "mean"),
    "Sales_RollStd_7": (7, "std"),
    "Sales_RollMean_14": (14, "mean"),
    "Sales_RollStd_14": (14, "std"),
}


def _recursive_feature_row(row, history):
    """Replace sales-derived validation features with information available at prediction time."""
    row = row.copy()
    history = pd.Series(history, dtype=float)

    for col, lag in LAG_COLUMNS.items():
        row[col] = float(history.iloc[-lag]) if len(history) >= lag else 0.0

    for col, (window, stat) in ROLLING_COLUMNS.items():
        if len(history) == 0:
            value = 0.0
        else:
            values = history.iloc[-window:]
            value = values.mean() if stat == "mean" else values.std()
            if pd.isna(value):
                value = 0.0
        row[col] = float(value)

    return row


def recursive_forecast(model, store_frame, split_date, features):
    """Forecast each validation date recursively without using future actual sales."""
    train_store = store_frame[store_frame["Date"] <= split_date].sort_values("Date")
    val_store = store_frame[store_frame["Date"] > split_date].sort_values("Date")

    history = train_store["Sales"].astype(float).tolist()
    predictions = []

    for _, row in val_store.iterrows():
        feature_row = _recursive_feature_row(row[features], history)
        X = pd.DataFrame([feature_row], columns=features)
        pred = float(model.predict(X)[0])
        # Sales cannot be negative.
        pred = max(0.0, pred)
        predictions.append(pred)
        history.append(pred)

    return val_store, predictions


def save_feature_importance(model, feature_names, path, model_name, importance_type="gain"):
    """Save normalized feature importance for reporting and dashboard visualization."""
    if model_name == "XGBOOST":
        # The sklearn feature_importances_ for XGBoost uses the model's
        # configured importance measure (gain by default for this estimator).
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif model_name == "LIGHTGBM":
        importance = np.asarray(
            model.booster_.feature_importance(importance_type=importance_type),
            dtype=float,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    imp_df = pd.DataFrame({
        "Feature": list(feature_names),
        "Importance": importance,
    }).sort_values("Importance", ascending=False)

    total = imp_df["Importance"].sum()
    if total > 0:
        imp_df["ImportancePct"] = imp_df["Importance"] / total * 100.0
    else:
        imp_df["ImportancePct"] = 0.0

    imp_df["Model"] = model_name
    imp_df.to_csv(path, index=False)
    return imp_df


def train_ml_models(df, val_days=VAL_DAYS):
    print("Preparing data for ML models...")
    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "_", x))
    df = df.sort_values(["Store", "Date"]).copy()

    # Forecast demand only on open days, matching the classical and DL stages.
    df = df[df["Open"] == 1].copy()

    max_date = df["Date"].max()
    split_date = max_date - pd.Timedelta(days=val_days)

    train = df[df["Date"] <= split_date].copy()
    val = df[df["Date"] > split_date].copy()

    drop_cols = ["Date", "Sales", "Customers"]
    features = [c for c in train.columns if c not in drop_cols]

    X_train, y_train = train[features], train["Sales"]
    print(f"Training on {len(X_train)} samples, common validation cutoff: {split_date.date()}.")

    # --- XGBoost ---
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)

    # --- LightGBM ---
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )
    lgb_model.fit(X_train, y_train)

    processed_dir = os.path.join("..", "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    save_feature_importance(
        xgb_model,
        features,
        os.path.join(processed_dir, "xgboost_feature_importance.csv"),
        "XGBOOST",
    )
    save_feature_importance(
        lgb_model,
        features,
        os.path.join(processed_dir, "lightgbm_feature_importance.csv"),
        "LIGHTGBM",
    )
    print("Feature importance saved for XGBoost and LightGBM.")

    # Generate recursive forecasts separately for every store. The model is
    # still pooled across all stores; only the validation rollout is store-aware.
    store_results = {}
    for store_id, store_frame in df.groupby("Store", sort=True):
        store_frame = store_frame.sort_values("Date")
        val_store = store_frame[store_frame["Date"] > split_date]
        if val_store.empty:
            continue

        xgb_val, xgb_preds = recursive_forecast(
            xgb_model, store_frame, split_date, features
        )
        lgb_val, lgb_preds = recursive_forecast(
            lgb_model, store_frame, split_date, features
        )

        # Both models use the exact same store/date validation rows.
        actual = xgb_val["Sales"].astype(float).tolist()
        dates = xgb_val["Date"].dt.strftime("%Y-%m-%d").tolist()

        store_results[str(int(store_id))] = {
            "dates": dates,
            "actual": actual,
            "xgboost": xgb_preds,
            "lightgbm": lgb_preds,
        }

    return store_results


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Loading feature dataset...")
    df = pd.read_csv("../data/features/train_features.csv", parse_dates=["Date"])

    store_results = train_ml_models(df)

    with open("../data/features/ml_results.json", "w") as f:
        json.dump(store_results, f)

    print(f"ML modeling complete. {len(store_results)} stores saved.")
