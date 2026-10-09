import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import json

warnings.filterwarnings("ignore")

VAL_DAYS = 42


def train_eval_store(df_store, split_date):
    """
    Train and evaluate classical models for one store.

    The final evaluation window is defined by the common calendar cutoff used
    by the pooled ML models. Closed-store days are excluded so classical and
    ML models are evaluated on the same demand observations.
    """
    df_store = (
        df_store.sort_values("Date")
        .copy()
    )

    # Only forecast demand on days when the store is open.
    df_store = df_store[df_store["Open"] == 1].copy()
    if df_store.empty:
        return None

    train = df_store[df_store["Date"] <= split_date].copy()
    val = df_store[df_store["Date"] > split_date].copy()

    if train.empty or val.empty:
        return None

    # Use the actual validation length rather than assuming every store has
    # exactly 42 open days.
    horizon = len(val)
    validation_dates = val["Date"].dt.strftime("%Y-%m-%d").tolist()

    results = {
        "dates": validation_dates,
        "actual": val["Sales"].astype(float).tolist(),
    }

    # --- ARIMA ---
    print("  Fitting ARIMA...")
    try:
        # Pass a plain numeric array so statsmodels does not depend on the
        # filtered pandas index (which is no longer a supported RangeIndex
        # after removing closed-store dates). The validation dates are tracked
        # separately in `results["dates"]`.
        y_train = train["Sales"].astype(float).to_numpy()
        arima_model = ARIMA(y_train, order=(5, 1, 0))
        arima_fit = arima_model.fit()
        arima_preds = np.asarray(arima_fit.forecast(steps=horizon), dtype=float)
    except Exception as e:
        print(f"  ARIMA failed: {e}")
        arima_preds = np.full(horizon, np.nan)

    results["arima"] = arima_preds.tolist()

    # --- Prophet ---
    print("  Fitting Prophet...")
    try:
        prophet_df = train[["Date", "Sales"]].rename(
            columns={"Date": "ds", "Sales": "y"}
        )
        prophet_model = Prophet(daily_seasonality=False)
        prophet_model.fit(prophet_df)

        # Predict on the exact same validation dates as ARIMA/ML.
        future = pd.DataFrame({"ds": val["Date"].values})
        forecast = prophet_model.predict(future)
        prophet_preds = forecast["yhat"].to_numpy(dtype=float)
    except Exception as e:
        print(f"  Prophet failed: {e}")
        prophet_preds = np.full(horizon, np.nan)

    results["prophet"] = prophet_preds.tolist()
    return results


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("../data/features", exist_ok=True)

    print("Loading feature dataset...")
    df = pd.read_csv("../data/features/train_features.csv", parse_dates=["Date"])
    global_max_date = df["Date"].max()
    split_date = global_max_date - pd.Timedelta(days=VAL_DAYS)

    # The proposal explicitly allows a representative subset for classical
    # models when fitting all 1,115 per-store models is too expensive.
    store_meta = pd.read_csv("../data/raw/store.csv")
    n_per_type = 5
    sample_stores = (
        store_meta.groupby("StoreType")["Store"]
        .apply(lambda x: x.sample(min(n_per_type, len(x)), random_state=42))
        .reset_index(drop=True)
        .astype(int)
        .tolist()
    )

    print(
        f"Running classical models for {len(sample_stores)} stratified stores: "
        f"{sorted(sample_stores)}"
    )

    store_results = {}
    for store_id in sample_stores:
        print(f"Processing Store {store_id}...")
        df_store = df[df["Store"] == store_id]
        if len(df_store) > 100:
            result = train_eval_store(df_store, split_date)
            if result is not None:
                store_results[str(int(store_id))] = result

    with open("../data/features/classical_results.json", "w") as f:
        json.dump(store_results, f)

    with open("../data/features/classical_sample_stores.json", "w") as f:
        json.dump(sorted(sample_stores), f, indent=2)

    print("Classical modeling complete. Results saved.")
