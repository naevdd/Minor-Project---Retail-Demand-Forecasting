import pandas as pd
import numpy as np
import os
import json


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError(
            f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}"
        )

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return np.nan, np.nan, np.nan

    # RMSE/MAE are calculated on the same open-store validation observations.
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))

    # MAPE excludes zero-demand observations because percentage error is undefined there.
    mask = actual > 0
    if mask.any():
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.nan

    return rmse, mae, mape


def add_records(records, store, model, data):
    for prediction_key in [model.lower()]:
        if prediction_key not in data:
            continue

        actual = data.get("actual", [])
        predicted = data[prediction_key]
        dates = data.get("dates", [])

        if not dates:
            # Old result files are intentionally rejected so stale, misaligned
            # results cannot silently enter the benchmark.
            print(f"Skipping {model} / Store {store}: no validation dates in result file.")
            return

        if not (len(actual) == len(predicted) == len(dates)):
            print(
                f"Skipping {model} / Store {store}: inconsistent lengths "
                f"dates={len(dates)}, actual={len(actual)}, predicted={len(predicted)}."
            )
            return

        rmse, mae, mape = calculate_metrics(actual, predicted)
        records.append(
            {
                "Store": int(store),
                "Model": model.upper(),
                "RMSE": rmse,
                "MAE": mae,
                "MAPE": mape,
                "ValidationDays": len(dates),
            }
        )


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Evaluating models...")

    eval_records = []

    files = {
        "classical": ("../data/features/classical_results.json", ["ARIMA", "PROPHET"]),
        "ml": ("../data/features/ml_results.json", ["XGBOOST", "LIGHTGBM"]),
        "dl": ("../data/features/dl_results.json", ["LSTM"]),
    }

    for _, (path, models) in files.items():
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            results = json.load(f)
        for store, data in results.items():
            for model in models:
                add_records(eval_records, store, model, data)

    df_metrics = pd.DataFrame(eval_records)
    if df_metrics.empty:
        raise RuntimeError("No valid model results found. Run the model scripts first.")

    out_path = "../data/processed/metrics.csv"
    df_metrics.to_csv(out_path, index=False)

    # Common benchmark = stores for which every currently evaluated model exists.
    model_count = df_metrics["Model"].nunique()
    common_stores = (
        df_metrics.groupby("Store")["Model"].nunique()
        .loc[lambda s: s == model_count]
        .index.tolist()
    )

    with open("../data/processed/common_benchmark_stores.json", "w") as f:
        json.dump([int(s) for s in common_stores], f, indent=2)

    print("\n--- Coverage ---")
    print(
        df_metrics.groupby("Model")["Store"]
        .nunique()
        .rename("Stores")
        .to_string()
    )

    print(f"\nCommon benchmark stores: {len(common_stores)}")
    print("\n--- Fair benchmark: common stores only ---")
    fair = df_metrics[df_metrics["Store"].isin(common_stores)]
    print(fair.groupby("Model")[['RMSE', 'MAE', 'MAPE']].mean().round(2))

    print(f"\nDetailed metrics saved to {out_path}")
