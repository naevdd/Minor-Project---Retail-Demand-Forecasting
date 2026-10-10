"""Unified fixed-holdout metrics, rolling-origin metrics and complexity reporting."""
from pathlib import Path
from itertools import combinations
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) != len(predicted):
        raise ValueError(f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}")
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) == 0:
        return np.nan, np.nan, np.nan
    errors = np.abs(actual - predicted)
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(errors))
    positive = actual > 0
    mape = float(np.mean(errors[positive] / actual[positive]) * 100.0) if positive.any() else np.nan
    return rmse, mae, mape


def add_records(records, store, model, data):
    key = model.lower()
    if key not in data:
        return
    actual, predicted, dates = data.get("actual", []), data[key], data.get("dates", [])
    if not dates or not (len(actual) == len(predicted) == len(dates)):
        print(f"Skipping {model} / Store {store}: missing or misaligned forecast arrays.")
        return
    rmse, mae, mape = calculate_metrics(actual, predicted)
    records.append({
        "Store": int(store), "Model": model.upper(), "RMSE": rmse, "MAE": mae,
        "MAPE": mape, "ValidationDays": len(dates),
    })


def summarize_runtime(output_dir: Path) -> pd.DataFrame:
    frames = []
    for filename in ("classical_runtime.csv", "ml_runtime.csv", "dl_runtime.csv"):
        path = output_dir / filename
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty and {"Model", "FitSeconds", "ForecastSeconds"}.issubset(frame.columns):
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["Model", "StoresOrModels", "FitSeconds", "ForecastSeconds", "TuningSeconds", "TotalSeconds"])
    runtime = pd.concat(frames, ignore_index=True, sort=False)
    for column in ("FitSeconds", "ForecastSeconds", "TuningSeconds"):
        if column not in runtime.columns:
            runtime[column] = 0.0
        runtime[column] = pd.to_numeric(runtime[column], errors="coerce").fillna(0.0)
    result = runtime.groupby("Model", as_index=False).agg(
        StoresOrModels=("Model", "size"), FitSeconds=("FitSeconds", "sum"),
        ForecastSeconds=("ForecastSeconds", "sum"), TuningSeconds=("TuningSeconds", "sum"),
    )
    result["TotalSeconds"] = result[["FitSeconds", "ForecastSeconds", "TuningSeconds"]].sum(axis=1)
    return result.sort_values("Model")


def generate_rolling_reports(output_dir: Path):
    path = output_dir / "rolling_origin_results.csv"
    if not path.exists():
        print("Rolling-origin results not found; skipping rolling metrics and significance tests.")
        return None, None
    results = pd.read_csv(path, parse_dates=["Date"])
    required = {"Fold", "Store", "Model", "Actual", "Prediction"}
    if not required.issubset(results.columns):
        print("Rolling-origin results do not have the expected columns; skipping significance tests.")
        return None, None

    summary_rows = []
    for fold, group in list(results.groupby("Fold")) + [("All", results)]:
        for model, model_frame in group.groupby("Model"):
            rmse, mae, mape = calculate_metrics(model_frame["Actual"], model_frame["Prediction"])
            summary_rows.append({
                "Fold": fold, "Model": model, "RMSE": rmse, "MAE": mae, "MAPE": mape,
                "ForecastObservations": len(model_frame), "Stores": model_frame["Store"].nunique(),
            })
    rolling_metrics = pd.DataFrame(summary_rows)
    rolling_metrics.to_csv(output_dir / "rolling_origin_metrics.csv", index=False)

    # Average each store's fold-level MAE first. The paired statistical unit is
    # the store (not each store-fold), preventing repeated folds from inflating n.
    blocks = results.assign(AbsoluteError=(results["Actual"] - results["Prediction"]).abs())
    store_errors = blocks.groupby(["Store", "Fold", "Model"], as_index=False)["AbsoluteError"].mean()
    store_errors = store_errors.groupby(["Store", "Model"], as_index=False)["AbsoluteError"].mean()
    pivot = store_errors.pivot(index="Store", columns="Model", values="AbsoluteError")
    models = sorted(pivot.columns.tolist())
    tests = []
    for model_a, model_b in combinations(models, 2):
        paired = pivot[[model_a, model_b]].dropna()
        if len(paired) < 2:
            continue
        differences = paired[model_a] - paired[model_b]
        try:
            test_result = wilcoxon(
                paired[model_a], paired[model_b], alternative="two-sided",
                zero_method="wilcox",
            )
            raw_p = float(test_result.pvalue)
        except ValueError:
            raw_p = 1.0
        tests.append({
            "ModelA": model_a, "ModelB": model_b,
            "MeanMAEDifference_A_minus_B": float(differences.mean()),
            "PairedStores": int(len(paired)), "RawPValue": raw_p,
            "HolmAdjustedPValue": np.nan,
            "Test": "Paired Wilcoxon signed-rank on store MAE averaged over folds; Holm-adjusted",
        })
    if tests:
        order = np.argsort([row["RawPValue"] for row in tests])
        m = len(tests)
        running_adjusted = 0.0
        for rank, idx in enumerate(order):
            adjusted = min(1.0, (m - rank) * tests[idx]["RawPValue"])
            running_adjusted = max(running_adjusted, adjusted)
            tests[idx]["HolmAdjustedPValue"] = running_adjusted
    significance = pd.DataFrame(tests)
    significance.to_csv(output_dir / "significance_tests.csv", index=False)
    return rolling_metrics, significance


def validate_run_manifests(feature_dir: Path) -> dict:
    manifest_names = ["classical_run_metadata.json", "ml_run_metadata.json", "dl_run_metadata.json"]
    loaded = {}
    for filename in manifest_names:
        path = feature_dir / filename
        if not path.exists():
            raise RuntimeError(
                f"Required current-run manifest is missing: {path.name}. "
                "Run python run_pipeline.py from the project root; do not evaluate bundled legacy results."
            )
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Could not read {path.name}: {exc}") from exc
        if metadata.get("model_version") != 2:
            raise RuntimeError(f"{path.name} was generated by an incompatible model version; rerun the pipeline.")
        loaded[filename] = metadata
    dates = {metadata.get("global_max_date") for metadata in loaded.values()}
    cutoffs = {metadata.get("split_date") for metadata in loaded.values()}
    if len(dates) != 1 or None in dates or len(cutoffs) != 1 or None in cutoffs:
        raise RuntimeError("Classical, ML, and LSTM output manifests do not share the same global date/cutoff.")
    if loaded["dl_run_metadata.json"].get("model_version") != 2:
        raise RuntimeError("LSTM results are not from the current implementation.")
    return {"global_max_date": dates.pop(), "split_date": cutoffs.pop()}


def main():
    started = time.perf_counter()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    feature_dir = PROJECT_ROOT / "data" / "features"
    processed_dir.mkdir(parents=True, exist_ok=True)
    validated_run = validate_run_manifests(feature_dir)
    evaluation_records = []
    result_files = [
        (feature_dir / "classical_results.json", ["ARIMA", "PROPHET"]),
        (feature_dir / "ml_results.json", ["XGBOOST", "LIGHTGBM"]),
        (feature_dir / "dl_results.json", ["LSTM"]),
    ]
    for path, models in result_files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            results = json.load(file)
        for store, data in results.items():
            for model in models:
                add_records(evaluation_records, store, model, data)

    metrics = pd.DataFrame(evaluation_records)
    if metrics.empty:
        raise RuntimeError("No valid model results found. Run modeling stages before evaluation.")
    metrics.to_csv(processed_dir / "metrics.csv", index=False)

    model_count = metrics["Model"].nunique()
    common_stores = (
        metrics.groupby("Store")["Model"].nunique().loc[lambda series: series == model_count].index.tolist()
    )
    (processed_dir / "common_benchmark_stores.json").write_text(
        json.dumps([int(store) for store in common_stores], indent=2), encoding="utf-8"
    )
    fair = metrics[metrics["Store"].isin(common_stores)]
    aggregate = fair.groupby("Model", as_index=False)[["RMSE", "MAE", "MAPE"]].mean()
    aggregate["CommonStores"] = len(common_stores)
    aggregate.to_csv(processed_dir / "common_benchmark_metrics.csv", index=False)

    runtime = summarize_runtime(processed_dir)
    runtime.to_csv(processed_dir / "runtime_summary.csv", index=False)
    complexity = aggregate.merge(runtime, on="Model", how="left")
    complexity.to_csv(processed_dir / "accuracy_complexity_summary.csv", index=False)
    rolling_metrics, significance = generate_rolling_reports(processed_dir)

    print("\n--- Model coverage ---")
    print(metrics.groupby("Model")["Store"].nunique().rename("Stores").to_string())
    print(f"\nCommon five-model benchmark stores: {len(common_stores)}")
    print(aggregate.round(2).to_string(index=False))
    if rolling_metrics is not None:
        print("\n--- Rolling-origin metrics (all folds) ---")
        print(rolling_metrics[rolling_metrics["Fold"].astype(str).eq("All")].round(2).to_string(index=False))
    if significance is not None and not significance.empty:
        print(f"\nPaired significance comparisons saved: {len(significance)}")

    # Manifest lets the dashboard distinguish post-fix reports from bundled
    # demonstration outputs produced by an earlier code version.
    feature_path = feature_dir / "train_features.csv"
    if feature_path.exists():
        feature_dates = pd.read_csv(feature_path, usecols=["Date"], parse_dates=["Date"])["Date"]
        latest_date = feature_dates.max().strftime("%Y-%m-%d")
        final_cutoff = (feature_dates.max() - pd.Timedelta(days=42)).strftime("%Y-%m-%d")
    else:
        latest_date, final_cutoff = None, None
    manifest = {
        "pipeline_version": 2,
        "evaluated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "latest_dataset_date": latest_date,
        "final_holdout_cutoff": final_cutoff,
        "validated_model_manifests": True,
        "model_output_date": validated_run["global_max_date"],
        "model_output_split_date": validated_run["split_date"],
        "fixed_holdout_models": sorted(metrics["Model"].unique().tolist()),
        "common_benchmark_stores": len(common_stores),
        "rolling_origin_available": (processed_dir / "rolling_origin_results.csv").exists(),
        "significance_tests_available": (processed_dir / "significance_tests.csv").exists(),
        "runtime_summary_available": (processed_dir / "runtime_summary.csv").exists(),
    }
    (processed_dir / "pipeline_metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nEvaluation outputs written to {processed_dir} in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
