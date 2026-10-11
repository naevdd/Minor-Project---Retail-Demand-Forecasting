"""Per-store LSTM forecasting with aligned recursive inference and checkpoints."""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from modeling_common import get_model_features, update_sales_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
VAL_DAYS = 42
SEQ_LENGTH = 30
MAX_EPOCHS = 50
PATIENCE = 5
MODEL_VERSION = 2


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_sequences(data, target, dates, seq_length):
    xs, ys, ds = [], [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(target[i + seq_length])
        ds.append(dates[i + seq_length])
    return np.asarray(xs), np.asarray(ys), np.asarray(ds)


def _checkpoint_model(model, scaler_x, scaler_y, features, store_id, artifact_dir, split_date):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / f"store_{int(store_id):04d}.pt"
    torch.save({
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "input_dim": len(features), "hidden_dim": 64, "num_layers": 2,
        "sequence_length": SEQ_LENGTH, "features": features,
        "store_id": int(store_id), "split_date": pd.Timestamp(split_date).strftime("%Y-%m-%d"),
        "model_version": MODEL_VERSION,
    }, weights_path)
    scaler_metadata = {
        "store_id": int(store_id), "features": features,
        "x_scale": scaler_x.scale_.tolist(), "x_min": scaler_x.min_.tolist(),
        "x_data_min": scaler_x.data_min_.tolist(), "x_data_max": scaler_x.data_max_.tolist(),
        "y_scale": scaler_y.scale_.tolist(), "y_min": scaler_y.min_.tolist(),
        "y_data_min": scaler_y.data_min_.tolist(), "y_data_max": scaler_y.data_max_.tolist(),
    }
    (artifact_dir / f"store_{int(store_id):04d}_scalers.json").write_text(
        json.dumps(scaler_metadata, indent=2), encoding="utf-8"
    )
    return weights_path


def train_store(
    df_store: pd.DataFrame,
    features: list[str],
    split_date,
    forecast_end=None,
    seq_length: int = SEQ_LENGTH,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    artifact_dir: Path | None = None,
):
    """Fit one LSTM to a global cutoff; predict each validation row before appending it to history."""
    started_total = time.perf_counter()
    frame = df_store.sort_values("Date").copy()
    if "Open" in frame.columns:
        frame = frame[frame["Open"].eq(1)]
    if forecast_end is not None:
        frame = frame[frame["Date"] <= pd.Timestamp(forecast_end)]
    train_df = frame[frame["Date"] <= pd.Timestamp(split_date)].copy()
    test_df = frame[frame["Date"] > pd.Timestamp(split_date)].copy()
    if test_df.empty or len(train_df) < seq_length + 60:
        return None

    # Early stopping uses only the pre-test training window.
    internal_val_days = min(42, max(14, len(train_df) // 10))
    internal_split = train_df["Date"].max() - pd.Timedelta(days=internal_val_days)
    fit_df = train_df[train_df["Date"] <= internal_split].copy()
    if len(fit_df) <= seq_length + 10:
        return None

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(fit_df[features])
    scaler_y.fit(fit_df[["Sales"]])

    train_scaled = scaler_x.transform(train_df[features])
    target_scaled = scaler_y.transform(train_df[["Sales"]])
    seq_dates = train_df["Date"].to_numpy()
    X_seq, y_seq, target_dates = create_sequences(train_scaled, target_scaled, seq_dates, seq_length)
    fit_mask = target_dates <= np.datetime64(internal_split)
    early_mask = target_dates > np.datetime64(internal_split)
    X_fit_np, y_fit_np = X_seq[fit_mask], y_seq[fit_mask]
    X_early_np, y_early_np = X_seq[early_mask], y_seq[early_mask]
    if len(X_fit_np) == 0 or len(X_early_np) == 0:
        return None

    X_fit = torch.as_tensor(X_fit_np, dtype=torch.float32, device=DEVICE)
    y_fit = torch.as_tensor(y_fit_np, dtype=torch.float32, device=DEVICE)
    X_early = torch.as_tensor(X_early_np, dtype=torch.float32, device=DEVICE)
    y_early = torch.as_tensor(y_early_np, dtype=torch.float32, device=DEVICE)
    loader = DataLoader(TensorDataset(X_fit, y_fit), batch_size=32, shuffle=True)

    model = LSTMForecaster(input_dim=len(features)).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss, best_state, waiting, epochs_run = float("inf"), None, 0, 0

    fit_start = time.perf_counter()
    for _ in range(max_epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(X_early), y_early).item())
        epochs_run += 1
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            waiting = 0
        else:
            waiting += 1
        if waiting >= patience:
            break
    fit_seconds = time.perf_counter() - fit_start
    if best_state is not None:
        model.load_state_dict(best_state)

    # Aligned blind recursive prediction:
    # 1) use the previous 30 rows to predict the current row;
    # 2) only then append this row's known covariates and predicted sale for later steps.
    model.eval()
    history_sales = train_df["Sales"].astype(float).tolist()
    history_features = [row for row in scaler_x.transform(train_df[features])]
    predictions = []
    forecast_start = time.perf_counter()
    for _, base_row in test_df.iterrows():
        feature_row = update_sales_features(base_row[features], history_sales)
        scaled_row = scaler_x.transform(pd.DataFrame([feature_row], columns=features))[0]
        sequence = np.asarray(history_features[-seq_length:], dtype=np.float32)
        if len(sequence) < seq_length:
            # This should be prevented by the training-length guard above.
            raise RuntimeError("Insufficient history for an LSTM inference sequence.")
        X_test = torch.as_tensor(sequence[None, :, :], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            prediction_scaled = model(X_test).detach().cpu().numpy()
        prediction = float(scaler_y.inverse_transform(prediction_scaled)[0, 0])
        prediction = max(0.0, prediction) if np.isfinite(prediction) else 0.0
        predictions.append(prediction)
        history_sales.append(prediction)
        history_features.append(scaled_row)
    forecast_seconds = time.perf_counter() - forecast_start

    artifact_path = None
    if artifact_dir is not None:
        artifact_path = _checkpoint_model(
            model, scaler_x, scaler_y, features, int(df_store["Store"].iloc[0]), artifact_dir, split_date
        )
    return {
        "dates": test_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "actual": test_df["Sales"].astype(float).tolist(),
        "lstm": predictions,
        "epochs_run": epochs_run,
        "fit_seconds": fit_seconds,
        "forecast_seconds": forecast_seconds,
        "artifact": str(artifact_path.relative_to(PROJECT_ROOT)) if artifact_path else None,
        "total_seconds": time.perf_counter() - started_total,
    }


def _write_results(results: dict, path: Path):
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(results, allow_nan=False), encoding="utf-8")
    temp_path.replace(path)


def main():
    feature_path = PROJECT_ROOT / "data" / "features" / "train_features.csv"
    frame = pd.read_csv(feature_path, parse_dates=["Date"], low_memory=False).sort_values(["Store", "Date"])
    global_max_date = frame["Date"].max()  # Calculate once globally, before splitting per store.
    split_date = global_max_date - pd.Timedelta(days=VAL_DAYS)
    if "Open" in frame.columns:
        frame = frame[frame["Open"].eq(1)].copy()
    features = get_model_features(frame)
    store_ids = sorted(frame["Store"].astype(int).unique())

    feature_dir = PROJECT_ROOT / "data" / "features"
    output_path = feature_dir / "dl_results.json"
    metadata_path = feature_dir / "dl_run_metadata.json"
    artifact_dir = PROJECT_ROOT / "models" / "lstm"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_version": MODEL_VERSION,
        "global_max_date": global_max_date.strftime("%Y-%m-%d"),
        "split_date": split_date.strftime("%Y-%m-%d"),
        "features": features, "sequence_length": SEQ_LENGTH,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
    }

    results = {}
    if output_path.exists() and metadata_path.exists():
        try:
            old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if old_metadata == metadata:
                results = json.loads(output_path.read_text(encoding="utf-8"))
                print(f"Resuming only from compatible results: {len(results)} stores")
            else:
                print("Existing LSTM output uses a different configuration; rebuilding results.")
        except (json.JSONDecodeError, OSError):
            print("Existing LSTM results are invalid; rebuilding results.")
    elif output_path.exists():
        print("Legacy LSTM results lack a run manifest; rebuilding to avoid stale forecasts.")

    runtime_rows = []
    print(f"Training LSTM for {len(store_ids)} stores; global cutoff={split_date.date()} on {DEVICE}.")
    for index, store_id in enumerate(store_ids, start=1):
        key = str(int(store_id))
        checkpoint = artifact_dir / f"store_{int(store_id):04d}.pt"
        if key in results and checkpoint.exists():
            continue
        store_frame = frame[frame["Store"].astype(int).eq(int(store_id))].copy()
        result = train_store(
            store_frame, features, split_date,
            seq_length=SEQ_LENGTH, max_epochs=MAX_EPOCHS, patience=PATIENCE,
            artifact_dir=artifact_dir,
        )
        if result is None:
            print(f"[{index}/{len(store_ids)}] Store {store_id}: skipped (insufficient training/validation data)")
            continue
        results[key] = result
        runtime_rows.append({
            "Model": "LSTM", "Store": int(store_id), "Scope": "Per-store model",
            "FitSeconds": result["fit_seconds"], "ForecastSeconds": result["forecast_seconds"],
            "TuningSeconds": 0.0, "TotalSeconds": result["total_seconds"], "Artifact": result["artifact"],
        })
        print(f"[{index}/{len(store_ids)}] Store {store_id}: {result['epochs_run']} epochs, {result['total_seconds']:.1f}s")
        if index % 25 == 0:
            _write_results(results, output_path)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            pd.DataFrame(runtime_rows).to_csv(PROJECT_ROOT / "data" / "processed" / "dl_runtime_partial.csv", index=False)

    _write_results(results, output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Rebuild runtime coverage from the completed result file so a resumed run
    # still reports durations for stores completed in an earlier process.
    complete_runtime_rows = []
    for store_key, result in results.items():
        if not isinstance(result, dict) or "lstm" not in result:
            continue
        complete_runtime_rows.append({
            "Model": "LSTM", "Store": int(store_key), "Scope": "Per-store model",
            "FitSeconds": float(result.get("fit_seconds", 0.0)),
            "ForecastSeconds": float(result.get("forecast_seconds", 0.0)),
            "TuningSeconds": 0.0, "TotalSeconds": float(result.get("total_seconds", 0.0)),
            "Artifact": result.get("artifact", f"models/lstm/store_{int(store_key):04d}.pt"),
        })
    pd.DataFrame(complete_runtime_rows).to_csv(PROJECT_ROOT / "data" / "processed" / "dl_runtime.csv", index=False)

    # A partial runtime CSV is useful for recovery during an interrupted run,
    # but is redundant once every store has a completed forecast and the full
    # runtime report above has been written.
    partial_runtime_path = PROJECT_ROOT / "data" / "processed" / "dl_runtime_partial.csv"
    all_stores_complete = all(
        str(int(store_id)) in results
        and isinstance(results[str(int(store_id))], dict)
        and "lstm" in results[str(int(store_id))]
        for store_id in store_ids
    )
    if all_stores_complete:
        partial_runtime_path.unlink(missing_ok=True)

    print(f"LSTM modeling complete: {len(results)} stores. Checkpoints: {artifact_dir}")


if __name__ == "__main__":
    main()
