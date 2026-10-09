import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

VAL_DAYS = 42
SEQ_LENGTH = 30


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_sequences(data, target, dates, seq_length):
    xs, ys, ds = [], [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i : i + seq_length])
        ys.append(target[i + seq_length])
        ds.append(dates[i + seq_length])
    return np.array(xs), np.array(ys), np.array(ds)


def _update_sales_features(row, history):
    """Create validation-time lag/rolling features using only observed/predicted history."""
    row = row.copy()
    history = pd.Series(history, dtype=float)

    for col, lag in {
        "Sales_Lag_1": 1,
        "Sales_Lag_7": 7,
        "Sales_Lag_14": 14,
        "Sales_Lag_30": 30,
    }.items():
        row[col] = float(history.iloc[-lag]) if len(history) >= lag else 0.0

    for col, window, stat in [
        ("Sales_RollMean_7", 7, "mean"),
        ("Sales_RollStd_7", 7, "std"),
        ("Sales_RollMean_14", 14, "mean"),
        ("Sales_RollStd_14", 14, "std"),
    ]:
        if len(history) == 0:
            value = 0.0
        else:
            values = history.iloc[-window:]
            value = values.mean() if stat == "mean" else values.std()
            if pd.isna(value):
                value = 0.0
        row[col] = float(value)

    return row


def train_store(
    df_store,
    features,
    seq_length=SEQ_LENGTH,
    val_days=VAL_DAYS,
    max_epochs=50,
    patience=5,
):
    """
    Train one LSTM with the common calendar validation cutoff.

    The final test period is rolled forward recursively: after the first
    prediction, that prediction—not the future actual—is used to construct
    subsequent lag/rolling features. This makes the LSTM forecast horizon
    comparable with the blind multi-step classical forecasts.
    """
    df_store = df_store.sort_values("Date").copy()
    df_store = df_store[df_store["Open"] == 1].copy()

    if len(df_store) < seq_length + val_days + 20:
        return None

    split_date = df_store["Date"].max() - pd.Timedelta(days=val_days)
    train_df = df_store[df_store["Date"] <= split_date].copy()
    test_df = df_store[df_store["Date"] > split_date].copy()

    if len(test_df) == 0 or len(train_df) <= seq_length + 10:
        return None

    # Internal validation is carved from the training period so the final
    # evaluation period remains untouched by early stopping.
    internal_val_days = min(42, max(14, len(train_df) // 10))
    internal_split = train_df["Date"].max() - pd.Timedelta(days=internal_val_days)
    fit_df = train_df[train_df["Date"] <= internal_split].copy()

    if len(fit_df) <= seq_length:
        return None

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(fit_df[features])
    scaler_y.fit(fit_df[["Sales"]])

    # Training/early-validation sequences use only the historical training
    # period. The scaler itself is fitted strictly on fit_df.
    train_X_scaled = scaler_x.transform(train_df[features])
    train_y_scaled = scaler_y.transform(train_df[["Sales"]])
    train_dates = train_df["Date"].to_numpy()
    X_seq, y_seq, seq_dates = create_sequences(
        train_X_scaled, train_y_scaled, train_dates, seq_length
    )

    fit_mask = seq_dates <= np.datetime64(internal_split)
    early_mask = seq_dates > np.datetime64(internal_split)

    X_fit_np = X_seq[fit_mask]
    y_fit_np = y_seq[fit_mask]
    X_early_np = X_seq[early_mask]
    y_early_np = y_seq[early_mask]

    if len(X_fit_np) == 0 or len(X_early_np) == 0:
        return None

    X_fit = torch.tensor(X_fit_np, dtype=torch.float32).to(device)
    y_fit = torch.tensor(y_fit_np, dtype=torch.float32).to(device)
    X_early = torch.tensor(X_early_np, dtype=torch.float32).to(device)
    y_early = torch.tensor(y_early_np, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_fit, y_fit), batch_size=32, shuffle=True
    )

    model = LSTMForecaster(
        input_dim=X_fit.shape[2], hidden_dim=64, num_layers=2, output_dim=1
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(max_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_early), y_early).item()

        epochs_run += 1
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Recursive multi-step test forecast. We start with the actual training
    # history, then append each prediction as we move through the test window.
    model.eval()
    history_sales = train_df["Sales"].astype(float).tolist()
    history_features = [
        row for row in scaler_x.transform(train_df[features])
    ]
    predictions = []

    for _, base_row in test_df.iterrows():
        feature_row = _update_sales_features(base_row[features], history_sales)
        scaled_row = scaler_x.transform(pd.DataFrame([feature_row], columns=features))[0]
        history_features.append(scaled_row)

        seq_input = np.asarray(history_features[-seq_length:], dtype=np.float32)
        X_test = torch.tensor(seq_input[None, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_scaled = model(X_test).cpu().numpy()

        pred = float(scaler_y.inverse_transform(pred_scaled)[0, 0])
        pred = max(0.0, pred)
        predictions.append(pred)
        history_sales.append(pred)

    actual = test_df["Sales"].astype(float).tolist()
    dates = test_df["Date"].dt.strftime("%Y-%m-%d").tolist()

    return {
        "dates": dates,
        "actual": actual,
        "lstm": predictions,
        "epochs_run": epochs_run,
    }


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Loading feature dataset for DL...")
    df = pd.read_csv("../data/features/train_features.csv", parse_dates=["Date"])
    df = df[df["Open"] == 1].sort_values(["Store", "Date"])

    drop_cols = ["Date", "Sales", "Customers"]
    features = [c for c in df.columns if c not in drop_cols]

    all_stores = df["Store"].unique()
    print(
        f"Training LSTM for ALL {len(all_stores)} stores | "
        f"max_epochs=50 | early_stopping patience=5"
    )

    out_path = "../data/features/dl_results.json"
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            store_results = json.load(f)
        print(f"Resuming from {len(store_results)} already-completed stores.")
    else:
        store_results = {}

    for i, target_store in enumerate(all_stores):
        store_key = str(int(target_store))
        if store_key in store_results:
            continue

        df_store = df[df["Store"] == target_store].copy()
        result = train_store(
            df_store,
            features,
            SEQ_LENGTH,
            VAL_DAYS,
            max_epochs=50,
            patience=5,
        )

        if result is None:
            print(f"[{i+1}/{len(all_stores)}] Store {target_store}: Skipped")
            continue

        store_results[store_key] = result
        print(
            f"[{i+1}/{len(all_stores)}] Store {target_store}: "
            f"Done ({result['epochs_run']} epochs)"
        )

        if (i + 1) % 50 == 0:
            with open(out_path, "w") as f:
                json.dump(store_results, f)
            print(f"  >>> Checkpoint saved ({len(store_results)} stores completed)")

    with open(out_path, "w") as f:
        json.dump(store_results, f)

    print(f"\nDL modeling complete. {len(store_results)} stores saved to {out_path}")
