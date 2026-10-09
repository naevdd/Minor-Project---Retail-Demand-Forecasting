import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Retail Demand Forecasting",
    layout="wide",
    page_icon="📈",
)


@st.cache_data
def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
metrics_path = os.path.join(BASE_DIR, "data/processed/metrics.csv")
classical_path = os.path.join(BASE_DIR, "data/features/classical_results.json")
ml_path = os.path.join(BASE_DIR, "data/features/ml_results.json")
dl_path = os.path.join(BASE_DIR, "data/features/dl_results.json")
common_path = os.path.join(BASE_DIR, "data/processed/common_benchmark_stores.json")
xgb_importance_path = os.path.join(BASE_DIR, "data/processed/xgboost_feature_importance.csv")
lgb_importance_path = os.path.join(BASE_DIR, "data/processed/lightgbm_feature_importance.csv")


df_metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
c_res = load_json(classical_path)
ml_res = load_json(ml_path)
dl_res = load_json(dl_path)
common_stores = {int(s) for s in load_json(common_path) if str(s).strip().isdigit()}

if not df_metrics.empty:
    df_metrics["Store"] = pd.to_numeric(df_metrics["Store"], errors="coerce")
    df_metrics = df_metrics.dropna(subset=["Store"]).copy()
    df_metrics["Store"] = df_metrics["Store"].astype(int)
    df_metrics["Model"] = df_metrics["Model"].astype(str).str.upper()

    # Derive coverage from the current metrics file so missing/stale optional
    # metadata never hides the stores that actually have all five models.
    all_metric_models = set(df_metrics["Model"].unique())
    metrics_common_stores = {
        int(store)
        for store, group in df_metrics.groupby("Store")
        if all_metric_models.issubset(set(group["Model"]))
    }
    if metrics_common_stores:
        common_stores = metrics_common_stores

MODEL_CATEGORIES = {
    "ARIMA": "🏛️ Classical",
    "PROPHET": "🏛️ Classical",
    "XGBOOST": "🌳 Machine Learning",
    "LIGHTGBM": "🌳 Machine Learning",
    "LSTM": "🧠 Deep Learning",
}

MODEL_COLORS = {
    "ARIMA": "#636EFA",
    "PROPHET": "#EF553B",
    "XGBOOST": "#00CC96",
    "LIGHTGBM": "#AB63FA",
    "LSTM": "#FFA15A",
}

st.title("📈 Retail Demand Forecasting")

if df_metrics.empty:
    st.warning("No metrics found. Run the modeling and evaluation pipeline first.")
    st.stop()

# Reject stale result formats before presenting misleading comparisons.
result_files_are_current = all(
    (not data) or all("dates" in row for row in data.values())
    for data in (c_res, ml_res, dl_res)
)
if not result_files_are_current:
    st.warning(
        "Saved model results do not contain aligned validation dates. "
        "Re-run the updated modeling/evaluation stages before using the dashboard."
    )
    st.stop()


def get_result(store, model):
    key = str(int(store))
    if model in {"ARIMA", "PROPHET"}:
        data = c_res.get(key)
        pred_key = model.lower()
    elif model in {"XGBOOST", "LIGHTGBM"}:
        data = ml_res.get(key)
        pred_key = model.lower()
    else:
        data = dl_res.get(key)
        pred_key = "lstm"

    if not data or pred_key not in data:
        return None

    dates = data.get("dates", [])
    actual = data.get("actual", [])
    predictions = data.get(pred_key, [])
    if not dates or not (len(dates) == len(actual) == len(predictions)):
        return None

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Actual": pd.to_numeric(actual, errors="coerce"),
            "Prediction": pd.to_numeric(predictions, errors="coerce"),
        }
    ).sort_values("Date")


def calculate_metrics(actual, predicted):
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")
    frame = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if frame.empty:
        return float("nan"), float("nan"), float("nan")

    error = frame["actual"] - frame["predicted"]
    rmse = (error.pow(2).mean()) ** 0.5
    mae = error.abs().mean()
    positive = frame[frame["actual"] > 0]
    mape = (
        (positive["actual"] - positive["predicted"]).abs()
        .div(positive["actual"])
        .mean()
        * 100
        if not positive.empty
        else float("nan")
    )
    return rmse, mae, mape


# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Controls")

stores = sorted(df_metrics["Store"].astype(int).unique())
classical_stores = set(
    df_metrics.loc[df_metrics["Model"].isin(["ARIMA", "PROPHET"]), "Store"]
    .astype(int)
    .unique()
)
# Start on a store where ARIMA and Prophet exist, rather than Store 1 (which
# is not in the representative classical-model sample).
benchmark_defaults = sorted(classical_stores & common_stores)
default_store = benchmark_defaults[0] if benchmark_defaults else (stores[0] if stores else None)

def format_store_option(store):
    store = int(store)
    suffix = " · classical benchmark" if store in classical_stores else ""
    return f"Store {store}{suffix}"

selected_stores = st.sidebar.multiselect(
    "🏪 Select Store(s)",
    stores,
    default=[default_store] if default_store is not None else [],
    format_func=format_store_option,
)

if classical_stores:
    st.sidebar.caption(
        f"ARIMA/Prophet were run on {len(classical_stores)} representative stores only. "
        "Choose a store marked ‘classical benchmark’ to compare them with XGBoost, LightGBM, and LSTM."
    )

if not selected_stores:
    st.info("Select at least one store.")
    st.stop()

# Only models available for every selected store are offered, so multi-store
# comparisons cannot silently mix missing model results.
store_model_sets = []
for store in selected_stores:
    models = set(
        df_metrics.loc[df_metrics["Store"] == store, "Model"].astype(str).str.upper()
    )
    store_model_sets.append(models)

available_models = sorted(
    set.intersection(*store_model_sets) if store_model_sets else set(),
    key=lambda m: list(MODEL_CATEGORIES).index(m) if m in MODEL_CATEGORIES else 999,
)

selected_models = st.sidebar.multiselect(
    "📊 Select Models to Compare",
    available_models,
    default=available_models,
)

if not available_models:
    st.sidebar.warning(
        "The selected stores do not share any evaluated models. Select one benchmark store, "
        "or select stores with common model coverage."
    )

if any(store not in common_stores for store in selected_stores):
    st.sidebar.caption(
        "ARIMA/Prophet are available only for the representative classical-model benchmark stores."
    )
else:
    st.sidebar.caption("Selected stores are in the five-model common benchmark.")

# Determine the selectable calendar range from the chosen models/stores.
all_dates = []
for store in selected_stores:
    for model in selected_models:
        frame = get_result(store, model)
        if frame is not None:
            all_dates.extend(frame["Date"].tolist())

if not all_dates:
    st.warning("No aligned results are available for the current selection.")
    st.stop()

min_date = min(all_dates).date()
max_date = max(all_dates).date()
selected_range = st.sidebar.date_input(
    "📅 Validation Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_range, tuple):
    if len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date = end_date = selected_range[0]
else:
    start_date = end_date = selected_range

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model Categories:**\n"
    "* 🏛️ **Classical**: ARIMA, Prophet\n"
    "* 🌳 **ML**: XGBoost, LightGBM\n"
    "* 🧠 **DL**: LSTM"
)

# --- MAIN CONTENT ---
tab1, tab2, tab3 = st.tabs(
    ["📈 Forecast Explorer", "📊 Aggregate Benchmarks", "🔍 Feature Importance"]
)


with tab1:
    st.header("Forecasts vs Actuals")

    if not selected_models:
        st.info("Select at least one model from the sidebar.")
        st.stop()

    for store in selected_stores:
        st.subheader(f"Store {store}")
        model_frames = {}

        for model in selected_models:
            frame = get_result(store, model)
            if frame is None:
                continue
            frame = frame[
                (frame["Date"].dt.date >= start_date)
                & (frame["Date"].dt.date <= end_date)
            ].copy()
            if not frame.empty:
                model_frames[model] = frame

        if not model_frames:
            st.warning("No results fall inside the selected date range for this store.")
            continue

        # Compare models only on dates shared by every selected model for the store.
        common_dates = None
        for frame in model_frames.values():
            current = set(frame["Date"])
            common_dates = current if common_dates is None else common_dates.intersection(current)
        common_dates = sorted(common_dates or [])

        if not common_dates:
            st.warning("Selected models have no common validation dates for this store.")
            continue

        actual_frame = next(iter(model_frames.values()))
        actual_frame = actual_frame[actual_frame["Date"].isin(common_dates)][["Date", "Actual"]]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=actual_frame["Date"],
                y=actual_frame["Actual"],
                mode="lines+markers",
                name="Actual Sales",
                line=dict(color="black", width=3),
                marker=dict(size=5, color="black"),
            )
        )

        for model in selected_models:
            if model not in model_frames:
                continue
            frame = model_frames[model]
            frame = frame[frame["Date"].isin(common_dates)]
            fig.add_trace(
                go.Scatter(
                    x=frame["Date"],
                    y=frame["Prediction"],
                    mode="lines",
                    name=model,
                    line=dict(width=2, color=MODEL_COLORS.get(model)),
                )
            )

        fig.update_layout(
            xaxis_title="Validation Date",
            yaxis_title="Daily Sales",
            hovermode="x unified",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Common validation observations: {common_dates[0].date()} to "
            f"{common_dates[-1].date()} ({len(common_dates)} observations)."
        )

        st.markdown("**Selected-range error metrics**")
        cols = st.columns(len(selected_models))
        for i, model in enumerate(selected_models):
            if model not in model_frames:
                continue
            frame = model_frames[model]
            frame = frame[frame["Date"].isin(common_dates)]
            rmse, mae, mape = calculate_metrics(frame["Actual"], frame["Prediction"])
            cols[i].metric(
                label=model,
                value=f"{mape:.2f}% MAPE",
                delta=f"RMSE {rmse:.0f} | MAE {mae:.0f}",
                delta_color="inverse",
            )


with tab2:
    st.header("Overall Model Benchmarks")
    st.markdown(
        "The primary comparison uses the common benchmark stores so every model is represented "
        "on the same store population."
    )

    model_count = df_metrics["Model"].nunique()
    common = (
        df_metrics.groupby("Store")["Model"]
        .nunique()
        .loc[lambda s: s == model_count]
        .index
    )
    fair_df = df_metrics[df_metrics["Store"].isin(common)]
    agg_df = fair_df.groupby("Model")[["RMSE", "MAE", "MAPE"]].mean().reset_index()
    agg_df["Category"] = agg_df["Model"].map(MODEL_CATEGORIES)
    agg_df = agg_df.sort_values("MAPE")

    st.subheader(f"Fair Benchmark — {len(common)} Common Stores")
    st.dataframe(
        agg_df[["Category", "Model", "MAPE", "RMSE", "MAE"]],
        use_container_width=True,
        hide_index=True,
    )

    coverage = (
        df_metrics.groupby("Model")["Store"]
        .nunique()
        .reset_index(name="Stores Evaluated")
    )
    coverage["Category"] = coverage["Model"].map(MODEL_CATEGORIES)
    st.subheader("Model Coverage")
    st.dataframe(
        coverage[["Category", "Model", "Stores Evaluated"]],
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig_mape = px.bar(
            agg_df,
            x="Model",
            y="MAPE",
            color="Category",
            title="Average MAPE — Common Benchmark Stores (Lower is Better)",
        )
        fig_mape.update_layout(xaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_mape, use_container_width=True)

    with col2:
        fig_rmse = px.bar(
            agg_df,
            x="Model",
            y="RMSE",
            color="Category",
            title="Average RMSE — Common Benchmark Stores (Lower is Better)",
        )
        fig_rmse.update_layout(xaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_rmse, use_container_width=True)


with tab3:
    st.header("XGBoost / LightGBM Feature Importance")
    st.markdown("Feature importance from the pooled machine-learning models.")

    importance_frames = []
    if os.path.exists(xgb_importance_path):
        xgb_imp = pd.read_csv(xgb_importance_path)
        xgb_imp["Model"] = "XGBOOST"
        importance_frames.append(xgb_imp)
    if os.path.exists(lgb_importance_path):
        lgb_imp = pd.read_csv(lgb_importance_path)
        lgb_imp["Model"] = "LIGHTGBM"
        importance_frames.append(lgb_imp)

    if not importance_frames:
        st.info("Feature importance files are not available yet. Re-run src/04_ml_modeling.py.")
    else:
        importance_df = pd.concat(importance_frames, ignore_index=True)
        top_n = st.slider("Top features", min_value=5, max_value=25, value=15)

        for model in ["XGBOOST", "LIGHTGBM"]:
            model_df = importance_df[importance_df["Model"] == model].copy()
            if model_df.empty:
                continue
            model_df = model_df.sort_values("ImportancePct", ascending=False).head(top_n)
            model_df = model_df.sort_values("ImportancePct", ascending=True)

            fig = px.bar(
                model_df,
                x="ImportancePct",
                y="Feature",
                orientation="h",
                title=f"{model} — Top {len(model_df)} Features",
                labels={"ImportancePct": "Importance (%)"},
            )
            st.plotly_chart(fig, use_container_width=True)
