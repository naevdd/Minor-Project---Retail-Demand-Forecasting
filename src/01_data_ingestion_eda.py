import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EDA_DIR = PROCESSED_DIR / "eda"


def load_data():
    print("Loading datasets...")
    train = pd.read_csv(RAW_DIR / "train.csv", low_memory=False)
    store = pd.read_csv(RAW_DIR / "store.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")

    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])
    return train, store, test


def clean_and_merge(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning and merging data...")
    df = train.merge(store, on="Store", how="left")

    # Preserve the original project cleaning decisions.
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].median()
    )
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(1)
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(1900)
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)
    df["PromoInterval"] = df["PromoInterval"].fillna("")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1
    return df


def _save(fig, filename: str) -> None:
    path = EDA_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.relative_to(PROJECT_ROOT)}")


def run_eda(df: pd.DataFrame) -> None:
    print("Running EDA...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the modeling input unchanged apart from the existing cleaning steps.
    clean_path = PROCESSED_DIR / "train_clean.csv"
    df.to_csv(clean_path, index=False)
    print(f"Cleaned dataset saved to {clean_path}")

    open_df = df[df["Open"] == 1].copy()
    open_df["IsHoliday"] = (
        open_df["StateHoliday"].astype(str).ne("0")
        | open_df["SchoolHoliday"].eq(1)
    )

    # 1. Data-quality summary, including an outlier diagnostic rather than
    # deleting potentially genuine retail demand spikes.
    numeric_missing = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())
    open_sales = open_df["Sales"].astype(float)
    q1, q3 = open_sales.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = max(0.0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    outlier_count = int(((open_sales < lower) | (open_sales > upper)).sum())

    summary = pd.DataFrame(
        [
            {"Metric": "Rows", "Value": len(df)},
            {"Metric": "Stores", "Value": df["Store"].nunique()},
            {"Metric": "Date min", "Value": df["Date"].min().date()},
            {"Metric": "Date max", "Value": df["Date"].max().date()},
            {"Metric": "Duplicate rows", "Value": duplicate_rows},
            {"Metric": "Closed-day rows", "Value": int((df["Open"] == 0).sum())},
            {"Metric": "Open-day zero-sales rows", "Value": int((open_df["Sales"] == 0).sum())},
            {"Metric": "Sales IQR outlier rows (open days)", "Value": outlier_count},
            {"Metric": "Sales IQR lower bound", "Value": lower},
            {"Metric": "Sales IQR upper bound", "Value": upper},
            {"Metric": "Missing cells", "Value": int(numeric_missing.sum())},
        ]
    )
    summary.to_csv(EDA_DIR / "data_quality_summary.csv", index=False)

    missing = numeric_missing[numeric_missing > 0].sort_values(ascending=False)
    missing.to_csv(EDA_DIR / "missing_values_by_column.csv", header=["MissingCount"])

    # Outliers are reported rather than automatically removed because extreme
    # sales can be genuine promotion/seasonality effects in retail data.
    print(f"  Detected {outlier_count:,} sales IQR outlier rows; retained them for modeling.")

    # 2. Overall sales trend.
    daily = open_df.groupby("Date", as_index=False)["Sales"].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["Date"], daily["Sales"])
    ax.set_title("Average Daily Sales Across Open Stores")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Sales")
    _save(fig, "sales_trend.png")

    # 3. Day-of-week seasonality.
    dow = open_df.groupby("DayOfWeek", as_index=False)["Sales"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(dow["DayOfWeek"].astype(str), dow["Sales"])
    ax.set_title("Average Sales by Day of Week")
    ax.set_xlabel("Day of Week (1 = Monday)")
    ax.set_ylabel("Average Sales")
    _save(fig, "day_of_week_seasonality.png")

    # 4. Monthly seasonality.
    month = open_df.groupby("Month", as_index=False)["Sales"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(month["Month"], month["Sales"], marker="o")
    ax.set_title("Average Sales by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Sales")
    _save(fig, "monthly_seasonality.png")

    # 5. Promotion effect.
    promo = open_df.groupby("Promo", as_index=False)["Sales"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(promo["Promo"].astype(str), promo["Sales"])
    ax.set_title("Average Sales: Promotion vs No Promotion")
    ax.set_xlabel("Promotion (0/1)")
    ax.set_ylabel("Average Sales")
    _save(fig, "promotion_effect.png")

    # 6. Holiday effect.
    holiday = open_df.groupby("IsHoliday", as_index=False)["Sales"].mean()
    holiday["Holiday"] = holiday["IsHoliday"].map({False: "No Holiday", True: "Holiday"})
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(holiday["Holiday"], holiday["Sales"])
    ax.set_title("Average Sales: Holiday vs Non-Holiday")
    ax.set_xlabel("")
    ax.set_ylabel("Average Sales")
    _save(fig, "holiday_effect.png")

    # 7. Store-type demand patterns.
    store_type = open_df.groupby("StoreType", as_index=False)["Sales"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(store_type["StoreType"], store_type["Sales"])
    ax.set_title("Average Sales by Store Type")
    ax.set_xlabel("Store Type")
    ax.set_ylabel("Average Sales")
    _save(fig, "store_type_effect.png")

    # 8. Competition distance effect: robust quantile bins avoid a misleading
    # raw scatter of more than one million observations.
    comp = open_df[["CompetitionDistance", "Sales"]].dropna().copy()
    if comp["CompetitionDistance"].nunique() >= 4:
        comp["DistanceBin"] = pd.qcut(
            comp["CompetitionDistance"], q=10, duplicates="drop"
        )
        comp_binned = comp.groupby("DistanceBin", observed=True)["Sales"].mean().reset_index()
        comp_binned["DistanceBin"] = comp_binned["DistanceBin"].astype(str)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(comp_binned["DistanceBin"], comp_binned["Sales"])
        ax.set_title("Average Sales by Competition-Distance Decile")
        ax.set_xlabel("Competition Distance Decile")
        ax.set_ylabel("Average Sales")
        ax.tick_params(axis="x", rotation=45)
        _save(fig, "competition_distance_effect.png")

    # Export the summary statistics used in the console and report.
    sales_by_dow = open_df.groupby("DayOfWeek")["Sales"].mean()
    print("\nAverage Sales by Day of Week (Open Stores):")
    print(sales_by_dow.round(2).to_string())
    print(f"\nEDA complete. Outputs saved under {EDA_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    train, store, test = load_data()
    df_clean = clean_and_merge(train, store)
    run_eda(df_clean)
