import pandas as pd
import numpy as np
import os

def create_features(df):
    print("Sorting data by Store and Date...")
    df.sort_values(['Store', 'Date'], inplace=True)
    
    print("Creating lag features...")
    # Lags for Sales (we use shift, grouped by Store)
    for lag in [1, 7, 14, 30]:
        df[f'Sales_Lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)
        
    print("Creating rolling window statistics...")
    # Rolling mean/std
    for window in [7, 14]:
        df[f'Sales_RollMean_{window}'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f'Sales_RollStd_{window}'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
        
    print("Encoding categorical variables...")
    # Convert StateHoliday to string to handle mixed types (0, '0', 'a', etc.)
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    
    # One-hot encode all categorical columns
    cat_cols = ['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Fill any remaining missing values with 0 (important for ML/DL models)
    df.fillna(0, inplace=True)
    
    return df

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs('../data/features', exist_ok=True)
    
    print("Loading cleaned dataset...")
    in_path = '../data/processed/train_clean.csv'
    df = pd.read_csv(in_path, parse_dates=['Date'], low_memory=False)
    
    df_features = create_features(df)
    
    out_path = '../data/features/train_features.csv'
    df_features.to_csv(out_path, index=False)
    print(f"Feature engineered dataset saved to {out_path}")
