# Core remaining changes

These files implement only the remaining core proposal requirements:

1. `src/01_data_ingestion_eda.py`
   - EDA outputs for trend, day-of-week seasonality, monthly seasonality, promotion effect, holiday effect, store type, and competition distance.
   - data-quality summary and outlier diagnostic.
   - preserves the existing cleaned modeling dataset logic.

2. `src/04_ml_modeling.py`
   - saves normalized feature importance for XGBoost and LightGBM under `data/processed/`.
   - does not change the model architecture or validation protocol.

3. `dashboard/app.py`
   - store(s) selection
   - validation date-range selection
   - model comparison
   - forecast vs actual plots on aligned dates
   - selected-range RMSE/MAE/MAPE
   - feature-importance visualization

## Apply

Extract this ZIP into the project root and overwrite the three matching files.

Then run:

```powershell
python .\src\01_data_ingestion_eda.py
python .\src\04_ml_modeling.py
python .\src\06_evaluation.py
streamlit run .\dashboard\app.py
```

The LSTM results do not need to be retrained again. `06_evaluation.py` is rerun after the XGBoost/LightGBM stage so `metrics.csv` reflects the latest model outputs.
