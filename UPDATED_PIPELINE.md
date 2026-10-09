# Updated Forecasting Benchmark

The pipeline now follows the approved minor-project proposal while fixing the evaluation inconsistencies in the previous implementation.

## What changed

- ARIMA/Prophet remain **per-store classical baselines** on the deterministic 20-store stratified benchmark sample, as allowed by the proposal's compute limitation.
- XGBoost/LightGBM remain **pooled models across all stores**.
- LSTM remains an extension trained across all stores.
- All models use the same **calendar validation cutoff** (last 42 calendar days of the available training data) and exclude closed-store observations from scoring.
- Model result files now store explicit validation dates.
- XGBoost/LightGBM validation is now **recursive multi-step**, so future actual sales are not fed into lag/rolling features during the validation rollout.
- LSTM scalers are fit on training data only.
- LSTM early stopping uses an internal validation slice of the training period, leaving the final test window untouched.
- LSTM validation is recursive rather than using actual future sales in test-period lag features.
- Evaluation rejects stale/misaligned result files and records validation observation counts.
- The dashboard plots actual dates instead of generic Step 1...42 labels.
- The dashboard only offers models that actually exist for the selected store.
- Aggregate benchmarking uses the **common store population** for the primary five-model comparison, while also displaying model coverage.
- Fixed model colours are used so colours do not change when the sidebar selection changes.

## Re-run

From the project root:

```bash
python run_pipeline.py
```

Or run the four stages individually:

```bash
python src/03_classical_modeling.py
python src/04_ml_modeling.py
python src/05_dl_modeling.py
python src/06_evaluation.py
```

Then:

```bash
streamlit run dashboard/app.py
```

The existing JSON/CSV result files in this archive are from the previous implementation. The dashboard intentionally refuses to display them until the updated pipeline has been run, because the new result files contain explicit validation dates and use the corrected evaluation protocol.
