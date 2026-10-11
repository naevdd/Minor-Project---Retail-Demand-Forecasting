# Retail Demand Forecasting Using Classical, Machine Learning, and Deep Learning Approaches

A comparative retail sales forecasting project using the **Rossmann Store Sales dataset**. This project evaluates classical time-series models, pooled gradient-boosting models, and deep learning to investigate the trade-off between forecasting accuracy and model complexity.

## 1. Project Overview

Retail demand forecasting helps businesses plan inventory, promotions, and staffing by estimating future sales. Inaccurate forecasts can lead to excess inventory, increased holding costs, stockouts, and lost revenue.

This project develops an end-to-end forecasting pipeline using historical daily sales data from 1,115 Rossmann stores. It compares five forecasting models across three modeling paradigms:

- **Classical time-series forecasting:** ARIMA and Prophet
- **Machine learning:** XGBoost and LightGBM
- **Deep learning:** Long Short-Term Memory (LSTM)

The central objective is to understand whether increasing model complexity produces meaningful improvements in forecasting accuracy.

The project includes data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and an interactive Streamlit dashboard.

## 2. Objectives

The primary objectives are:

1. Explore historical retail sales to identify temporal patterns, seasonality, promotion effects, holiday effects, and differences between store types.
2. Develop a reusable feature-engineering pipeline using lagged sales, rolling statistics, calendar attributes, and store characteristics.
3. Establish classical forecasting baselines using per-store ARIMA and Prophet models.
4. Train pooled XGBoost and LightGBM models across all 1,115 stores to learn shared sales patterns.
5. Explore LSTM as a deep-learning approach for sequential demand forecasting.
6. Evaluate forecasting performance using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).
7. Compare model performance and complexity through an interactive dashboard.

## 3. Dataset

**Source:** [Rossmann Store Sales — Kaggle](https://www.kaggle.com/c/rossmann-store-sales)

The dataset contains historical daily sales records for 1,115 Rossmann stores in Germany, covering the period from 2013 to 2015.

### Dataset files

| File | Description |
|---|---|
| `train.csv` | Historical sales observations, including store ID, date, sales, customer count, promotions, opening status, and holiday indicators. |
| `store.csv` | Store metadata, including store type, assortment, and competition-related information. |
| `test.csv` | Original Kaggle competition test observations without the target sales labels. |

The project uses the training data and store metadata to construct its historical forecasting dataset. The original Kaggle test file is not used to calculate the reported validation metrics.

### Important data considerations

- Store attributes are merged with daily sales observations.
- Dates are converted to datetime format.
- Missing store metadata values are handled during preprocessing.
- Temporal and calendar features are extracted from the date.
- Store closures are excluded from the demand-forecasting evaluation.
- Zero actual sales are excluded from MAPE calculations because percentage error is undefined when the actual value is zero.

The original dataset is not included in this repository by default. Download the files from Kaggle and place them in the expected project directory.

## 4. Methodology

The project follows a sequential modeling workflow.

### Stage 1: Data ingestion and exploratory data analysis

The raw sales and store datasets are cleaned, merged, and examined to understand the characteristics of the data.

The EDA stage generates descriptive statistics and visualizations covering:

- Sales trends over time
- Day-of-week and monthly seasonality
- Promotion and holiday effects
- Average sales by store type
- Competition-distance patterns
- Missing values, duplicates, and potential outliers

These analyses provide context for feature engineering and model interpretation.

### Stage 2: Feature engineering

The cleaned dataset is transformed into a reusable feature dataset.

**Lag features**

- `Sales_Lag_1`
- `Sales_Lag_7`
- `Sales_Lag_14`
- `Sales_Lag_30`

These features represent historical sales at selected time intervals.

**Rolling-window features**

- Seven-day rolling mean and standard deviation
- Fourteen-day rolling mean and standard deviation

Rolling statistics are shifted so that they use historical information rather than the target day's sales.

**Additional features**

- Year, month, day, and day of week
- Promotion indicators
- Holiday indicators
- Store identifiers and store-level metadata
- Encoded categorical store attributes

The resulting feature dataset is consumed by the downstream modeling stages.

### Stage 3: Classical time-series forecasting

Two classical forecasting approaches are implemented.

**ARIMA — AutoRegressive Integrated Moving Average**

ARIMA models temporal dependence through autoregressive, differencing, and moving-average components. In this implementation, it serves as a per-store statistical baseline using historical sales.

**Prophet**

Prophet models trend and seasonality in time series. It provides a second classical benchmark with a different approach to representing temporal patterns.

Due to computational constraints, ARIMA and Prophet are fitted to a stratified sample of 20 representative stores. Both classical models use the same benchmark store sample.

### Stage 4: Machine-learning forecasting

XGBoost and LightGBM are trained as pooled models using observations from all 1,115 stores.

Rather than fitting an independent tree model for each store, the pooled models learn shared relationships across the retail dataset.

The engineered features allow the models to use historical sales patterns, calendar effects, promotions, and store characteristics.

During the final validation forecast, predictions are generated recursively: each predicted sales value is incorporated into the history used to construct later lag and rolling features. This avoids using future actual sales as model inputs during the forecast rollout.

Feature-importance scores are generated for both gradient-boosting models to help interpret which input variables contribute to their predictions.

### Stage 5: Deep-learning forecasting

An LSTM network is implemented using PyTorch.

LSTM is a recurrent neural-network architecture designed to learn temporal dependencies from sequential data. The model uses a sequence length of 30 observations and incorporates engineered features.

The implementation includes:

- Sequence construction
- Training-only scaling
- An internal validation period for early stopping
- GPU acceleration when CUDA is available
- Recursive multi-step forecasting
- Saved validation predictions and actual values

LSTM is treated as the deep-learning extension of the project. Temporal Fusion Transformer (TFT) remains an optional stretch goal rather than a completed core model.

## 5. Evaluation Protocol

The project uses two complementary evaluation designs: a final fixed holdout for the primary benchmark and a three-fold expanding-window rolling-origin backtest for robustness. Both preserve chronological ordering; observations are not randomly shuffled across time.

### Fixed-holdout benchmark

| Parameter | Implementation |
|---|---|
| Training cutoff | 19 June 2015 |
| Validation period | 20 June – 31 July 2015 |
| Forecast horizon | Up to 42 calendar days |
| Evaluation rows | Open-store observations in the validation period |
| ARIMA and Prophet coverage | 20 stratified stores |
| XGBoost and LightGBM coverage | 1,115 stores each |
| LSTM coverage | 1,115 stores |

The number of validation observations varies by store because closed-store days are excluded from the demand-forecasting evaluation. The direct five-model comparison uses the 20 stores for which all five models have forecasts, ensuring that the models are compared on the same stores and dates.

### Rolling-origin validation

A separate three-fold expanding-window backtest evaluates all five models on the same 20 representative stores. Each fold uses a 14-calendar-day validation horizon and only information available before its forecast origin. The folds are:

| Fold | Training cutoff | Validation end |
|---|---|---|
| 1 | 10 April 2015 | 24 April 2015 |
| 2 | 24 April 2015 | 8 May 2015 |
| 3 | 8 May 2015 | 22 May 2015 |

The folds are before the final fixed-holdout period. The generated `rolling_origin_results.csv` contains 3,595 prediction rows in total, equal to 719 forecast observations per model across the three folds and 20 stores. The rolling-origin benchmark complements rather than replaces the fixed-holdout results.

### Hyperparameter selection

The implementation uses bounded candidate searches for ARIMA orders and Prophet configurations, and for the XGBoost and LightGBM model parameters. Rolling-origin tuning uses only each fold's historical training portion and an internal chronological validation segment; the fold's evaluation observations are not used to select that fold's parameters. The selected settings and candidate scores are recorded in the tuning-result CSV files.

### Evaluation metrics

**Root Mean Squared Error (RMSE)** is the square root of mean squared prediction error and penalizes larger errors more strongly.

**Mean Absolute Error (MAE)** is the mean absolute difference between actual and predicted sales.

**Mean Absolute Percentage Error (MAPE)** expresses absolute error relative to actual sales as a percentage. Observations with zero actual sales are excluded from MAPE because percentage error is undefined at zero. RMSE and MAE are reported alongside MAPE so that the evaluation is not dependent on a percentage metric alone.

Lower values are better for all three metrics.

### Statistical comparisons

The evaluation script performs paired Wilcoxon signed-rank tests on store-level MAE averaged over the rolling-origin folds. Ten pairwise model comparisons are made and the Holm correction is applied to control for multiple testing. These tests complement the descriptive metric rankings; statistical significance is not inferred from point-estimate rankings alone.

### Prediction intervals

ARIMA and Prophet also save nominal 95% prediction intervals for the fixed-holdout forecasts. Their empirical coverage is reported with the results so that uncertainty estimates can be assessed, rather than assuming that nominal 95% intervals automatically achieve 95% coverage.

## 6. Results

### Fixed-holdout benchmark: common 20-store sample

The following table reports the saved results from the fixed holdout, restricted to the 20 stores for which all five models are available.

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| **XGBoost** | **1,130.03** | **910.50** | **13.45%** |
| LightGBM | 1,180.56 | 951.30 | 13.91% |
| LSTM | 1,190.19 | 943.51 | 14.18% |
| Prophet | 1,493.70 | 1,243.34 | 17.68% |
| ARIMA | 1,924.47 | 1,603.13 | 24.53% |

XGBoost has the lowest RMSE, MAE, and MAPE in this benchmark. LightGBM and LSTM are close on the absolute-error metrics, while Prophet and ARIMA have higher errors. These results describe the selected stores and this training configuration; they do not establish that the same ranking will hold for every store or future time period.

### Rolling-origin validation

The combined rolling-origin metrics across the three folds are:

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| **XGBoost** | **1,345.06** | **987.92** | **13.51%** |
| LightGBM | 1,373.54 | 1,014.23 | 13.94% |
| Prophet | 1,630.39 | 1,192.26 | 16.22% |
| LSTM | 1,595.33 | 1,172.46 | 16.58% |
| ARIMA | 2,242.82 | 1,658.54 | 21.53% |

XGBoost also has the lowest errors in the rolling-origin results. This adds evidence that its fixed-holdout ranking is not unique to a single forecast window. However, three folds and 20 stores remain a limited sample of possible retail conditions.

### Statistical significance

At a 5% significance level after Holm correction, the paired tests find significant differences between ARIMA and each of the other four models. None of the comparisons among XGBoost, LightGBM, LSTM, and Prophet is significant after correction. Therefore, the results support XGBoost as the best *point-estimate performer* in these experiments, but they do not demonstrate that it is statistically superior to every other non-ARIMA model.

The exact raw and Holm-adjusted p-values, along with the paired-store counts, are available in `data/processed/significance_tests.csv`.

### Prediction-interval coverage

Empirical coverage of the nominal 95% fixed-holdout intervals, calculated over the saved ARIMA and Prophet forecast points, is:

| Model | Empirical coverage | Mean interval width |
|---|---:|---:|
| ARIMA | 97.7% | Approximately 11,354 sales units |
| Prophet | 77.6% | Approximately 3,732 sales units |

ARIMA's intervals are wide and cover slightly more observations than the nominal level; Prophet's intervals cover fewer observations than expected for a well-calibrated 95% interval. These results do not invalidate the point-forecast metrics, but they show that uncertainty estimates need separate assessment and may require calibration.

## 7. Accuracy–Complexity Trade-off

The accuracy results show that model complexity alone does not guarantee better forecasts. XGBoost has the lowest point-estimate errors in both reported evaluations, while LightGBM and LSTM remain close on the fixed-holdout absolute-error metrics. The statistical tests do not establish significant differences among those non-ARIMA models after correcting for multiple comparisons.

### Recorded runtime

The latest run's aggregate runtime report is:

| Model | Workload recorded | Fit time (s) | Forecast time (s) | Tuning time (s) | Total recorded time (s) |
|---|---|---:|---:|---:|---:|
| ARIMA | 20 store models | 2.79 | 0.03 | 6.42 | 9.24 |
| Prophet | 20 store models | 4.31 | 0.74 | 14.88 | 19.93 |
| LightGBM | 1 pooled model | 2.61 | 180.24 | 18.54 | 201.39 |
| XGBoost | 1 pooled model | 6.65 | 300.24 | 23.19 | 330.08 |
| LSTM | 1,115 per-store models | 1,778.09 | 130.05 | 0.00 | 1,908.14 |

These are measurements from the recorded run environment, not hardware-independent benchmark figures. The workloads differ: ARIMA and Prophet are each fitted for 20 stores, the tree models are pooled across the dataset, and LSTM trains a separate model for each of 1,115 stores. Total times include the recorded fitting, forecasting, and tuning components. Memory consumption was not measured, so the accuracy–complexity comparison is based on forecast errors and elapsed time rather than a complete resource-utilization study.

### Overall interpretation

The experiments favour XGBoost on the reported point-forecast metrics, while the runtime report shows that its recursive forecast stage took longer than LightGBM's in this run. LSTM required the greatest total recorded time because 1,115 separate networks were fitted. Those observations describe this implementation and hardware; they should not be generalized as universal rankings of the algorithms.

The intervals, significance tests, and rolling-origin results show why a model should be assessed through several complementary criteria rather than a single error metric or an assumption that a more complex model will perform better.

## 8. Project Structure

```text
Retail_Demand_Forecasting/
├── .gitignore
├── README.md
├── requirements.txt
├── run_pipeline.py
├── dashboard/
│   └── app.py
├── src/
│   ├── 01_data_ingestion_eda.py
│   ├── 02_feature_engineering.py
│   ├── 03_classical_modeling.py
│   ├── 04_ml_modeling.py
│   ├── 05_dl_modeling.py
│   ├── 06_evaluation.py
│   ├── 07_rolling_origin_validation.py
│   └── modeling_common.py
└── data/
    ├── raw/                         # downloaded locally; not included
    ├── features/
    │   ├── classical_results.json
    │   ├── classical_run_metadata.json
    │   ├── dl_results.json
    │   ├── dl_run_metadata.json
    │   ├── ml_results.json
    │   └── ml_run_metadata.json
    └── processed/
        ├── metrics.csv
        ├── common_benchmark_metrics.csv
        ├── rolling_origin_results.csv
        ├── rolling_origin_metrics.csv
        ├── significance_tests.csv
        ├── runtime_summary.csv
        ├── tuning-result CSV files
        ├── feature-importance CSV files
        └── eda/
```

`data/raw/`, the large intermediate training tables, and the `models/` checkpoint directory are not included in the source repository. The pipeline creates the intermediate tables and model checkpoints locally. `models/` is ignored by Git because the trained artifacts can be large; the saved forecast JSON and evaluation reports are included so the dashboard can display the completed run without retraining.

### Key directories

- **`src/`** contains ingestion/EDA, feature engineering, classical modeling, pooled machine learning, LSTM training, rolling-origin validation, and final evaluation.
- **`dashboard/`** contains the Streamlit application.
- **`data/raw/`** is where the two required Rossmann input CSVs are placed locally.
- **`data/features/`** contains saved forecast results and model-run manifests.
- **`data/processed/`** contains EDA outputs, metrics, tuning results, feature importance, rolling-origin predictions and summaries, statistical tests, and runtime reports.
- **`models/`** is generated locally when modeling runs and contains trained checkpoints; it is deliberately not tracked by Git.

The saved forecast JSON files are used by the dashboard to display predictions without retraining models on every page interaction.

## 9. Installation and Setup

### Prerequisites

- Python version compatible with the packages in `requirements.txt`
- Git
- The Rossmann Store Sales training and store metadata CSVs
- A GPU is optional. CPU execution is supported, although LSTM training can take much longer.

### Clone the repository

```powershell
git clone https://github.com/naevdd/Minor-Project---Retail-Demand-Forecasting.git
cd Minor-Project---Retail-Demand-Forecasting
```

### Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install or replace PyTorch with a build compatible with your system using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Then verify the installation:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

A `True` result from `torch.cuda.is_available()` indicates that PyTorch can access CUDA. GPU acceleration is optional; if CUDA is unavailable, the LSTM stage runs on CPU.

### Add the dataset

Download the Rossmann Store Sales training dataset from [Kaggle](https://www.kaggle.com/c/rossmann-store-sales) and place these files at the following paths:

```text
data/raw/train.csv
data/raw/store.csv
```

These are the two files required by the pipeline. The original competition `test.csv` is not used to calculate the validation metrics.

## 10. Running the Project

Run commands from the project root. The recommended full run is:

```powershell
python .\run_pipeline.py
```

The runner executes the stages in this order:

1. `src/01_data_ingestion_eda.py` — cleaning, data-quality report, and EDA plots.
2. `src/02_feature_engineering.py` — lag, rolling, calendar, and categorical features.
3. `src/03_classical_modeling.py` — ARIMA and Prophet for the 20-store benchmark.
4. `src/04_ml_modeling.py` — pooled XGBoost and LightGBM models.
5. `src/05_dl_modeling.py` — LSTM models for individual stores.
6. `src/07_rolling_origin_validation.py` — three-fold expanding-window backtest.
7. `src/06_evaluation.py` — fixed-holdout metrics, rolling-origin metrics, statistical comparisons, and runtime summaries.

The full pipeline can take substantial time, particularly when training 1,115 LSTM models and running the rolling-origin backtest. The run saves forecasts and reports under `data/features/` and `data/processed/`, with trained checkpoints under the locally generated `models/` directory.

For the final reported results, allow all stages to finish successfully. If you use `--skip-rolling-validation` for a diagnostic run, do not treat previously existing rolling-origin reports as evidence that the skipped validation stage ran in that execution.

### Launch the dashboard

After the outputs have been generated, run:

```powershell
streamlit run .\dashboard\app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. The dashboard uses the saved forecast and evaluation files; it does not need to retrain the models when users browse the results.

## 11. Interactive Dashboard

The dashboard provides four main tabs.

### Forecast Explorer

Allows users to select stores, choose available models, specify a validation date range, and compare predicted sales against actual sales. When ARIMA or Prophet are selected, their saved lower and upper prediction-interval bounds are also shown where available.

For a store included in the classical benchmark sample, the available comparison can include all five models: ARIMA, Prophet, XGBoost, LightGBM, and LSTM. ARIMA and Prophet are only available for the 20 representative benchmark stores; they are not fitted for every store. The dashboard aligns selected-model forecasts to common dates for a fair per-store display.

### Aggregate Benchmarks

Displays aggregate RMSE, MAE, and MAPE comparisons. The primary five-model benchmark uses the common sample of 20 stores. A separate model-coverage section shows the number of stores evaluated by each model.

### Feature Importance

Displays feature-importance rankings for XGBoost and LightGBM. These rankings help identify which engineered inputs are most influential within the fitted tree models. Feature importance describes model reliance, not necessarily causal relationships; a highly ranked feature should not automatically be interpreted as the cause of increased or decreased sales.

### Validation & Runtime

Displays the three-fold rolling-origin metrics, paired Wilcoxon comparisons with Holm-adjusted p-values, and runtime summaries. These reports let users examine performance across forecast origins and compare measured elapsed time alongside the accuracy metrics. Runtime comparisons should be interpreted in light of the different workloads used by the classical, pooled tree, and per-store LSTM models.

## 12. Generated Outputs

| Output | Purpose |
|---|---|
| `data/processed/train_clean.csv` | Cleaned, merged historical data; generated locally. |
| `data/features/train_features.csv` | Engineered training table; generated locally and excluded from Git because of its size. |
| `data/features/classical_results.json` | ARIMA and Prophet predictions, actuals, and lower/upper interval bounds for benchmark stores. |
| `data/features/ml_results.json` | XGBoost and LightGBM forecasts, dates, and actual values. |
| `data/features/dl_results.json` | LSTM forecasts, dates, actual values, training information, and artifact references. |
| `data/features/*_run_metadata.json` | Configuration/split manifests used to identify model-output versions. |
| `data/processed/metrics.csv` | Per-store fixed-holdout metrics. |
| `data/processed/common_benchmark_metrics.csv` | Five-model results on the same 20-store sample. |
| `data/processed/rolling_origin_results.csv` | Per-store predictions from the three expanding-window folds. |
| `data/processed/rolling_origin_metrics.csv` | Per-fold and combined rolling-origin RMSE, MAE, and MAPE. |
| `data/processed/significance_tests.csv` | Paired Wilcoxon tests and Holm-adjusted p-values. |
| `data/processed/runtime_summary.csv` | Aggregate fitting, forecasting, tuning, and total recorded times. |
| `data/processed/classical_tuning_results.csv`, `ml_tuning_results.csv` | Candidate configurations and tuning scores. |
| `data/processed/classical_runtime.csv`, `ml_runtime.csv`, `dl_runtime.csv` | Model-level runtime measurements. |
| `data/processed/accuracy_complexity_summary.csv` | Common-store accuracy joined to runtime measurements. |
| `data/processed/xgboost_feature_importance.csv`, `lightgbm_feature_importance.csv` | Feature-importance tables for the pooled tree models. |
| `data/processed/eda/` | EDA plots, data-quality statistics, and missing-value summaries. |

The forecast JSONs and evaluation reports are included in the repository as generated results. The raw data, large intermediate feature table, and trained model checkpoints are not included. Re-running the pipeline with the dataset regenerates those local artifacts.

## 13. Limitations

- ARIMA and Prophet are evaluated on 20 stratified stores because fitting them for all 1,115 stores is computationally expensive. Therefore, the fair five-model benchmark covers this common 20-store subset, not all stores.
- The data covers 2013–2015 and one retail chain. Performance on newer data or other retailers has not been established.
- XGBoost has the lowest point-estimate errors in the saved fixed-holdout and rolling-origin reports, but the paired significance tests do not establish statistically significant differences among XGBoost, LightGBM, LSTM, and Prophet after Holm correction.
- The nominal 95% prediction intervals are not perfectly calibrated: empirical coverage is approximately 97.7% for ARIMA and 77.6% for Prophet in the fixed-holdout predictions. ARIMA's intervals are also wide.
- Runtime results reflect different workloads and the machine used for the recorded run. Memory consumption was not measured, so the runtime table is not a standardized compute benchmark.
- MAPE excludes zero actual-sales observations, and should be interpreted alongside RMSE and MAE.
- Recursive forecasts can amplify errors over a multi-day horizon. The benchmark metrics summarize the evaluation windows and do not guarantee equally stable predictions for every individual store.
- Model-checkpoint files are generated locally and ignored by Git. The checked-in forecast files support dashboard use, but loading a trained checkpoint requires rerunning the training pipeline or separately preserving the local `models/` directory.
- Temporal Fusion Transformer (TFT) remains an optional stretch goal and is not part of the five-model results.

## 14. Future Work

Potential extensions that build on the current implementation include:

- Calibrating forecast intervals and evaluating coverage across stores and horizons.
- Testing on newer periods or an additional retail dataset to assess external validity.
- Measuring memory use and conducting a more standardized accuracy–runtime comparison.
- Expanding the bounded tuning search and checking sensitivity to forecast horizon and store characteristics.
- Investigating stores and time windows where recursive forecast errors grow sharply.
- Connecting predictions to inventory/replenishment decisions and measuring their operational impact.
- Exploring Temporal Fusion Transformer (TFT) if sufficient time and resources are available; it is a stretch goal, not a required component of the current project.

These are extensions. The fixed-holdout benchmark, rolling-origin validation, paired significance tests, runtime reports, and Streamlit dashboard described above are implemented in the current pipeline.

## 15. Conclusion

This project developed and evaluated five approaches for retail sales forecasting using the Rossmann Store Sales dataset: ARIMA, Prophet, XGBoost, LightGBM, and LSTM. The workflow includes exploratory data analysis, feature engineering, bounded hyperparameter searches, a fixed chronological holdout, three-fold expanding-window validation, paired statistical comparisons, runtime measurement, and an interactive Streamlit dashboard.

On the common 20-store fixed-holdout benchmark, XGBoost achieved the lowest RMSE (1,130.03), MAE (910.50), and MAPE (13.45%). XGBoost also produced the lowest aggregate errors in rolling-origin validation. LightGBM and LSTM remained competitive on the fixed-holdout absolute-error metrics. The significance tests, however, did not establish significant differences among the non-ARIMA models after Holm correction.

The prediction-interval analysis also showed that nominal 95% intervals do not necessarily deliver 95% empirical coverage. Together, the results demonstrate the value of evaluating point accuracy, uncertainty, statistical evidence, and recorded runtime rather than assuming that model complexity alone determines forecasting quality. The findings are specific to the historical data, chosen store sample, evaluation design, and recorded runtime environment; further validation on newer or external data would be needed before making broader claims.

## 16. References

1. Rossmann Store Sales Dataset. Kaggle. https://www.kaggle.com/c/rossmann-store-sales
2. Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. *The American Statistician*, 72(1), 37–45.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.
4. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*.
5. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
6. Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*. OTexts. https://otexts.com/fpp3/
