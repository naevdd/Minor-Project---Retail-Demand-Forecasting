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

All five models are evaluated on a shared time-based validation cutoff.

| Parameter | Current implementation |
|---|---|
| Training cutoff | 19 June 2015 |
| Validation calendar period | 20 June – 31 July 2015 |
| Forecast horizon | Up to 42 calendar days |
| Evaluation observations | Open-store validation observations |
| Classical-model coverage | 20 representative stores |
| XGBoost coverage | 1,115 stores |
| LightGBM coverage | 1,115 stores |
| LSTM coverage | 1,115 stores |

The number of evaluated observations can vary between stores because the evaluation excludes closed-store days.

### Evaluation metrics

**Root Mean Squared Error (RMSE)**

RMSE measures the square root of the average squared prediction error. It penalizes large errors more heavily than MAE.

**Mean Absolute Error (MAE)**

MAE measures the average absolute difference between actual and predicted sales. Lower MAE indicates smaller absolute errors.

**Mean Absolute Percentage Error (MAPE)**

MAPE measures absolute errors relative to actual sales, expressed as a percentage. In this project, observations with zero actual sales are excluded from the MAPE calculation.

Lower values indicate better performance for all three metrics.

### Common-store benchmark

ARIMA and Prophet are evaluated on 20 representative stores, while the pooled models cover all 1,115 stores.

To make a direct five-model comparison, the aggregate benchmark restricts the comparison to the 20 stores for which all five models have evaluation results.

This common-store benchmark is the primary basis for the accuracy–complexity discussion below. It should not be interpreted as a five-model comparison across all 1,115 stores.

**Validation note:** The current reported benchmark uses a fixed time-based holdout. It should not be described as a completed rolling-origin cross-validation study unless multiple rolling forecast origins have also been evaluated.

## 6. Results

The following results are from the current evaluation run over the 20 common benchmark stores.

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| ARIMA | 1,942.24 | 1,619.83 | 25.10% |
| Prophet | 1,481.55 | 1,237.08 | 17.55% |
| XGBoost | 1,477.06 | 1,202.84 | 18.84% |
| LightGBM | **1,444.70** | **1,163.33** | 17.91% |
| LSTM | 1,856.78 | 1,375.84 | 21.91% |

Lower values indicate better predictive accuracy.

These results are descriptive and specific to the selected stores, training configuration, and validation period.

## 7. Accuracy–Complexity Trade-off

The results show that a more complex model does not necessarily produce the best forecast under every accuracy metric.

### Prophet: competitive percentage accuracy

Prophet achieves the lowest MAPE of 17.55%. Its MAPE is only 0.36 percentage points lower than LightGBM's 17.91%.

This demonstrates that a classical model can remain competitive with a more feature-intensive machine-learning approach in relative-error performance.

### LightGBM: lowest absolute errors

LightGBM achieves the lowest RMSE and MAE across the common benchmark stores.

Compared with ARIMA, LightGBM reduces RMSE by approximately 25.6% and MAE by approximately 28.2%. This suggests that pooled learning and engineered features can improve absolute-error performance over a univariate classical baseline.

### XGBoost: competitive but not the strongest booster

XGBoost performs reasonably closely to LightGBM, but its reported errors are slightly higher across all three metrics.

In this experiment, the additional modeling flexibility of XGBoost does not produce an accuracy advantage over LightGBM.

### LSTM: complexity without a demonstrated accuracy advantage

LSTM has higher aggregate errors than Prophet and LightGBM in the current benchmark. Its greater training and computational requirements have therefore not translated into better reported accuracy under the current configuration.

This does not establish that LSTM is inherently inferior. The result may depend on training history, architecture, hyperparameters, and recursive forecast behaviour.

### Overall interpretation

The current evidence favours Prophet when relative percentage accuracy is the primary criterion and LightGBM when minimizing absolute forecast errors is more important.

The results support the project's central research question: the benefits of additional model complexity should be established empirically rather than assumed.

The reported metrics do not, by themselves, establish statistical significance or quantify accuracy gained per unit of runtime or memory. Those conclusions would require additional testing and direct measurement of computational costs.

## 8. Project Structure

```text
Retail_Demand_Forecasting/
│
├── README.md
├── run_pipeline.py
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── 01_data_ingestion_eda.py
│   ├── 02_feature_engineering.py
│   ├── 03_classical_modeling.py
│   ├── 04_ml_modeling.py
│   ├── 05_dl_modeling.py
│   └── 06_evaluation.py
│
└── data/
    ├── raw/
    │   ├── train.csv
    │   ├── store.csv
    │   └── test.csv
    │
    ├── processed/
    │   ├── train_clean.csv
    │   ├── metrics.csv
    │   ├── common_benchmark_stores.json
    │   ├── xgboost_feature_importance.csv
    │   ├── lightgbm_feature_importance.csv
    │   └── eda/
    │
    └── features/
        ├── train_features.csv
        ├── classical_results.json
        ├── ml_results.json
        └── dl_results.json
```

Some output files are generated only after their corresponding pipeline stages have run.

### Key directories

**`src/`**

Contains the scripts for ingestion and EDA, feature engineering, classical modeling, machine learning, deep learning, and evaluation.

**`dashboard/`**

Contains the Streamlit application used to explore model forecasts and benchmark results.

**`data/raw/`**

Contains the original input CSV files downloaded from Kaggle.

**`data/processed/`**

Contains cleaned data, EDA outputs, evaluation metrics, common benchmark store information, and feature-importance tables.

**`data/features/`**

Contains the engineered feature dataset and saved forecast results for the classical, ML, and DL models.

The saved JSON prediction files are used by the dashboard to display forecasts without retraining models on every page interaction.

## 9. Installation and Setup

### Prerequisites

- Python
- Git
- Access to the Rossmann Store Sales dataset
- A GPU is optional; CPU execution is supported, although LSTM training can take longer.

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with the HTTPS URL of your GitHub repository and its local folder name.

### Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Install dependencies

Install the core libraries:

```powershell
python -m pip install pandas numpy matplotlib seaborn scikit-learn statsmodels prophet xgboost lightgbm streamlit plotly
```

Install PyTorch separately using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose a build compatible with your operating system, Python version, and CUDA configuration.

To use an NVIDIA GPU, verify that the CUDA-enabled PyTorch build is installed:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

A `True` result from `torch.cuda.is_available()` indicates that PyTorch can access CUDA.

### Add the dataset

Download the required Rossmann CSV files from Kaggle and place them in:

```text
data/raw/train.csv
data/raw/store.csv
data/raw/test.csv
```

The scripts expect these filenames and locations.

## 10. Running the Project

Run commands from the project root unless noted otherwise.

### Step 1: Data cleaning and EDA

```powershell
python .\src\01_data_ingestion_eda.py
```

This generates the cleaned training dataset and EDA outputs.

### Step 2: Feature engineering

```powershell
python .\src\02_feature_engineering.py
```

This creates `data/features/train_features.csv`.

### Step 3: Run modeling and evaluation

```powershell
python .\run_pipeline.py
```

The pipeline runs the following stages in order:

1. `03_classical_modeling.py`
2. `04_ml_modeling.py`
3. `05_dl_modeling.py`
4. `06_evaluation.py`

The full run may take some time, especially when fitting the classical models and training LSTM on CPU.

Existing result files may be reused by the deep-learning stage. If the saved results were generated by an older incompatible implementation, regenerate them before relying on the evaluation output.

### Step 4: Launch the dashboard

```powershell
streamlit run .\dashboard\app.py
```

Streamlit will display a local URL in the terminal, typically `http://localhost:8501`.

Open that address in a browser.

## 11. Interactive Dashboard

The dashboard provides three main sections.

### Forecast Explorer

Allows users to select stores, choose available models, specify a validation date range, and compare predicted sales against actual sales.

For a store included in the classical benchmark sample, the available comparison can include all five models:

- ARIMA
- Prophet
- XGBoost
- LightGBM
- LSTM

ARIMA and Prophet are only available for the 20 representative benchmark stores. They are not fitted for every store.

The dashboard displays error metrics for the selected store/model results and aligns forecast points with their validation dates.

### Aggregate Benchmarks

Displays aggregate RMSE, MAE, and MAPE comparisons.

The primary five-model benchmark uses the common sample of 20 stores. A separate model-coverage section shows the number of stores evaluated by each model.

### Feature Importance

Displays feature-importance rankings for XGBoost and LightGBM. These rankings help identify which engineered inputs are most influential within the fitted tree models.

Feature importance describes model reliance, not necessarily causal relationships. A feature with high importance should not automatically be interpreted as the cause of increased or decreased sales.

## 12. Generated Outputs

| Output | Purpose |
|---|---|
| `train_clean.csv` | Cleaned and merged historical training data. |
| `train_features.csv` | Historical training data with engineered lag, rolling, calendar, and categorical features. |
| `classical_results.json` | Classical forecasts and actual validation values for the benchmark stores. |
| `ml_results.json` | XGBoost and LightGBM forecasts, dates, and actual values. |
| `dl_results.json` | LSTM forecasts, dates, actual values, and training information. |
| `metrics.csv` | Per-store model evaluation metrics. |
| `common_benchmark_stores.json` | IDs of stores available for the direct five-model comparison. |
| `xgboost_feature_importance.csv` | Ranked XGBoost feature-importance values. |
| `lightgbm_feature_importance.csv` | Ranked LightGBM feature-importance values. |
| `eda/` | EDA plots and data-quality summaries. |

The CSV metrics and JSON forecasts are generated artifacts. They can be regenerated by running the appropriate pipeline stages.

## 13. Limitations

- The classical models are evaluated on 20 representative stores because fitting ARIMA and Prophet for all 1,115 stores can be computationally expensive.
- The direct five-model aggregate comparison is therefore based on those 20 stores.
- The dataset covers 2013–2015 and represents a particular retail chain and geographic context.
- The current benchmark uses a fixed time-based holdout; broader rolling-origin validation would provide additional evidence about performance stability across forecast origins.
- MAPE excludes zero actual-sales observations, so it should be interpreted alongside RMSE and MAE.
- Recursive multi-step forecasting can amplify prediction errors. Individual-store forecast curves may therefore behave differently from the aggregate benchmark.
- Current results do not independently establish statistical significance between models.
- TFT remains a stretch goal and is not included in the reported five-model benchmark.
- Model performance may change with different hyperparameters, forecasting horizons, feature representations, or store populations.

## 14. Future Work

Potential extensions include:

- Rolling-origin cross-validation for more robust performance assessment.
- Statistical significance testing on paired forecast errors.
- Measuring model training time, inference time, and resource requirements.
- Further investigation of recursive forecast instability in individual stores.
- Hyperparameter optimization and systematic sensitivity analysis.
- Exploring Temporal Fusion Transformer if resources and time permit.

These extensions should be distinguished from functionality already implemented and evaluated.

## 15. Conclusion

This project provides a comparative retail demand forecasting pipeline spanning classical statistical forecasting, pooled machine learning, and deep learning.

The current common-store benchmark shows that Prophet achieves the lowest MAPE, LightGBM achieves the lowest RMSE and MAE, and the LSTM implementation does not outperform those approaches on the reported metrics.

These findings illustrate why forecasting models should be selected based on measured performance and computational considerations rather than assumed superiority from model complexity alone.

The resulting pipeline and interactive dashboard provide a foundation for understanding demand patterns, examining individual-store forecasts, and evaluating the accuracy–complexity trade-off in retail forecasting.

## 16. References

1. Rossmann Store Sales Dataset. Kaggle. https://www.kaggle.com/c/rossmann-store-sales
2. Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. *The American Statistician*, 72(1), 37–45.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.
4. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*.
5. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
6. Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*. OTexts. https://otexts.com/fpp3/
