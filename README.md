# Demand Forecasting + Anomaly Detection

End-to-end time series forecasting project on Rossmann Store Sales dataset (~1M rows, 1115 stores).

## Results

| Model | Test MAPE | Test RMSE | CV MAPE |
|-------|-----------|-----------|---------|
| LightGBM | **8.20%** | 453 | 9.09% |
| Prophet | 17.16% | 879 | 14.04% |
| SARIMA | 17.95% | 900 | N/A |

**Winner: LightGBM** - 2x better accuracy than classical models.

## Key Findings

1. LightGBM outperforms Prophet and SARIMA by 2x (MAPE 8% vs 17%)
2. Top predictive features: `Sales_lag_1`, `Week`, `Promo`
3. Promo increases sales ~37% on average
4. December peak: sales 60-80% above baseline
5. 14 anomalies detected (1.8%) - 8 on Mondays
6. High anomalies correlate with Promo + holiday proximity

## Dataset

[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) - 1M rows, 1115 German drugstores, 3 years daily sales (2013-2015).

> Data not included in repo. Download from Kaggle and place in `data/`.

## Tech Stack

- Python 3.11+
- `lightgbm` - gradient boosting
- `prophet` - Facebook Prophet
- `statsmodels` - SARIMA
- `plotly` - interactive visualizations
- `pandas`, `numpy`, `scikit-learn`

## Project Structure
demand-forecasting/
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb    # Lags, rolling stats, holidays
│   ├── 03_prophet_model.ipynb          # Prophet + walk-forward CV
│   ├── 04_arima_model.ipynb            # SARIMA
│   ├── 05_lightgbm_model.ipynb         # LightGBM + feature importance
│   └── 06_comparison_and_anomaly.ipynb # Model comparison + anomaly detection
├── data/                               # Not in repo - download from Kaggle
├── requirements.txt
└── .gitignore

## Setup

```bash
git clone https://github.com/nurbolsultanov/demand-forecasting.git
cd demand-forecasting
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Download data from Kaggle, place in `data/`, then run notebooks in order.