# Real Estate Investment Advisor

An end-to-end machine learning project for analyzing Indian housing listings and estimating investment potential. It provides a reproducible EDA pipeline, classification and five-year price forecasting, experiment logging with MLflow, and an interactive Streamlit dashboard.

## Dataset and modeling assumptions

The listing file is attributed to the [India House Price Prediction Kaggle dataset](https://www.kaggle.com/datasets/ankushpanday1/india-house-price-prediction). NHB RESIDEX city exports are stored in `data/nhb/`; `python scripts/merge_nhb_hpi.py` creates `data/processed/nhb_hpi_combined.csv`, one row per city with quarter month-year columns and the `Current-Q` view. `python scripts/train.py` compares five-year HPI forecasting methods with rolling-origin cross-validation and a latest-period holdout, along with the listing proxy benchmarks. The app applies the selected city's projected HPI growth multiplier to a listing's asking price. NHB indices are city-level benchmarks, not property-linked resale transactions. See [`docs/data_provenance.md`](docs/data_provenance.md) for details.

- **Good Investment**: an engineered rule label. The annual appreciation proxy is estimated from an assumed 6% annual market appreciation baseline, with a modest age-related depreciation adjustment. Records whose proxy appreciation exceeds 4% are labeled good. This is not a verified investment return.
- **Five-year resale estimate**: asking price multiplied by the city's projected five-year NHB HPI ratio. This is a city-market estimate; the source has no matched property resale labels and the estimate cannot account for local property-specific effects. Cities without NHB coverage have no HPI estimate.

Change the assumptions in `src/config.py` before training if you have locally validated market assumptions. Exclude the ID and raw price-derived fields from classifier inputs to avoid target leakage. This tool is educational decision support, not financial advice.

The brief mentions crime and infrastructure analysis, which cannot be completed from the supplied columns. The EDA reports correlations that the dataset supports (including amenities and transit); no crime or infrastructure values are invented. If those sources become available, merge them by locality before using them.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_eda.py
python scripts/merge_nhb_hpi.py
python scripts/train.py
streamlit run app.py
```


The single `scripts/train.py` command fits and evaluates the city HPI forecast and runs the separate assumption-based listing proxy benchmarks. The HPI forecaster compares five transparent time-series methods using expanding-window five-year backtests, selects by cross-validation MAPE, scores the latest four forecast origins as a temporal holdout, and refits the selected method separately on each city's full history. It writes forecasts to `artifacts/hpi_forecast_by_city.csv` and evaluation tables to `reports/`. Run `notebooks/02_model_selection_and_evaluation.ipynb` to review the classifier's grouped-CV comparison.

The HPI estimate uses only the matched city index history and applies its forecast ratio to the asking price; it is not trained on property resale transactions. The investment classifier still predicts an engineered proxy label. The dashboard supports filtering by city, property type, BHK, area, and price and shows city HPI estimates only where NHB coverage exists. Neither output is evidence of property-specific investment return or resale accuracy.

## Project layout

```text
app.py                 Streamlit dashboard
notebooks/01_...       Cleaning and exploratory analysis
notebooks/02_...       Cross-validated model selection and evaluation
src/config.py          Paths, source schema, and documented proxy assumptions
src/data.py            Loading, cleaning, feature engineering, target construction
src/modeling.py        Listing proxy model construction
src/hpi_forecast.py    Auditable city HPI forecasting methods
scripts/run_eda.py     Reproducible analysis and charts
scripts/merge_nhb_hpi.py Merge city-wise NHB RESIDEX exports
scripts/train.py       Cross-validate, compare, select, and evaluate models
reports/               Generated EDA report and charts
artifacts/             Trained models and evaluation metrics
docs/                  Methodology, findings, use cases, and video script
```

## Outputs

`scripts/train.py` writes per-city HPI forecasts and a model-selection JSON under `artifacts/`, plus rolling-origin CV, holdout predictions, summary metrics, and the listing proxy benchmark results under `reports/` and `artifacts/`. The proxy-classifier and regression scores do not measure actual investment or resale accuracy. MLflow logs the proxy benchmark runs locally under `mlruns/`. The dashboard uses the HPI forecast for its five-year city-market estimate and labels investment classification as a proxy.

## Independent project assessment

See [the rubric-based assessment](docs/project_assessment.md) for category scores and remaining validation work. The revised implementation score is 3.6/5. Business validity remains limited by the absence of observed investment or resale outcomes in the supplied data.
