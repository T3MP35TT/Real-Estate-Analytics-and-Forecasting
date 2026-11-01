# Real Estate Investment Advisor

**Predicting Property Profitability & Future Value**

An end-to-end real estate analytics prototype for exploring Indian housing listings, comparing property markets, and producing five-year city-level housing-index scenarios. The project combines data cleaning, exploratory data analysis (EDA), feature engineering, model benchmarking, NHB RESIDEX forecasting, MLflow experiment tracking, and an interactive Streamlit application.

**Live application:** [Real Estate Investment Advisor – Streamlit](https://real-estate-analytics-and-forecasting-kartikey-singh.streamlit.app/)

> **Important:** This is an exploratory decision-support prototype, not a validated investment recommendation or property-resale valuation system. The listing classifier uses an assumption-generated label, the regression benchmark uses a formula-generated target, and the five-year property estimate applies a city-level HPI forecast to an asking price. None of these is validated against matched property resale outcomes.

---

## 1. Project objectives

The project was designed to explore four questions:

1. How are asking prices, property characteristics, and listing patterns distributed across Indian cities and states?
2. How does a listing compare with similar properties in its city, property type, BHK configuration, and size band?
3. What does the selected city's historical housing-price index imply for a five-year market scenario?
4. How can model selection, evaluation, and experiment tracking be made reproducible?

The original project brief also asks for an investment classification and a five-year price estimate. The available data does not contain observed investment returns or matched five-year resale prices, so those targets are explicitly treated as proxies or scenarios.

## 2. Data sources

### Property listings

The project listing dataset is attributed to the [India House Price Prediction dataset on Kaggle](https://www.kaggle.com/datasets/ankushpanday1/india-house-price-prediction).

Expected project path:

```text
data/india_housing_prices.csv
```

It contains approximately 250,000 listing records and 23 source columns, including location, property type, BHK, area, asking price, property age, furnishing, floor information, nearby schools and hospitals, transport access, parking, security, amenities, facing, ownership, and availability.

These are **listing asking prices**, not verified registered transaction prices. The file does not provide matched resale outcomes, rental cash flows, or a reliable listing date for every record.

### NHB RESIDEX

Quarterly city-level NHB RESIDEX HPI exports are stored under:

```text
data/nhb/
```

The project merge script creates:

```text
data/processed/nhb_hpi_combined.csv
```

The merged modelling input has one row per city and quarter columns in month-year format. The workflow consistently selects the `Current-Q` view; alternate `Previous-Q` extracts are not averaged into it.

The current modelling history used in the recorded run covers 28 cities and 51 quarters through December 2025. This is a **city-level assessment-price index**, not property-level resale data. Refresh and reconcile the source series/base-year links before relying on a new operational forecast.

See [`docs/data_provenance.md`](docs/data_provenance.md).

## 3. Application features

The deployed Streamlit application contains these pages:

| Page | Main capabilities |
|---|---|
| Home | Workspace summary and navigation |
| Executive Overview | Filtered headline KPIs, market snapshot, executive insights |
| Market Insights | State/city/property filters, asking-price distributions, segment comparisons, historical HPI context |
| Property Estimator | Property input form, comparable asking-price analysis, city-HPI five-year scenario, experimental proxy-screening result |
| Listing Explorer | Detailed listing filters, record exploration, summary metrics, CSV export |
| Forecast Performance | HPI candidate comparison, rolling-origin CV, temporal holdout diagnostics, scenario errors and fallback information |
| Methodology & Data | Data inventory, source notes, assumptions, limitations and managed-data workspace |

The Property Estimator's comparable-market section uses same-city, same-property-type and same-BHK listings, preferring a ±25% area cohort when at least five comparables are available. The comparable median is a screening reference based on asking prices, not a formal valuation.

## 4. Methodology and target definitions

### Data preparation

The workflow validates the source schema, standardizes text and data types, checks duplicates and invalid values, recomputes price per square foot from price and area, engineers additional features, and records data-quality flags. Model preprocessing is fitted within scikit-learn pipelines so imputation, scaling, and categorical encoding are learned from training folds rather than the full dataset.

Selected data-quality observations from the supplied listing file:

- 250,000 source rows across 23 columns.
- No source-column missing values and no exact duplicate rows in the recorded input.
- 116,304 records have `Floor_No > Total_Floors`; the raw values are retained and the issue is flagged/masked in a derived clean field rather than silently rewritten.
- 2,534 reported price-per-square-foot values are nonpositive; the analysis recomputes the measure from asking price and area.
- Amenity strings are canonicalized to reduce formatting-only variations.

See [`docs/methodology.md`](docs/methodology.md) for details.

### Investment classification proxy

The constructed annual appreciation proxy is:

```text
Annual_Appreciation_Proxy
    = 6% - (Property_Age × 0.2 percentage points)
```

A record is labelled positive when the proxy exceeds 4%:

```text
Good_Investment = 1
    when Annual_Appreciation_Proxy > 4%
otherwise 0
```

The target is derived from assumptions and property age. It is **not an observed investment-success label**. Classification metrics measure how well the fitted classifier reconstructs this rule on held-out cities; they do not measure the probability of making a profit.

### Five-year listing regression proxy

The benchmark target is generated by compounding the asking price at the assumed 6% annual appreciation rate:

```text
Price_After_5Y_Proxy
    = Price_in_Lakhs × (1 + 0.06)^5
```

The target is therefore deterministic. Near-perfect regression metrics are expected and do not demonstrate five-year resale-price accuracy.

The deployed Property Estimator does **not** use the selected regression candidate as its five-year property-price engine. It applies the selected city's NHB HPI growth multiplier to the entered asking price to produce a city-market scenario.

### City HPI forecasting

The HPI workflow compares 21 candidate forecasting methods using 20-quarter (five-year) rolling-origin forecasts. It uses earlier origins for cross-validation/model selection and eight later origins for temporal holdout evaluation. The selected Theta method is refitted using each eligible city's full available history for the app.

The historical evaluation windows overlap, so their errors are correlated. The empirical scenario range is an indicative pooled historical-error range, not a calibrated property-level prediction interval.

## 5. Recorded model results

The following values are from the completed training run recorded on 1 November 2026. They are evaluation results for the project targets as defined above—not evidence of real-world investment or property resale accuracy.

### Classification — selected Logistic Regression

| Metric | City-holdout result |
|---|---:|
| Accuracy | 50.72% |
| Precision | 20.83% |
| Recall | 49.36% |
| F1 score | 29.30% |
| ROC AUC | 0.501 |

**Interpretation:** ROC AUC is effectively at chance level. The classifier should remain experimental and should not be used to rank properties or make investment decisions.

### Regression — selected Linear Regression benchmark

| Metric | City-holdout result |
|---|---:|
| MAE | 0.000733 lakh |
| RMSE | 0.000897 lakh |
| R² | 0.99999999998 |

The zero-error formula baseline reproduces the formula-generated target exactly. Consequently, the near-perfect regression score is target reconstruction, not validated resale prediction. The app uses the city-HPI scenario for its five-year estimate.

### NHB HPI — selected Theta

| Evaluation | Result |
|---|---:|
| Rolling-origin CV MAPE | 15.10% |
| Temporal holdout MAPE | 16.64% |
| Holdout MAE skill vs last-value baseline | 0.287 |
| Fallback rate | 0.00% |
| History | 28 cities · 51 quarters |
| Forecast horizon | 20 quarters (five years) |

**Interpretation:** Theta improved holdout MAE by approximately 28.7% relative to the last-value baseline. The holdout MAPE still indicates material forecast error. These metrics evaluate city HPI indices, not individual property resale transactions.

## 6. Model validation and experiment tracking

Listing model evaluation uses:

- A 20% city-group holdout, keeping test cities separate from training cities.
- Five-fold `GroupKFold` cross-validation grouped by city.
- Imputation, scaling, and one-hot encoding inside the model pipelines.
- Candidate-family comparisons and a bounded hyperparameter search.
- Holdout, segment, reference-baseline, and run-manifest outputs.

HPI model evaluation uses expanding-history, 20-quarter rolling-origin forecasts, candidate-method comparisons, cross-validation selection, and a later temporal holdout.

MLflow experiment tracking is stored in the project-root SQLite database:

```text
mlflow.db
```

The experiment name is `real-estate-investment-advisor`. Recorded runs include HPI forecast validation, selected classifier, and selected regressor, with parameters, metrics, and logged artifacts. MLflow is local experiment tracking; the SQLite database is not automatically hosted with the public Streamlit application.

To view the local experiment UI from the project root:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 7. Quick start (Windows PowerShell)

Run commands from the repository root.

```powershell
# Create the virtual environment
py -m venv .venv

# Install dependencies without requiring activation
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Generate EDA reports
.\.venv\Scripts\python.exe scripts\run_eda.py

# Merge the manually downloaded NHB files
.\.venv\Scripts\python.exe scripts\merge_nhb_hpi.py

# Train, compare, validate and log models
.\.venv\Scripts\python.exe scripts\train.py

# Launch the application
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If NHB files have already been merged and the generated outputs are present, you do not need to merge them again just to launch the app. Similarly, training is not required for every app launch when the required artifacts already exist.

For notebooks, open:

```text
notebooks/01_data_cleaning_and_eda_v50.ipynb
notebooks/02_model_selection_and_evaluation_v50.ipynb
```

Run notebook cells in order when you need to review the detailed analysis and model evaluation.

## 8. Project structure

```text
project-root/
├── app.py
├── pages/
│   ├── 01_Executive_Overview.py
│   ├── 02_Market_Insights.py
│   ├── 03_Property_Estimator.py
│   ├── 04_Listing_Explorer.py
│   ├── 05_Forecast_Performance.py
│   └── 06_Methodology_and_Data.py
├── notebooks/
│   ├── 01_data_cleaning_and_eda_v50.ipynb
│   └── 02_model_selection_and_evaluation_v50.ipynb
├── scripts/
│   ├── run_eda.py
│   ├── merge_nhb_hpi.py
│   ├── train.py
│   └── test_hpi_forecast.py
├── src/
│   ├── config.py
│   ├── data.py
│   ├── modeling.py
│   └── hpi_forecast.py
├── utils/
│   ├── helper.py
│   ├── sidebar.py
│   ├── theme.py
│   └── data_manager.py
├── data/
│   ├── india_housing_prices.csv
│   ├── nhb/
│   └── processed/
├── artifacts/
├── reports/
├── docs/
├── requirements.txt
└── .gitignore
```

Generated files inside `artifacts/`, `reports/`, and `data/processed/` depend on the local inputs and training run. Keep large source data, environment folders, credentials, and local-only databases out of Git unless you have a clear reason and permission to publish them.

## 9. Documentation

- [Methodology](docs/methodology.md) — preparation, feature engineering, target construction, validation design and app calculations.
- [Data provenance](docs/data_provenance.md) — source attribution, coverage, selected HPI view and data limitations.
- [Findings and use cases](docs/findings_and_use_cases.md) — analytical results, supported use cases and interpretation guardrails.
- [Project assessment](docs/project_assessment.md) — evidence-based evaluation of implementation and predictive validity.
- [Video walkthrough script](docs/video_script.md) — suggested structure for a project demonstration.

## 10. Known limitations and responsible use

- There are no matched property purchase/resale records, transaction dates, rental income, vacancy, taxes, maintenance, financing, or holding-period cash flows.
- The classification label is assumption-generated. Its holdout ROC AUC of 0.501 does not support reliable investment screening.
- The five-year listing regression target is formula-generated. Its near-perfect R² is not evidence of future resale accuracy.
- NHB RESIDEX is an aggregate city-level assessment-price index. Applying its growth ratio to an individual listing assumes proportional city-level movement and cannot account for locality, property condition, building quality, negotiation, or property-specific effects.
- The listing file has no listing-date field, so true historical listing-price trends cannot be reconstructed.
- Crime rates and verified infrastructure scores are not supplied, so those effects cannot be measured.
- City HPI forecasts are based on a limited historical window; overlapping backtest horizons are not independent.
- Empirical lower/upper scenario multipliers are indicative historical-error context, not guaranteed bounds or calibrated confidence intervals.
- Source redistribution terms and NHB series/base-year updates should be reviewed before wider redistribution or operational use.

**Use the app for exploratory analysis and scenario discussion only. It is not financial advice and should not be used as the sole basis for purchasing or investing in property.**

## 11. Future improvements

1. Obtain property-level registered sale and resale observations with consistent property/location identifiers and dates.
2. Define investment performance using realized appreciation, rental yield, transaction costs, financing, taxes, vacancy and maintenance.
3. Replace proxy targets with observed forward outcomes and validate on chronological and geographic holdouts.
4. Reconcile and refresh NHB RESIDEX linked base-year series; log source download dates and versions.
5. Add locality-level data with documented definitions and reliable geographic joins.
6. Monitor forecast error, model drift, missing-city coverage, and segment-level performance after each refresh.

---

**Built with:** Python · Pandas · NumPy · scikit-learn · Statsmodels · Plotly · Streamlit · MLflow
