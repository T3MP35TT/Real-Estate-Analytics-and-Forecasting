# 🏡 Real Estate Investment Advisor — Predicting Property Profitability & Future Value

Real Estate Investment Advisor is an end-to-end real estate analytics prototype that combines **Python, scikit-learn, Statsmodels, MLflow, and Streamlit** to explore 250,000 Indian property listings, benchmark investment- and price-prediction models, and forecast city-level housing prices — while being explicit about what each result does and doesn't prove.

🔗 **Live application:** https://real-estate-analytics-and-forecasting-kartikey-singh.streamlit.app/

> **Read this before the metrics below:** this is an exploratory decision-support prototype, not a validated investment or resale-valuation system. The investment classifier and the five-year price regression are both trained on **assumption-generated targets**, not observed outcomes. Only the city-level HPI forecast is evaluated against real historical index data. The sections below say exactly which is which.

## 📌 Project Overview

Property listings carry structured signals across location, size, price, age, and amenities — but a listing's *asking* price is not its *resale* outcome, and most public datasets don't contain matched buy/sell pairs. This project is built around that gap rather than around it.

It's designed to answer:

- 🏙️ How do asking prices and listing patterns differ across Indian states and cities?
- 📏 How does a given property compare with similar listings in its city, type, and size band?
- 📈 What does a city's historical NHB RESIDEX housing-price index imply for a five-year scenario?
- 🧪 Can model selection, evaluation, and experiment tracking be made fully reproducible?
- ⚠️ Where do the "investment classification" and "future value" asks outrun what the data can actually support — and how should that be communicated instead of hidden?

## 🎯 Business & Analytical Objectives

- Profile and clean 250,000 listing records across 23 source columns
- Engineer density, binary, and price-per-area features inside leakage-safe pipelines
- Construct and clearly label proxy targets for "good investment" and "five-year price" where no ground truth exists
- Evaluate every model on a **city-group holdout** so no test city ever appears in training
- Forecast city-level HPI from real NHB RESIDEX history using 21 candidate time-series methods
- Track every run, parameter, and metric in MLflow for reproducibility
- Ship all of it through a 7-page Streamlit app, and report honestly on what each number means

## 📊 Dataset & Key Metrics

| Metric | Value |
|---|---:|
| 🏠 Listing records | 250,000 |
| 📋 Source columns | 23 |
| ❓ Missing values (source columns) | 0 |
| 🔁 Exact duplicate rows | 0 |
| ⚠️ Rows with `Floor_No > Total_Floors` | 116,304 (flagged, not silently rewritten) |
| 🏙️ Cities in NHB HPI history | 28 |
| 📅 HPI quarters | 51 (through Dec 2025) |
| 🔮 Forecast horizon | 20 quarters (5 years) |

**Headline finding:** the two prediction targets built for this project behave very differently once evaluated honestly:

| Model | Target | City-holdout result | What it actually means |
|---|---|---|---|
| Logistic Regression | `Good_Investment` (assumption-generated proxy) | ROC AUC **0.501** | Chance level. Cannot be used to rank or screen properties. |
| Linear Regression | `Price_After_5Y_Proxy` (deterministic formula) | R² **≈ 1.0** | The model perfectly reconstructs a formula, not a real five-year resale price. |
| Drift method | City NHB HPI, 20 quarters ahead | Holdout MAPE **15.4%**, 35.7% MAE improvement over a last-value baseline, 90.6% directional accuracy | The one genuinely evidence-based forecast in the project — a real time-series model beating a naive baseline on real index history. |

The app's five-year Property Estimator uses the **HPI scenario**, not the regression model, for exactly this reason.

## 🧠 Analytical Framework

### 🏙️ Market & Listing Analytics
State/city/property-type distributions, segment comparisons, and amenity/price associations across all 250,000 listings.

### 🔍 Comparable Property Analysis
Same-city, same-type, same-BHK comparables, preferring a ±25% area cohort when at least five comparables exist — a screening reference, not a formal valuation.

### 🧪 Investment Classification (proxy, experimental)
```text
Annual_Appreciation_Proxy = 6% − (Property_Age × 0.2 pp)
Good_Investment = 1  if Annual_Appreciation_Proxy > 4%,  else 0
```
A logistic regression is evaluated on how well it reconstructs this *rule*, not on real investment success.

### 💰 Five-Year Price Benchmark (proxy, experimental)
```text
Price_After_5Y_Proxy = Price_in_Lakhs × (1.06)^5
```
A deterministic target — near-perfect regression metrics are expected and demonstrate formula reconstruction, not forecasting skill.

### 📈 City HPI Forecasting (evidence-based)
21 candidate methods — naive, drift, growth-trend, ETS, ARIMA/SARIMA, Theta, and global ML growth models — compared via 20-quarter rolling-origin cross-validation, then re-ranked on a later temporal holdout. **Drift** was selected on lowest CV MAPE and carried an 8-origin holdout MAPE of 15.4%.

## 🛡️ Data Quality & Defensive Analytics

- **Two proxy targets are labeled as proxies everywhere** — in the model metadata, the app, and this README — so a reader can never mistake "reconstructs a formula" for "predicts the future."
- **City-group holdouts, not random splits**, for both classifiers: test cities are fully excluded from training, which is what makes the 0.501 ROC AUC meaningful rather than optimistic.
- **`Floor_No > Total_Floors` affects 116,304 rows (46.5%)** — flagged in a derived field and retained raw rather than silently corrected, so downstream users can decide how to treat it.
- **Price-per-square-foot is recomputed** from price and area rather than trusted from source, after finding 2,534 nonpositive reported values.
- **HPI forecast errors are correlated across overlapping 5-year windows** — the reported MAPE is historical evidence, not an independent-trial guarantee, and the 80% empirical error band is explicitly labeled indicative rather than a calibrated prediction interval.
- **NHB RESIDEX is a city-level index**, not a property-level outcome — applying its growth multiplier to one listing assumes proportional city movement and ignores locality, condition, and negotiation effects.

Full limitations are documented directly in `artifacts/metrics.json` and the executive report — see Documentation below.

## 🎨 Dashboard Experience

### ⚡ Streamlit (7 pages)

| Page | Focus |
|---|---|
| 🏠 Home | Workspace summary and navigation |
| 1️⃣ Executive Overview | Filtered KPIs, market snapshot |
| 2️⃣ Market Insights | State/city/property filters, price distributions, HPI context |
| 3️⃣ Property Estimator | Comparable-market analysis + city-HPI 5-year scenario + experimental classifier output |
| 4️⃣ Listing Explorer | Record-level filtering, CSV export |
| 5️⃣ Forecast Performance | HPI candidate comparison, rolling-origin CV, temporal holdout diagnostics |
| 6️⃣ Methodology & Data | Data inventory, assumptions, limitations, managed-data workspace |

## 🛠️ Data Management

The Methodology & Data page includes a managed-data workspace (`data/managed_data.db`) for inspecting and maintaining working records separately from the source listing file, so edits never touch the original 250,000-row baseline.

## 🗄️ Data Architecture

```text
india_housing_prices.csv       — 250,000 listings, 23 source columns
data/processed/
  nhb_hpi_combined.csv         — 28 cities × 51 quarters, Current-Q view
managed_data.db                — working copy for the data-management workspace
artifacts/
  classifier.joblib            — logistic regression (Good_Investment proxy)
  regressor.joblib             — linear regression (Price_After_5Y_Proxy)
  regression_benchmark.joblib  — benchmark candidate, retained for comparison only
  hpi_forecast_model.json      — selected drift-method HPI model, per city
  metrics.json                 — full CV/holdout metrics for every model
  run_manifest.json            — source hashes, versions, row counts for this run
mlflow.db                      — experiment tracking (experiment: real-estate-investment-advisor)
```

## 🔄 Data Pipeline

```text
📄 india_housing_prices.csv  +  📄 NHB RESIDEX quarterly exports
        ↓
🛡️ Schema validation, dtype standardization, duplicate/invalid checks
        ↓
🧮 Feature engineering (density scores, binary flags, recomputed price/sqft)
        ↓
🏷️ Proxy-target construction (appreciation rule, 5-year compounding formula)
        ↓
       ┌───────────────────────┬───────────────────────┐
       ▼                       ▼                        ▼
  🧪 Classifier            💰 Regressor            📈 HPI forecaster
  (GroupKFold CV,          (GroupKFold CV,          (21 candidates,
   city holdout)            city holdout)            rolling-origin CV
       │                       │                      + temporal holdout)
       └───────────┬───────────┘                       │
                    ▼                                   ▼
             📊 MLflow tracking                  🗄️ artifacts/*.json, *.joblib
                    │                                   │
                    └─────────────┬─────────────────────┘
                                  ▼
                           ⚡ Streamlit app  +  📝 Executive report
```

## 💻 Technical Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Pipeline and application logic |
| 🐼 pandas / NumPy | Cleaning, feature engineering |
| 🤖 scikit-learn | Pipelines, GroupKFold CV, classifier/regressor |
| 📉 Statsmodels | ETS, ARIMA/SARIMA, structural trend candidates |
| 📊 MLflow | Experiment tracking (SQLite backend) |
| ⚡ Streamlit | 7-page interactive app |
| 📈 Plotly | Visualization |
| 📝 PDF/DOCX reporting | Executive report generation |
| 🌿 joblib | Model persistence |

## 📊 Data Analyst / ML Skills Demonstrated

**Experimental design:** city-group holdouts (never random splits) so leakage can't inflate a score; 5-fold `GroupKFold` CV for hyperparameter tuning strictly separated from the final holdout.

**Honest evaluation:** reporting a 0.501 ROC AUC as chance-level rather than burying it, and explaining *why* a near-perfect R² is a red flag rather than a win.

**Time-series forecasting:** rolling-origin cross-validation across 21 candidate methods (naive baselines through ARIMA/SARIMA, Theta, and global ML growth models), with a later, untouched temporal holdout and a documented fallback policy for failed fits.

**Reproducibility:** MLflow-tracked runs, versioned source-file hashes in `run_manifest.json`, and pipeline-internal preprocessing so no transformation is ever fitted on test data.

## 📁 Project Structure

```text
Real Estate Investment Advisor/
│
├── Home.py                              Streamlit entry point
├── pages/
│   ├── 01_Executive_Overview.py
│   ├── 02_Market_Insights.py
│   ├── 03_Property_Estimator.py
│   ├── 04_Listing_Explorer.py
│   ├── 05_Forecast_Performance.py
│   └── 06_Methodology_and_Data.py
│
├── notebooks/
│   ├── 01_data_cleaning_and_eda.ipynb
│   └── 02_model_selection_and_evaluation.ipynb
│
├── scripts/
│   ├── run_eda.py
│   ├── merge_nhb_hpi.py
│   ├── train.py
│   ├── test_hpi_forecast.py
│   └── build_executive_report.py
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── modeling.py
│   └── hpi_forecast.py
│
├── utils/
│   ├── helper.py, sidebar.py, theme.py, data_manager.py
│
├── data/
│   ├── india_housing_prices.csv
│   ├── nhb/                             Raw NHB RESIDEX exports
│   ├── processed/                       Merged HPI input
│   └── managed_data.db
│
├── artifacts/                           Trained models, metrics, run manifest
├── reports/                             30+ EDA/backtest CSVs, figures, DOCX report
├── Real Estate Analysis Executive Report.pdf
├── Real Estate Analysis.mp4
└── requirements.txt
```

## 🚀 Getting Started

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python scripts/run_eda.py          # generate EDA reports
python scripts/merge_nhb_hpi.py    # merge NHB RESIDEX exports
python scripts/train.py            # train, compare, validate, log to MLflow
streamlit run Home.py
```

If the NHB files are already merged and the artifacts exist, you don't need to re-run `merge_nhb_hpi.py` or `train.py` just to launch the app. To view the MLflow experiment UI locally:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 📚 Documentation & Reports

| Resource | Covers |
|---|---|
| [`Real Estate Analysis Executive Report.pdf`](<Real Estate Analysis Executive Report.pdf>) | Leadership-facing findings and limitations |
| [`reports/Real_Estate_Investment_Advisor_Executive_Report.docx`](reports/Real_Estate_Investment_Advisor_Executive_Report.docx) | Same report, editable source |
| `artifacts/metrics.json` | Full CV, holdout, and segment metrics for every model |
| `artifacts/run_manifest.json` | Source file hashes, package versions, row counts |
| `notebooks/01_data_cleaning_and_eda.ipynb` | Full cleaning and EDA walkthrough |
| `notebooks/02_model_selection_and_evaluation.ipynb` | Model comparison and selection |

## 🔐 Known Repo Issues

- **`Real Estate Analysis.mp4` is ~220 MB**, well over GitHub's 100 MB per-file push limit. It needs Git LFS or external hosting (YouTube, Drive) with a link here — it will not push as a normal tracked file.
- **The repo doesn't include a `docs/` folder.** If you maintain separate methodology/provenance/findings docs elsewhere, link them here; otherwise `artifacts/metrics.json`, `run_manifest.json`, and the executive report above are the current source of truth for methodology and limitations.
- **Keep `data/india_housing_prices.csv` (40 MB) in mind** if your Git host or plan has size-conscious limits — it's well under GitHub's cap but worth knowing before you add more large source files.

## 💡 Key Design Principles

1. **Label every proxy target as a proxy**, in the code, the metrics file, and the app — never just in a footnote.
2. **A city-group holdout is non-negotiable.** No test city ever appears during training, for either classifier or regressor.
3. **A suspiciously perfect score is a finding, not a win.** R² ≈ 1.0 on a deterministic target is reported as target reconstruction, explicitly.
4. **Only deploy what's actually evidenced.** The app uses the HPI scenario, not the regression model, for its five-year estimate — because that's the one with real evaluation behind it.
5. **State the gap the data can't close.** No matched resale, rental, tax, or financing data exists, and the README says so rather than implying otherwise.

## 📈 Why This Project Matters

The discipline here — distinguishing a model that reconstructs a known formula from one that actually forecasts something, and saying so explicitly even when the "exciting" number (R² ≈ 1.0) is the misleading one — is the same discipline that matters in any applied ML setting with imperfect ground truth: churn prediction with a leaky proxy label, fraud models evaluated on the wrong split, or any "AI investment advisor" built on scraped listings rather than realized outcomes.

## ⭐ Project Highlights

```text
🏠 250,000 Listings  +  🏙️ 28 Cities, 51 Quarters of Real HPI History
🧪 2 Proxy Models (honestly scored at chance-level / formula-reconstruction)
📈 1 Evidence-Based Forecast (21 candidates compared, 35.7% skill over baseline)
⚡ 7-Page Streamlit App  +  📊 MLflow Tracking  +  📝 Executive Report
```

---

## 👤 Author

**Kartikey Singh**

**Business Analyst | Power BI | Python | SQL | Excel**

🔗 LinkedIn: [Kartikey_Singh](https://www.linkedin.com/in/btwitskartiksinghdatanalyst/)

💻 GitHub: [Kartikey_Singh](https://github.com/T3MP35TT)

💼 Portfolio: [Kartikey_Singh](https://sites.google.com/view/kartikeysingh09/home)

## 📄 License

No license file is included yet. Add a `LICENSE` (e.g. MIT or Apache-2.0) before treating this as open source.

⭐ If you find the project useful, consider giving the repository a star.
