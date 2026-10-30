# Project video script (about 3 minutes)

## 0:00-0:25 - Problem

This project explores how data analysis and machine learning can support real-estate screening. It combines a reproducible Python workflow with a Streamlit dashboard for comparing listings and viewing investment scenarios.

## 0:25-0:55 - Dataset and honest scope

The Kaggle India housing file contains asking prices and property attributes, but no matched resale records. NHB RESIDEX provides quarterly city-level assessment indices. We use those city histories to select and backtest a five-year HPI forecasting method, then apply the projected city growth ratio to a listing's asking price. This estimates city-market movement, not property-specific resale performance.

## 0:55-1:30 - Data preparation and EDA

The pipeline removes exact duplicates, coerces numeric fields, recomputes price per square foot, and derives amenity-density indicators. The EDA examines distributions, data quality, city median prices, area-price association, property segments, and amenity coverage. The map shows city-level asking-price gaps; it does not locate individual properties. Unsupported crime and infrastructure claims are left out.

## 1:30-2:05 - Model comparison and validation

For the HPI estimate, five time-series methods are compared using expanding-window five-year rolling-origin CV. The latest four forecast origins are kept as a temporal holdout. The selected damped-growth method scored 15.4% CV MAPE and 19.3% holdout MAPE on city HPI values. These are index forecast errors, not individual resale-price errors. The separate investment classifier still predicts an age-based proxy label.

## 2:05-2:40 - Dashboard walkthrough

In the dashboard, I filter listings by city, property type, BHK, area, and asking price. I enter property details to see the proxy classification and, when NHB covers the city, a five-year HPI-based city-market estimate. The dashboard also displays the model's historical holdout error.

## 2:40-3:00 - Next steps

This forecast covers city-level index movement only; it cannot distinguish neighborhoods or property characteristics. A property-level resale model still needs dated transactions linked to property attributes and later-period validation.

## Recording checklist

1. Run `python scripts/run_eda.py`, `python scripts/merge_nhb_hpi.py`, and `python scripts/train.py`.
2. Open `notebooks/02_model_selection_and_evaluation.ipynb` to show CV rankings and unseen-city holdout results.
3. Start the demo with `streamlit run app.py`.
4. Show the HPI city projections, apply filters, submit an example property, and explain the city-level limitation.
5. Include `reports/` and `artifacts/metrics.json` so viewers can review the outputs.
