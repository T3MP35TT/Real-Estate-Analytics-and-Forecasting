# Methodology, findings, and use cases

## Objective

Provide an interactive analysis of Indian property listings with a proxy investment classification and a city-HPI-based five-year price estimate. The Kaggle listing CSV contains asking prices and property attributes, not matched property resale transactions. NHB quarterly city-level RESIDEX indices provide historical market movement, not property resale labels. Provenance and source limitations are recorded in [`data_provenance.md`](data_provenance.md).

## Preprocessing and feature engineering

`src/data.py` trims text and standardizes common missing-value tokens, drops exact duplicate rows, coerces numeric columns, excludes nonpositive area or price, and recomputes price per square foot from price in lakhs and area. It derives property age from 2026 minus build year, school and hospital counts per 1,000 square feet, and binary parking/security indicators. Model pipelines impute numeric values with medians and missingness indicators, impute categorical values by mode, scale numeric columns, and one-hot encode categories. Unknown categories are supported at inference. `Year_Built`, which directly determines the classification proxy, is excluded from classifier inputs and used only to calculate the target and its explicit rule reference. A missing build year leaves the age-based classification proxy unknown and that row is excluded from classification.

## Targets and limitations

- The investment proxy equals assumed 6% baseline appreciation less 0.2 percentage points for each property year of age. Values above 4% receive label 1. It is a transparent rule, not a label learned from observed investor profit; only age determines this label, so feature importance and confidence cannot establish actual profitability.
- The five-year estimate multiplies the asking price by the city's forecast NHB HPI ratio. Candidate time-series methods are no-change, log-linear trend, trailing five-year growth, trailing three-year growth, and damped recent growth. Selection uses expanding-window rolling-origin five-year cross-validation, with the latest four origins reserved as a temporal holdout. The selected method is refit to each city's complete index history.
- HPI metrics evaluate future city index values, not property resale prices. The estimate assumes the listing price is aligned to the latest available HPI level and changes proportionally with its city index; listing dates are unavailable, and locality, condition, property type, and timing differences are not modeled. Cities without NHB coverage are shown as unavailable rather than assigned another city's growth.
- Security is not crime rate; school/hospital counts are not an infrastructure score. Neither crime impact nor infrastructure versus resale can be tested from the provided CSV. EDA excludes these unsupported claims.
- Assumed values and current-year reference can be edited in `src/config.py` and `src/data.py`; retrain and regenerate reports after changes.

## Evaluation and experiment tracking

The investment-proxy classifier uses city-grouped CV and a city holdout; it predicts the engineered age-based label only. The same `scripts/train.py` command evaluates five HPI forecasting methods at a 20-quarter horizon. Each rolling-origin prediction uses only observations available by its forecast origin. Earlier origins are used for CV, and the latest four are held out for temporal scoring. The forecast windows overlap, so their errors are correlated and the history is limited. The selected method is refit by city, and its projected HPI ratio is applied to each listing's asking price. This does not validate property-specific resale accuracy.

The HPI forecast report includes CV comparisons, forecast-origin predictions, final holdout metrics, city projections, and the source hash. MAPE, MAE, and RMSE measure city-HPI forecast errors. These scores are not property resale metrics. The classifier reports standard binary-classification metrics for the engineered investment proxy only.

## EDA questions

The EDA notebook profiles distributions and outliers; quantifies area-price and amenity associations; compares city, property-type, and BHK segments with rank tests and effect sizes; and plots patterns with sample counts and uncertainty ranges. The area-price chart reports correlation and binned means so weak or absent patterns remain visible. For crime and infrastructure analyses, add sourced locality-level measures and resale observations before drawing conclusions.

## Business use cases

- Compare current listings and asking prices across cities and property types.
- Screen properties using transparent, user-adjustable assumptions.
- Demonstrate a property analysis workflow for a real-estate platform or analyst.
- Identify which external datasets need to be collected before a decision-grade investment product is possible.

## Findings and generated outputs

The HPI forecast method is selected on earlier rolling origins and scored on the latest four origins. The current run selected damped recent growth: CV MAPE 15.4%, final holdout MAPE 19.3%, over 112 city-origin forecasts. The holdout spans forecast origins Mar-Dec 2020 and compares projected city HPI with observed values through Dec 2025. These are correlated five-year index forecasts, not property resale outcomes. The project-level rubric and limitations are in [`project_assessment.md`](project_assessment.md).
