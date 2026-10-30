# Real Estate Project Assessment

## Rubric and scoring method

This assessment adapts the eight sections in the supplied *Technical Review Feedback & Evaluation Template* to this real-estate project. The Strava-specific workout comments in that document do not apply. Scores reflect what is currently evidenced in the repository, not what could be claimed after adding unsupported assumptions.

| Score | Meaning |
| ---: | --- |
| 1 | Missing or seriously incorrect |
| 2 | Major gaps; limited evidence |
| 3 | Partially meets expectations |
| 4 | Strong, with specific gaps |
| 5 | Complete, reproducible, and supported by appropriate evidence |

## Scores

| Evaluation area | Score | Evidence and remaining gap |
| --- | :---: | --- |
| Data foundations | **4/5** | The 250,000-row listing file has a recorded Kaggle source, local SHA-256, and cleaning summary. NHB city-level HPI exports are merged with provenance and checksums; they are aggregate assessment indices, not property-level resale data. Crime/infrastructure outcomes and matched resale prices are still unavailable. |
| Methodology and experimental design | **4/5** | The HPI forecast compares five methods using expanding-window five-year rolling-origin CV and reserves the latest four origins for temporal holdout scoring. Overlapping horizons and only 13 years of index history limit the strength of the evidence. |
| Modeling and feature engineering | **4/5** | Five auditable city-index forecasting methods are compared; the CV-selected method is refit on each city's complete history. The property estimate applies this city HPI ratio to asking price, but is not trained on property-linked resale outcomes. |
| Evaluation and metrics | **4/5** | The HPI model reports 20 rolling CV origins and a 112 city-origin final holdout, with MAPE, MAE, RMSE, and directional accuracy. Holdout MAPE is 19.3%; these metrics measure city-index forecasts, not resale prices. |
| Code quality and reproducibility | **4/5** | Data, forecasting, and app code are modular. HPI output, selected method, rolling predictions, source URL, and input hash are recorded. There is no automated test suite or lockfile. |
| Robustness and limitations | **3/5** | Rolling-origin validation tests historical time transfer and the app surfaces the holdout error. The overlapping five-year horizons are correlated; property-linked outcomes, locality coverage, and post-2025 validation are unavailable. |
| Communication | **4/5** | README, methodology, notebooks, EDA reports, and training outputs make the workflow reviewable. The model results need regeneration after the latest corrections, and the project has a video script rather than a recorded walkthrough. |
| Business value | **2/5** | The project can demonstrate listing exploration and scenario screening. It cannot currently support a real investment decision because the targets are formula-generated and key market-risk/outcome variables are absent. |

**Overall: 3.6/5** (unweighted mean of the eight category scores, rounded to one decimal).

## Why the score is not 5/5

The HPI forecast adds a historical target and temporal backtest at the city-index level. The latest-period holdout MAPE of 19.3% shows meaningful forecast error, and the metric does not represent individual property resale accuracy. A decision-grade property resale model requires dated, property-linked sales and later-period validation.

## Work needed to reach 5/5

1. **Methodology/modeling:** calibrate the proxy classifier before presenting probabilities as confidence; explore a wider set of time-series baselines when more NHB history becomes available.
2. **Evaluation:** evaluate actual resale/return outcomes when available, and report uncertainty and errors by city/property segment.
3. **Reproducibility:** run the workflow in a clean environment, record a tested dependency lock, and add automated checks for cleaning, splits, and target leakage.
4. **Robustness/business value:** with existing data, review worst-performing cities/property types and fold train/test gaps. Temporal validation, drift monitoring, and decision-value analysis require time-stamped outcomes that the current data does not contain.
5. **Communication:** execute the notebook from top to bottom, confirm generated artifact tables and conclusions, and record the project walkthrough.

## Current interpretation

Treat the app as an educational listing-analysis tool. The city-HPI estimate is a historically backtested market projection, not a property-specific resale prediction. The “Good Investment” classifier remains a rule-generated proxy, not a prediction of actual profitability.
