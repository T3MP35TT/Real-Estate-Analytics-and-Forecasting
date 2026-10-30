from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "india_housing_prices.csv"
HPI_PATH = ROOT / "data" / "processed" / "nhb_hpi_combined.csv"
LISTING_SOURCE_URL = "https://www.kaggle.com/datasets/ankushpanday1/india-house-price-prediction"
HPI_SOURCE_URL = "https://residex.nhbonline.org.in/DashboardBY2018/DashboardBY2018"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
ARTIFACT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Transparent instructional proxies; replace with locally validated market data.
ANNUAL_APPRECIATION_ASSUMPTION = 0.06
AGE_DEPRECIATION_PER_YEAR = 0.002
GOOD_INVESTMENT_THRESHOLD = 0.04
FORECAST_YEARS = 5
REFERENCE_YEAR = 2026
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "BHK", "Size_in_SqFt", "Year_Built", "Floor_No", "Total_Floors",
    "Nearby_Schools", "Nearby_Hospitals", "Parking_Space_Binary",
    "Security_Binary", "School_Density_Score", "Hospital_Density_Score",
]
CATEGORICAL_FEATURES = [
    "State", "City", "Locality", "Property_Type", "Furnished_Status",
    "Public_Transport_Accessibility", "Amenities", "Facing", "Owner_Type",
    "Availability_Status",
]
CLASSIFICATION_FEATURES = [
    feature for feature in NUMERIC_FEATURES if feature != "Year_Built"
] + CATEGORICAL_FEATURES
REGRESSION_FEATURES = CLASSIFICATION_FEATURES + ["Price_in_Lakhs"]
