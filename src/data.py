import numpy as np
import pandas as pd

from src.config import (
    ANNUAL_APPRECIATION_ASSUMPTION,
    AGE_DEPRECIATION_PER_YEAR,
    DATA_PATH,
    FORECAST_YEARS,
    GOOD_INVESTMENT_THRESHOLD,
    REFERENCE_YEAR,
)

REQUIRED_COLUMNS = [
    "ID", "State", "City", "Locality", "Property_Type", "BHK",
    "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
    "Furnished_Status", "Floor_No", "Total_Floors", "Age_of_Property",
    "Nearby_Schools", "Nearby_Hospitals", "Public_Transport_Accessibility",
    "Parking_Space", "Security", "Amenities", "Facing", "Owner_Type",
    "Availability_Status",
]


def load_data(path=DATA_PATH):
    """Load, validate, de-duplicate, and engineer the supplied dataset."""
    df = pd.read_csv(path, low_memory=False)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    raw_rows = len(df)
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    missing_tokens = {"": np.nan, "na": np.nan, "n/a": np.nan, "null": np.nan,
                      "none": np.nan, "not available": np.nan, "unknown": np.nan}
    for col in text_columns:
        cleaned = df[col].astype("string").str.strip()
        df[col] = cleaned.mask(cleaned.str.lower().isin(missing_tokens), np.nan)
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    numeric = [
        "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
        "Age_of_Property", "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid_price_area = (df["Size_in_SqFt"] > 0) & (df["Price_in_Lakhs"] > 0)
    invalid_price_area_removed = int((~valid_price_area).sum())
    df = df.loc[valid_price_area].copy()
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 100_000 / df["Size_in_SqFt"]
    df["Age_of_Property"] = (REFERENCE_YEAR - df["Year_Built"]).clip(lower=0)
    df["Price_per_SqFt_Thousands"] = df["Price_per_SqFt"] / 1000
    df["School_Density_Score"] = df["Nearby_Schools"] / (df["Size_in_SqFt"] / 1000)
    df["Hospital_Density_Score"] = df["Nearby_Hospitals"] / (df["Size_in_SqFt"] / 1000)
    for col in ["Parking_Space", "Security"]:
        df[f"{col}_Binary"] = df[col].astype("string").str.lower().map({"yes": 1, "no": 0})
    # Listing-only data has no transaction history. This rule proxy uses age
    # and a fixed market assumption; it is not estimated from actual returns.
    age = df["Age_of_Property"].clip(lower=0)
    df["Annual_Appreciation_Proxy"] = ANNUAL_APPRECIATION_ASSUMPTION - age * AGE_DEPRECIATION_PER_YEAR
    df["Good_Investment"] = (
        df["Annual_Appreciation_Proxy"] > GOOD_INVESTMENT_THRESHOLD
    ).astype("float64").where(df["Age_of_Property"].notna())
    df["Price_After_5Y_Proxy"] = df["Price_in_Lakhs"] * (1 + ANNUAL_APPRECIATION_ASSUMPTION) ** FORECAST_YEARS
    df["Investment_Return_Proxy"] = df["Annual_Appreciation_Proxy"]
    df.attrs["quality"] = {
        "raw_rows": raw_rows,
        "exact_duplicates_removed": duplicates_removed,
        "invalid_area_or_price_removed": invalid_price_area_removed,
        "clean_rows": len(df),
        "missing_values_by_column": df.isna().sum().to_dict(),
    }
    return df


def model_frame(df):
    """Prepare model features while preserving task-specific missing targets."""
    from src.config import CLASSIFICATION_FEATURES, REGRESSION_FEATURES
    frame = df.copy()
    for col in CLASSIFICATION_FEATURES + REGRESSION_FEATURES:
        if col not in frame:
            frame[col] = np.nan
    frame["Amenities"] = frame["Amenities"].fillna("Unknown")
    return frame
