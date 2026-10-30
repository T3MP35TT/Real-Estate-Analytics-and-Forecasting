import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Running `python scripts/run_eda.py` puts scripts/ on sys.path, not the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORT_DIR
from src.data import load_data


def main():
    df = load_data()
    sns.set_theme(style="whitegrid")
    summary = df.describe(include="all").transpose()
    summary.to_csv(REPORT_DIR / "dataset_summary.csv")

    city = (df.groupby("City", dropna=False)
            .agg(listings=("ID", "count"), median_price_lakhs=("Price_in_Lakhs", "median"),
                 median_price_per_sqft=("Price_per_SqFt", "median"))
            .sort_values("median_price_lakhs", ascending=False))
    city.to_csv(REPORT_DIR / "city_price_summary.csv")
    ax = city.sort_values("median_price_lakhs").tail(20)["median_price_lakhs"].plot.barh(figsize=(10, 8))
    ax.set(title="Median listing price by city (top 20)", xlabel="Price (lakhs)", ylabel="City")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "city_price_trends.png", dpi=160); plt.close()

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df.sample(min(5000, len(df)), random_state=42), x="Size_in_SqFt",
                    y="Investment_Return_Proxy", hue="Good_Investment", alpha=.5, s=20)
    plt.title("Property area and investment return proxy")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "area_return.png", dpi=160); plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Security", y="Good_Investment", errorbar=None)
    plt.ylabel("Share labeled good (proxy)"); plt.title("Security indicator and proxy label")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "security_investment.png", dpi=160); plt.close()

    corr_cols = ["Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property",
                 "Nearby_Schools", "Nearby_Hospitals", "School_Density_Score", "Investment_Return_Proxy"]
    plt.figure(figsize=(10, 7)); sns.heatmap(df[corr_cols].corr(), cmap="vlag", center=0, annot=True, fmt=".2f")
    plt.title("Housing and amenity correlations")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=160); plt.close()

    note = """# EDA findings guide\n\nGenerated from the supplied listing dataset. Read `city_price_summary.csv` for the ranked city medians. The source has no crime-rate or infrastructure-score field and no resale observations, so these analyses are not represented by synthetic data. `Investment_Return_Proxy` is a rule based on assumed appreciation less an age adjustment; it is not observed return. School and hospital density scores are counts per 1,000 sq ft, a simple listing-level proxy rather than geographic density. Review the charts alongside these limitations.\n"""
    (REPORT_DIR / "eda_notes.md").write_text(note, encoding="utf-8")
    print(f"EDA outputs written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
