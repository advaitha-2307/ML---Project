import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

INPUT = Path("data/processed/indofloods_integrated.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT)

print("Dataset shape:", df.shape)

# ---------------------------------------------------------
# Basic information
# ---------------------------------------------------------

quality = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str).values,
    "missing_count": df.isna().sum().values,
    "missing_percent": (
        df.isna().mean().values * 100
    ),
    "unique_values": [
        df[col].nunique(dropna=True)
        for col in df.columns
    ]
})

# ---------------------------------------------------------
# Sort by missing percentage
# ---------------------------------------------------------

quality = quality.sort_values(
    "missing_percent",
    ascending=False
)

# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

output = REPORT_DIR / "data_quality_report.csv"

quality.to_csv(
    output,
    index=False
)

print("\nData quality report saved to:")
print(output)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n========== SUMMARY ==========")

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate EventIDs:", df["EventID"].duplicated().sum())
print("Total missing values:", df.isna().sum().sum())

print("\nTop 15 columns by missing percentage:")
print(
    quality.head(15).to_string(index=False)
)