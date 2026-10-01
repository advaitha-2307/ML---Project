import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

INPUT = Path("data/processed/indofloods_integrated.csv")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT)

print("=" * 70)
print("INDOFLOODS FEATURE PROFILING")
print("=" * 70)

print("\nDataset shape:", df.shape)

# ============================================================
# 1. BASIC FEATURE INFORMATION
# ============================================================

profile = pd.DataFrame({
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

# ============================================================
# 2. NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\n" + "=" * 70)
print("NUMERICAL FEATURES")
print("=" * 70)

print("Number of numerical columns:", len(numeric_columns))

for col in numeric_columns:
    print(col)

print("\n" + "=" * 70)
print("CATEGORICAL / TEXT FEATURES")
print("=" * 70)

print("Number of categorical/text columns:", len(categorical_columns))

for col in categorical_columns:
    print(col)

# ============================================================
# 3. HIGH MISSINGNESS FEATURES
# ============================================================

print("\n" + "=" * 70)
print("HIGH MISSINGNESS FEATURES")
print("=" * 70)

high_missing = profile[
    profile["missing_percent"] >= 50
].sort_values(
    "missing_percent",
    ascending=False
)

print(
    high_missing[
        [
            "column",
            "missing_count",
            "missing_percent",
            "unique_values"
        ]
    ].to_string(index=False)
)

# ============================================================
# 4. LOW VARIANCE / CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CONSTANT / NEAR-CONSTANT FEATURES")
print("=" * 70)

constant_features = []

for col in df.columns:

    unique_count = df[col].nunique(dropna=True)

    if unique_count <= 1:
        constant_features.append(col)

print("Constant features:", len(constant_features))

for col in constant_features:
    print(col)

# ============================================================
# 5. NUMERICAL SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL SUMMARY")
print("=" * 70)

numeric_summary = df[numeric_columns].describe().T

numeric_summary["missing_count"] = (
    df[numeric_columns].isna().sum()
)

numeric_summary["missing_percent"] = (
    df[numeric_columns].isna().mean() * 100
)

print(
    numeric_summary[
        [
            "count",
            "mean",
            "std",
            "min",
            "max",
            "missing_count",
            "missing_percent"
        ]
    ].head(30).to_string()
)

# ============================================================
# 6. HIGH CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("HIGHLY CORRELATED NUMERICAL FEATURES")
print("=" * 70)

# Correlation matrix
corr_matrix = df[numeric_columns].corr()

# Get only upper triangle
upper_triangle = corr_matrix.where(
    np.triu(
        np.ones(corr_matrix.shape),
        k=1
    ).astype(bool)
)

correlation_pairs = []

for col in upper_triangle.columns:

    for row in upper_triangle.index:

        value = upper_triangle.loc[row, col]

        if pd.notna(value) and abs(value) >= 0.90:

            correlation_pairs.append({
                "feature_1": row,
                "feature_2": col,
                "correlation": value
            })

high_corr_df = pd.DataFrame(
    correlation_pairs
)

if len(high_corr_df) > 0:

    high_corr_df = high_corr_df.sort_values(
        "correlation",
        key=lambda x: abs(x),
        ascending=False
    )

    print(
        high_corr_df.to_string(index=False)
    )

else:

    print("No feature pairs with |correlation| >= 0.90")

# ============================================================
# 7. CATEGORICAL FEATURE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL FEATURE CARDINALITY")
print("=" * 70)

categorical_summary = []

for col in categorical_columns:

    categorical_summary.append({
        "column": col,
        "unique_values": df[col].nunique(dropna=True),
        "missing_count": df[col].isna().sum(),
        "missing_percent": df[col].isna().mean() * 100
    })

categorical_summary_df = pd.DataFrame(
    categorical_summary
)

print(
    categorical_summary_df.to_string(index=False)
)

# ============================================================
# 8. TARGET DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 70)
print("TARGET VARIABLES")
print("=" * 70)

targets = [
    "Flood Type",
    "Peak Flood Level (m)",
    "Peak Discharge Q (cumec)",
    "Flood Volume (cumec)"
]

for target in targets:

    if target not in df.columns:
        continue

    print("\nTarget:", target)

    print(
        "Missing:",
        df[target].isna().sum()
    )

    print(
        "Unique values:",
        df[target].nunique(dropna=True)
    )

    if df[target].dtype == "object":

        print(
            df[target]
            .value_counts(dropna=False)
            .to_string()
        )

    else:

        print(
            df[target]
            .describe()
            .to_string()
        )

# ============================================================
# 9. DATE COLUMN ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DATE COLUMN ANALYSIS")
print("=" * 70)

date_columns = [
    col for col in df.columns
    if "date" in col.lower()
]

for col in date_columns:

    converted = pd.to_datetime(
        df[col],
        errors="coerce"
    )

    print(
        f"\n{col}"
    )

    print(
        "Valid dates:",
        converted.notna().sum()
    )

    print(
        "Missing/invalid dates:",
        converted.isna().sum()
    )

    if converted.notna().any():

        print(
            "Minimum:",
            converted.min()
        )

        print(
            "Maximum:",
            converted.max()
        )

# ============================================================
# 10. SAVE COMPLETE PROFILE
# ============================================================

profile_output = (
    REPORT_DIR /
    "feature_profile.csv"
)

profile.to_csv(
    profile_output,
    index=False
)

# ============================================================
# 11. SAVE CORRELATION MATRIX
# ============================================================

corr_output = (
    REPORT_DIR /
    "correlation_matrix.csv"
)

corr_matrix.to_csv(
    corr_output
)

# ============================================================
# 12. SAVE HIGH CORRELATION PAIRS
# ============================================================

high_corr_output = (
    REPORT_DIR /
    "high_correlation_pairs.csv"
)

high_corr_df.to_csv(
    high_corr_output,
    index=False
)

# ============================================================
# 13. SAVE CATEGORICAL SUMMARY
# ============================================================

categorical_output = (
    REPORT_DIR /
    "categorical_feature_summary.csv"
)

categorical_summary_df.to_csv(
    categorical_output,
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REPORTS CREATED")
print("=" * 70)

print(
    "\n1. Feature profile:"
)
print(profile_output)

print(
    "\n2. Correlation matrix:"
)
print(corr_output)

print(
    "\n3. High correlation pairs:"
)
print(high_corr_output)

print(
    "\n4. Categorical feature summary:"
)
print(categorical_output)

print("\n" + "=" * 70)
print("FEATURE PROFILING COMPLETE")
print("=" * 70)