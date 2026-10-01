import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "indofloods_modeling_base.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("INDOFLOODS MODELING BASE DATASET CHECK")
print("=" * 70)

print(f"\nLoading dataset from:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")


# ============================================================
# 1. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET INFORMATION")
print("=" * 70)

print(f"\nRows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print(f"Shape   : {df.shape}")


# ============================================================
# 2. COLUMN LIST
# ============================================================

print("\n" + "=" * 70)
print("2. ALL COLUMNS")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:3}. {column}")


# ============================================================
# 3. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("3. DATA TYPES")
print("=" * 70)

print("\nNumeric columns:")
numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

for column in numeric_columns:
    print(f"  - {column}")

print(f"\nTotal numeric columns: {len(numeric_columns)}")


print("\nCategorical / text columns:")
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

for column in categorical_columns:
    print(f"  - {column}")

print(f"\nTotal categorical/text columns: {len(categorical_columns)}")


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("4. MISSING VALUES")
print("=" * 70)

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percent.round(2)
})

missing_report = missing_report[
    missing_report["Missing Count"] > 0
].sort_values(
    by="Missing Percentage",
    ascending=False
)

if missing_report.empty:
    print("\nNo missing values found.")
else:
    print("\nColumns containing missing values:")
    print(missing_report.to_string())

missing_report.to_csv(
    REPORTS_DIR / "modeling_base_missing_values.csv"
)

print(
    f"\nSaved missing-value report to: "
    f"{REPORTS_DIR / 'modeling_base_missing_values.csv'}"
)


# ============================================================
# 5. CONSTANT COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("5. CONSTANT COLUMNS")
print("=" * 70)

constant_columns = []

for column in df.columns:
    if df[column].nunique(dropna=False) <= 1:
        constant_columns.append(column)

if constant_columns:
    for column in constant_columns:
        print(f"  - {column}")

    print(f"\nTotal constant columns: {len(constant_columns)}")
else:
    print("\nNo constant columns found.")


# ============================================================
# 6. IDENTIFIER COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("6. IDENTIFIER / GROUPING COLUMNS")
print("=" * 70)

identifier_candidates = [
    "EventID",
    "GaugeID",
    "Station",
    "River Name/ Tributory/ SubTributory"
]

for column in identifier_candidates:
    if column in df.columns:
        print(
            f"  - {column} | "
            f"unique values = {df[column].nunique(dropna=True)}"
        )


# ============================================================
# 7. TARGET COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("7. TARGET COLUMNS")
print("=" * 70)

target_columns = [
    "Flood Type",
    "Peak Flood Level (m)",
    "Peak Discharge Q (cumec)",
    "Flood Volume (cumec)"
]

target_summary = []

for target in target_columns:

    if target not in df.columns:
        print(f"\n[NOT FOUND] {target}")
        continue

    print(f"\nTarget: {target}")
    print(f"  Data type       : {df[target].dtype}")
    print(f"  Total rows      : {len(df)}")
    print(f"  Missing values  : {df[target].isna().sum()}")
    print(
        f"  Missing %       : "
        f"{df[target].isna().mean() * 100:.2f}%"
    )
    print(
        f"  Unique values   : "
        f"{df[target].nunique(dropna=True)}"
    )

    if df[target].dtype == "object":
        print("  Value counts:")
        print(df[target].value_counts(dropna=False).to_string())

    else:
        print("  Statistics:")
        print(df[target].describe().to_string())

    target_summary.append({
        "Target": target,
        "Rows": len(df),
        "Missing": df[target].isna().sum(),
        "Missing_Percentage": round(
            df[target].isna().mean() * 100, 2
        ),
        "Unique_Values": df[target].nunique(dropna=True)
    })


target_summary_df = pd.DataFrame(target_summary)

target_summary_df.to_csv(
    REPORTS_DIR / "modeling_base_target_summary.csv",
    index=False
)


# ============================================================
# 8. FLOOD TYPE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("8. FLOOD TYPE DISTRIBUTION")
print("=" * 70)

if "Flood Type" in df.columns:

    flood_counts = df["Flood Type"].value_counts(dropna=False)

    flood_percent = (
        df["Flood Type"]
        .value_counts(normalize=True, dropna=False)
        * 100
    )

    flood_distribution = pd.DataFrame({
        "Count": flood_counts,
        "Percentage": flood_percent.round(2)
    })

    print("\n", flood_distribution.to_string())

    flood_distribution.to_csv(
        REPORTS_DIR / "flood_type_distribution.csv"
    )


# ============================================================
# 9. DATE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("9. DATE FEATURES")
print("=" * 70)

date_columns = [
    column
    for column in df.columns
    if "date" in column.lower()
]

if date_columns:

    for column in date_columns:

        converted = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        valid_count = converted.notna().sum()
        missing_count_date = converted.isna().sum()

        print(f"\n{column}")
        print(f"  Valid dates   : {valid_count}")
        print(f"  Invalid/missing: {missing_count_date}")

        if valid_count > 0:
            print(f"  Minimum date  : {converted.min()}")
            print(f"  Maximum date  : {converted.max()}")

else:
    print("\nNo date columns detected.")


# ============================================================
# 10. NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("10. NUMERICAL FEATURE SUMMARY")
print("=" * 70)

if numeric_columns:

    numeric_summary = df[numeric_columns].describe().T

    print("\n", numeric_summary.to_string())

    numeric_summary.to_csv(
        REPORTS_DIR / "modeling_base_numeric_summary.csv"
    )


# ============================================================
# 11. CATEGORICAL CARDINALITY
# ============================================================

print("\n" + "=" * 70)
print("11. CATEGORICAL FEATURE CARDINALITY")
print("=" * 70)

categorical_summary = []

for column in categorical_columns:

    unique_count = df[column].nunique(dropna=True)
    missing = df[column].isna().sum()

    print(
        f"{column}: "
        f"{unique_count} unique, "
        f"{missing} missing"
    )

    categorical_summary.append({
        "Column": column,
        "Unique_Values": unique_count,
        "Missing": missing
    })

categorical_summary_df = pd.DataFrame(categorical_summary)

categorical_summary_df.to_csv(
    REPORTS_DIR / "modeling_base_categorical_summary.csv",
    index=False
)


# ============================================================
# 12. DUPLICATE EVENT CHECK
# ============================================================

print("\n" + "=" * 70)
print("12. DUPLICATE EVENT CHECK")
print("=" * 70)

if "EventID" in df.columns:

    total_events = len(df)

    unique_events = df["EventID"].nunique()

    duplicate_events = total_events - unique_events

    print(f"\nTotal EventID values : {total_events}")
    print(f"Unique EventID values: {unique_events}")
    print(f"Duplicate EventIDs   : {duplicate_events}")

    if duplicate_events == 0:
        print("\nNo duplicate EventIDs found.")
    else:
        print("\nWARNING: Duplicate EventIDs found.")


# ============================================================
# 13. GAUGE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("13. GAUGE DISTRIBUTION")
print("=" * 70)

if "GaugeID" in df.columns:

    gauge_count = df["GaugeID"].nunique()

    events_per_gauge = df["GaugeID"].value_counts()

    print(f"\nNumber of unique gauges: {gauge_count}")

    print("\nEvents per gauge:")
    print(events_per_gauge.describe().to_string())

    print("\nTop 10 gauges by number of events:")
    print(events_per_gauge.head(10).to_string())

    events_per_gauge.to_csv(
        REPORTS_DIR / "events_per_gauge.csv"
    )


# ============================================================
# 14. POSSIBLE POST-EVENT / LEAKAGE COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("14. POSSIBLE POST-EVENT / LEAKAGE COLUMNS")
print("=" * 70)

leakage_candidates = [
    "Peak Flood Level (m)",
    "Peak FL Date",
    "Peak Discharge Q (cumec)",
    "Peak Discharge Date",
    "Flood Volume (cumec)",
    "Event Duration (days)",
    "Time to Peak (days)",
    "Recession Time (day)",
    "Num Peak FL"
]

found_leakage_candidates = [
    column
    for column in leakage_candidates
    if column in df.columns
]

for column in found_leakage_candidates:
    print(f"  - {column}")


# ============================================================
# 15. POSSIBLE IDENTIFIER / NON-PREDICTOR COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("15. POSSIBLE NON-PREDICTOR COLUMNS")
print("=" * 70)

non_predictor_candidates = [
    "EventID",
    "GaugeID",
    "Station",
    "River Name/ Tributory/ SubTributory",
    "Privacy",
    "Reliability"
]

for column in non_predictor_candidates:
    if column in df.columns:
        print(f"  - {column}")


# ============================================================
# 16. FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("16. FINAL CHECK")
print("=" * 70)

print(f"\nDataset shape              : {df.shape}")
print(f"Numeric columns            : {len(numeric_columns)}")
print(f"Categorical/text columns   : {len(categorical_columns)}")
print(f"Columns with missing data  : {len(missing_report)}")
print(f"Constant columns           : {len(constant_columns)}")

if "EventID" in df.columns:
    print(
        f"Duplicate EventIDs         : "
        f"{df['EventID'].duplicated().sum()}"
    )

if "GaugeID" in df.columns:
    print(
        f"Unique GaugeIDs            : "
        f"{df['GaugeID'].nunique()}"
    )


# ============================================================
# REPORT FILES
# ============================================================

print("\n" + "=" * 70)
print("REPORTS CREATED")
print("=" * 70)

print(f"\nReports directory:")
print(REPORTS_DIR)

print("\nThe following reports were generated:")

report_files = [
    "modeling_base_missing_values.csv",
    "modeling_base_target_summary.csv",
    "flood_type_distribution.csv",
    "modeling_base_numeric_summary.csv",
    "modeling_base_categorical_summary.csv",
    "events_per_gauge.csv"
]

for report in report_files:
    report_path = REPORTS_DIR / report

    if report_path.exists():
        print(f"  ✓ {report}")


print("\n" + "=" * 70)
print("CHECK COMPLETED SUCCESSFULLY")
print("=" * 70)