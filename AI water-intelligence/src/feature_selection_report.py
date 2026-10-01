import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/indofloods_integrated.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

print("Dataset shape:", df.shape)

# ============================================================
# 1. TARGET VARIABLES
# ============================================================

targets = [
    "Flood Type",
    "Peak Flood Level (m)",
    "Peak Discharge Q (cumec)",
    "Flood Volume (cumec)"
]

# ============================================================
# 2. IDENTIFIERS
# ============================================================

identifiers = [
    "EventID",
    "GaugeID",
    "Station",
    "River Name/ Tributory/ SubTributory"
]

# ============================================================
# 3. DATE / TIME COLUMNS
# ============================================================

date_columns = [
    col for col in df.columns
    if "date" in col.lower()
    or "start_date" in col.lower()
    or "end_date" in col.lower()
]

# ============================================================
# 4. POSSIBLE POST-EVENT / LEAKAGE FEATURES
# ============================================================

leakage_columns = [
    "Peak Flood Level (m)",
    "Peak FL Date",
    "Num Peak FL",
    "Peak Discharge Q (cumec)",
    "Peak Discharge Date",
    "Flood Volume (cumec)",
    "Event Duration (days)",
    "Time to Peak (days)",
    "Recession Time (day)"
]

# ============================================================
# 5. MISSINGNESS
# ============================================================

missing_percent = df.isna().mean() * 100

high_missing = missing_percent[missing_percent >= 80].index.tolist()

# ============================================================
# 6. CONSTANT COLUMNS
# ============================================================

constant_columns = [
    col for col in df.columns
    if df[col].nunique(dropna=True) <= 1
]

# ============================================================
# 7. BUILD REPORT
# ============================================================

report = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str).values,
    "missing_percent": [
        missing_percent[col]
        for col in df.columns
    ],
    "unique_values": [
        df[col].nunique(dropna=True)
        for col in df.columns
    ]
})

def classify_column(col):

    if col in targets:
        return "TARGET"

    if col in identifiers:
        return "IDENTIFIER"

    if col in date_columns:
        return "DATE_TIME"

    if col in leakage_columns:
        return "POSSIBLE_LEAKAGE"

    if col in constant_columns:
        return "CONSTANT"

    if col in high_missing:
        return "HIGH_MISSINGNESS"

    return "POTENTIAL_FEATURE"


report["category"] = report["column"].apply(classify_column)

# ============================================================
# 8. SAVE REPORT
# ============================================================

output = REPORT_DIR / "feature_selection_report.csv"

report.to_csv(output, index=False)

print("\nFeature selection report saved to:")
print(output)

# ============================================================
# 9. PRINT GROUPS
# ============================================================

for category in [
    "TARGET",
    "IDENTIFIER",
    "DATE_TIME",
    "POSSIBLE_LEAKAGE",
    "CONSTANT",
    "HIGH_MISSINGNESS",
    "POTENTIAL_FEATURE"
]:

    cols = report.loc[
        report["category"] == category,
        "column"
    ].tolist()

    print("\n" + "=" * 60)
    print(category)
    print("=" * 60)

    for col in cols:
        print(col)

    print("Count:", len(cols))