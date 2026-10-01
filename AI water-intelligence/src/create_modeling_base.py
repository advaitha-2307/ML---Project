import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "indofloods_integrated.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "indofloods_modeling_base.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CREATING INDOFLOODS MODELING BASE DATASET")
print("=" * 70)

print("\nInput file:")
print(INPUT_PATH)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"\nInput dataset not found:\n{INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

print(f"\nOriginal dataset shape: {df.shape}")


# Keep original columns for reporting
original_columns = df.columns.tolist()


# ============================================================
# 1. REMOVE CONFIRMED CONSTANT / USELESS FEATURES
# ============================================================

constant_columns = [
    "No. of Seventhorder Streams",
    "No. of Eigthorder Streams",
    "Eighthorder Streams Length",
    "Eighthorder Streams Mean Length",
    "SeventhEighth Stream Length Ratio",
    "SeventhEighth Bifurcation Ratio",
    "Privacy"
]

constant_columns = [
    col for col in constant_columns
    if col in df.columns
]

print("\n" + "=" * 70)
print("1. REMOVING CONSTANT / USELESS FEATURES")
print("=" * 70)

for col in constant_columns:
    print(f"  Removing: {col}")

df = df.drop(columns=constant_columns)


# ============================================================
# 2. REMOVE EXTREMELY SPARSE FEATURES
# ============================================================

# Remove features with >= 80% missing values.
# Missing target values are NOT removed here.

missing_percentage = df.isna().mean() * 100

sparse_columns = missing_percentage[
    missing_percentage >= 80
].index.tolist()

print("\n" + "=" * 70)
print("2. REMOVING EXTREMELY SPARSE FEATURES")
print("=" * 70)

if sparse_columns:
    for col in sparse_columns:
        print(
            f"  Removing: {col} "
            f"({missing_percentage[col]:.2f}% missing)"
        )

    df = df.drop(columns=sparse_columns)

else:
    print("  No features with >=80% missing values found.")


# ============================================================
# 3. CREATE DATE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("3. CREATING DATE-DERIVED FEATURES")
print("=" * 70)

# Main event start date
if "Start Date" in df.columns:

    start_date = pd.to_datetime(
        df["Start Date"],
        errors="coerce"
    )

    df["Event_Start_Year"] = start_date.dt.year
    df["Event_Start_Month"] = start_date.dt.month
    df["Event_Start_DayOfYear"] = start_date.dt.dayofyear

    # Meteorological season
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Pre-Monsoon"
        elif month in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Post-Monsoon"

    df["Event_Season"] = start_date.dt.month.map(
        get_season
    )

    print("  Created:")
    print("    - Event_Start_Year")
    print("    - Event_Start_Month")
    print("    - Event_Start_DayOfYear")
    print("    - Event_Season")

else:
    print("  WARNING: Start Date not found.")


# ============================================================
# 4. REMOVE RAW DATE COLUMNS
# ============================================================

raw_date_columns = [
    "Start Date",
    "End Date",
    "Peak FL Date",
    "Peak Discharge Date",
    "Start_date",
    "End_date"
]

raw_date_columns = [
    col for col in raw_date_columns
    if col in df.columns
]

print("\n" + "=" * 70)
print("4. REMOVING RAW DATE COLUMNS")
print("=" * 70)

for col in raw_date_columns:
    print(f"  Removing: {col}")

df = df.drop(columns=raw_date_columns)


# ============================================================
# 5. REMOVE IDENTIFIERS FROM ML FEATURES
# ============================================================

# GaugeID is deliberately retained because it will be used
# later for grouped train/test splitting.
#
# It is NOT intended to be used as an ML predictor.

identifier_columns = [
    "EventID",
    "Station",
    "River Name/ Tributory/ SubTributory"
]

identifier_columns = [
    col for col in identifier_columns
    if col in df.columns
]

print("\n" + "=" * 70)
print("5. REMOVING IDENTIFIER COLUMNS")
print("=" * 70)

for col in identifier_columns:
    print(f"  Removing: {col}")

df = df.drop(columns=identifier_columns)


# ============================================================
# 6. SHOW POSSIBLE POST-EVENT VARIABLES
# ============================================================

post_event_columns = [
    "Peak Flood Level (m)",
    "Peak Discharge Q (cumec)",
    "Flood Volume (cumec)",
    "Event Duration (days)",
    "Time to Peak (days)",
    "Recession Time (day)",
    "Num Peak FL"
]

post_event_columns = [
    col for col in post_event_columns
    if col in df.columns
]

print("\n" + "=" * 70)
print("6. POST-EVENT VARIABLES RETAINED FOR TARGET-SPECIFIC REMOVAL")
print("=" * 70)

for col in post_event_columns:
    print(f"  - {col}")


# ============================================================
# 7. TARGET COLUMNS
# ============================================================

target_columns = [
    "Flood Type",
    "Peak Flood Level (m)",
    "Peak Discharge Q (cumec)",
    "Flood Volume (cumec)"
]

target_columns = [
    col for col in target_columns
    if col in df.columns
]

print("\n" + "=" * 70)
print("7. TARGET COLUMNS")
print("=" * 70)

for col in target_columns:
    print(f"  - {col}")


# ============================================================
# 8. FINAL COLUMN ORDER
# ============================================================

# Put identifiers/grouping columns first
priority_columns = []

if "EventID" in df.columns:
    priority_columns.append("EventID")

if "GaugeID" in df.columns:
    priority_columns.append("GaugeID")

# Then targets
priority_columns += [
    col for col in target_columns
    if col not in priority_columns
]

# Then remaining features
remaining_columns = [
    col for col in df.columns
    if col not in priority_columns
]

df = df[
    priority_columns + remaining_columns
]


# ============================================================
# 9. SAVE DATASET
# ============================================================

print("\n" + "=" * 70)
print("8. SAVING MODELING BASE DATASET")
print("=" * 70)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nSaved to:")
print(OUTPUT_PATH)

print(f"\nFinal dataset shape: {df.shape}")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\nOriginal rows     : {len(pd.read_csv(INPUT_PATH))}")
print(f"Final rows        : {len(df)}")
print(f"Original columns  : {len(original_columns)}")
print(f"Final columns     : {len(df.columns)}")

print(
    f"\nTotal missing values: "
    f"{df.isna().sum().sum():,}"
)

print(
    f"Columns with missing values: "
    f"{(df.isna().sum() > 0).sum()}"
)

print(
    f"Unique GaugeIDs: "
    f"{df['GaugeID'].nunique() if 'GaugeID' in df.columns else 'N/A'}"
)

print("\nFinal targets present:")

for target in target_columns:
    print(f"  ✓ {target}")

print("\n" + "=" * 70)
print("MODELING BASE DATASET CREATED SUCCESSFULLY")
print("=" * 70)