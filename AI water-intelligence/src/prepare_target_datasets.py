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
    / "indofloods_modeling_base.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PREPARING TARGET-SPECIFIC MODELING DATASETS")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print(f"\nInput dataset shape: {df.shape}")


# ============================================================
# TARGETS
# ============================================================

TARGET_CLASSIFICATION = "Flood Type"

TARGET_LEVEL = "Peak Flood Level (m)"

TARGET_DISCHARGE = "Peak Discharge Q (cumec)"

TARGET_VOLUME = "Flood Volume (cumec)"


# ============================================================
# IDENTIFIERS / GROUPING
# ============================================================

# GaugeID is retained because we will use it later for
# Group-based train/test splitting.
#
# It will NOT be used as an ML predictor.

GROUP_COLUMN = "GaugeID"

NON_PREDICTOR_COLUMNS = [
    "EventID",
    "GaugeID",
    "Station",
    "River Name/ Tributory/ SubTributory",
]


# ============================================================
# POST-EVENT / OUTCOME VARIABLES
# ============================================================

POST_EVENT_COLUMNS = [
    "Peak Flood Level (m)",
    "Peak FL Date",
    "Peak Discharge Q (cumec)",
    "Peak Discharge Date",
    "Flood Volume (cumec)",
    "Event Duration (days)",
    "Time to Peak (days)",
    "Recession Time (day)",
    "Num Peak FL",
]


# ============================================================
# FUNCTION TO PREPARE DATASET
# ============================================================

def create_target_dataset(
    data,
    target,
    output_filename,
    remove_columns
):

    print("\n" + "=" * 70)
    print(f"CREATING DATASET FOR: {target}")
    print("=" * 70)

    dataset = data.copy()

    # --------------------------------------------------------
    # Remove specified leakage/outcome variables
    # --------------------------------------------------------

    columns_to_remove = [
        column
        for column in remove_columns
        if column in dataset.columns
        and column != target
    ]

    # Remove identifiers from ML predictors.
    # GaugeID is also removed from predictors but retained
    # separately as a grouping column.
    predictor_identifier_columns = [
        column
        for column in NON_PREDICTOR_COLUMNS
        if column in dataset.columns
        and column != GROUP_COLUMN
        and column != target
    ]

    columns_to_remove += predictor_identifier_columns

    columns_to_remove = list(dict.fromkeys(columns_to_remove))

    print("\nColumns removed:")

    for column in columns_to_remove:
        print(f"  - {column}")

    dataset = dataset.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Keep only rows with a known target
    # --------------------------------------------------------

    before_rows = len(dataset)

    dataset = dataset.dropna(subset=[target])

    after_rows = len(dataset)

    print(f"\nRows before target filtering: {before_rows}")
    print(f"Rows after target filtering : {after_rows}")
    print(f"Rows removed                : {before_rows - after_rows}")

    # --------------------------------------------------------
    # Put target first
    # --------------------------------------------------------

    columns = dataset.columns.tolist()

    ordered_columns = []

    # Group column first
    if GROUP_COLUMN in columns:
        ordered_columns.append(GROUP_COLUMN)

    # Target second
    ordered_columns.append(target)

    # Remaining features
    ordered_columns += [
        column
        for column in columns
        if column not in ordered_columns
    ]

    dataset = dataset[ordered_columns]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / output_filename

    dataset.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved:")
    print(output_path)

    print(f"Final shape: {dataset.shape}")

    print(
        f"Missing values: "
        f"{dataset.isna().sum().sum():,}"
    )

    print(
        f"Number of features excluding target/group: "
        f"{len(dataset.columns) - 2}"
    )

    return dataset


# ============================================================
# 1. FLOOD TYPE CLASSIFICATION
# ============================================================

# For Flood Type prediction, all event outcomes must be removed.
# We retain environmental/catchment/rainfall information.

classification_df = create_target_dataset(
    data=df,
    target=TARGET_CLASSIFICATION,
    output_filename="flood_type_classification.csv",
    remove_columns=POST_EVENT_COLUMNS
)


# ============================================================
# 2. PEAK FLOOD LEVEL REGRESSION
# ============================================================

# Peak Flood Level is the target.
# All other event outcomes are removed.

level_remove_columns = [
    "Peak FL Date",
    "Peak Discharge Q (cumec)",
    "Peak Discharge Date",
    "Flood Volume (cumec)",
    "Event Duration (days)",
    "Time to Peak (days)",
    "Recession Time (day)",
    "Num Peak FL",
    "Flood Type",
]

level_df = create_target_dataset(
    data=df,
    target=TARGET_LEVEL,
    output_filename="peak_flood_level_regression.csv",
    remove_columns=level_remove_columns
)


# ============================================================
# 3. PEAK DISCHARGE REGRESSION
# ============================================================

discharge_remove_columns = [
    "Peak Flood Level (m)",
    "Peak FL Date",
    "Peak Discharge Date",
    "Flood Volume (cumec)",
    "Event Duration (days)",
    "Time to Peak (days)",
    "Recession Time (day)",
    "Num Peak FL",
    "Flood Type",
]

discharge_df = create_target_dataset(
    data=df,
    target=TARGET_DISCHARGE,
    output_filename="peak_discharge_regression.csv",
    remove_columns=discharge_remove_columns
)


# ============================================================
# 4. FLOOD VOLUME REGRESSION
# ============================================================

volume_remove_columns = [
    "Peak Flood Level (m)",
    "Peak FL Date",
    "Peak Discharge Q (cumec)",
    "Peak Discharge Date",
    "Event Duration (days)",
    "Time to Peak (days)",
    "Recession Time (day)",
    "Num Peak FL",
    "Flood Type",
]

volume_df = create_target_dataset(
    data=df,
    target=TARGET_VOLUME,
    output_filename="flood_volume_regression.csv",
    remove_columns=volume_remove_columns
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

datasets = {
    "Flood Type Classification": classification_df,
    "Peak Flood Level Regression": level_df,
    "Peak Discharge Regression": discharge_df,
    "Flood Volume Regression": volume_df,
}

for name, dataset in datasets.items():

    target = dataset.columns[1]

    print(f"\n{name}")
    print(f"  Target       : {target}")
    print(f"  Rows         : {len(dataset)}")
    print(f"  Columns      : {len(dataset.columns)}")
    print(
        f"  Missing      : "
        f"{dataset.isna().sum().sum():,}"
    )

print("\n" + "=" * 70)
print("ALL TARGET DATASETS CREATED SUCCESSFULLY")
print("=" * 70)