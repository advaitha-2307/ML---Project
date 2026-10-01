import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

GROUP_COLUMN = "GaugeID"


DATASETS = {
    "classification": {
        "file": "flood_type_classification.csv",
        "target": "Flood Type",
    },

    "peak_flood_level": {
        "file": "peak_flood_level_regression.csv",
        "target": "Peak Flood Level (m)",
    },

    "peak_discharge": {
        "file": "peak_discharge_regression.csv",
        "target": "Peak Discharge Q (cumec)",
    },

    "flood_volume": {
        "file": "flood_volume_regression.csv",
        "target": "Flood Volume (cumec)",
    },
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("INDOFLOODS GAUGE-BASED TRAIN / TEST SPLIT")
print("=" * 70)

print(f"\nTest size      : {TEST_SIZE}")
print(f"Random state   : {RANDOM_STATE}")
print(f"Grouping       : {GROUP_COLUMN}")


# ============================================================
# FUNCTION
# ============================================================

def prepare_split(name, config):

    filename = config["file"]
    target = config["target"]

    input_path = DATA_DIR / filename

    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    print(f"\nLoading:")
    print(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{input_path}"
        )

    df = pd.read_csv(input_path)

    print(f"\nDataset shape: {df.shape}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    if target not in df.columns:
        raise ValueError(
            f"Target '{target}' not found in {filename}"
        )

    if GROUP_COLUMN not in df.columns:
        raise ValueError(
            f"Grouping column '{GROUP_COLUMN}' "
            f"not found in {filename}"
        )

    # --------------------------------------------------------
    # Remove rows with missing target
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(subset=[target]).copy()

    after = len(df)

    print(f"\nRows before target filtering: {before}")
    print(f"Rows after target filtering : {after}")

    # --------------------------------------------------------
    # X and y
    # --------------------------------------------------------

    # GaugeID is NOT an ML feature.
    # It is used only to ensure that all events from the same
    # gauge stay in either train or test.

    groups = df[GROUP_COLUMN].copy()

    y = df[target].copy()

    X = df.drop(
        columns=[target, GROUP_COLUMN]
    ).copy()

    # --------------------------------------------------------
    # Group-based split
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()

    y_train = y.iloc[train_indices].copy()
    y_test = y.iloc[test_indices].copy()

    groups_train = groups.iloc[train_indices].copy()
    groups_test = groups.iloc[test_indices].copy()

    # --------------------------------------------------------
    # Check gauge leakage
    # --------------------------------------------------------

    train_gauges = set(groups_train.unique())
    test_gauges = set(groups_test.unique())

    overlap = train_gauges.intersection(test_gauges)

    print("\nGauge split:")
    print(f"  Total gauges : {groups.nunique()}")
    print(f"  Train gauges : {groups_train.nunique()}")
    print(f"  Test gauges  : {groups_test.nunique()}")
    print(f"  Overlap      : {len(overlap)}")

    if overlap:
        raise RuntimeError(
            "Gauge leakage detected! "
            "Some gauges exist in both train and test."
        )

    print("  ✓ No gauge overlap.")

    # --------------------------------------------------------
    # Dataset sizes
    # --------------------------------------------------------

    print("\nRows:")
    print(f"  Total : {len(df)}")
    print(f"  Train : {len(X_train)}")
    print(f"  Test  : {len(X_test)}")

    # --------------------------------------------------------
    # Classification distribution
    # --------------------------------------------------------

    if name == "classification":

        print("\nFlood Type distribution:")

        train_distribution = (
            y_train
            .value_counts()
            .to_frame("Count")
        )

        train_distribution["Percentage"] = (
            train_distribution["Count"]
            / len(y_train)
            * 100
        ).round(2)

        test_distribution = (
            y_test
            .value_counts()
            .to_frame("Count")
        )

        test_distribution["Percentage"] = (
            test_distribution["Count"]
            / len(y_test)
            * 100
        ).round(2)

        print("\nTRAIN:")
        print(train_distribution)

        print("\nTEST:")
        print(test_distribution)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    train_missing = X_train.isna().sum()
    train_missing = train_missing[
        train_missing > 0
    ].sort_values(ascending=False)

    test_missing = X_test.isna().sum()
    test_missing = test_missing[
        test_missing > 0
    ].sort_values(ascending=False)

    print("\nMissing predictor values:")
    print(f"  Train columns with missing: {len(train_missing)}")
    print(f"  Test columns with missing : {len(test_missing)}")

    # --------------------------------------------------------
    # Save X / y / groups
    # --------------------------------------------------------

    prefix = name

    X_train_path = SPLIT_DIR / f"{prefix}_X_train.csv"
    X_test_path = SPLIT_DIR / f"{prefix}_X_test.csv"

    y_train_path = SPLIT_DIR / f"{prefix}_y_train.csv"
    y_test_path = SPLIT_DIR / f"{prefix}_y_test.csv"

    groups_train_path = SPLIT_DIR / f"{prefix}_groups_train.csv"
    groups_test_path = SPLIT_DIR / f"{prefix}_groups_test.csv"

    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)

    y_train.to_frame(name=target).to_csv(
        y_train_path,
        index=False
    )

    y_test.to_frame(name=target).to_csv(
        y_test_path,
        index=False
    )

    groups_train.to_frame(name=GROUP_COLUMN).to_csv(
        groups_train_path,
        index=False
    )

    groups_test.to_frame(name=GROUP_COLUMN).to_csv(
        groups_test_path,
        index=False
    )

    # --------------------------------------------------------
    # Save split summary
    # --------------------------------------------------------

    summary = pd.DataFrame([{
        "Dataset": name,
        "Target": target,
        "Total_Rows": len(df),
        "Train_Rows": len(X_train),
        "Test_Rows": len(X_test),
        "Total_Gauges": groups.nunique(),
        "Train_Gauges": groups_train.nunique(),
        "Test_Gauges": groups_test.nunique(),
        "Gauge_Overlap": len(overlap),
        "Train_Missing_Predictor_Columns": len(train_missing),
        "Test_Missing_Predictor_Columns": len(test_missing),
        "Random_State": RANDOM_STATE,
        "Test_Size": TEST_SIZE,
    }])

    summary_path = SPLIT_DIR / f"{prefix}_split_summary.csv"

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\nSaved files:")
    print(f"  ✓ {X_train_path.name}")
    print(f"  ✓ {X_test_path.name}")
    print(f"  ✓ {y_train_path.name}")
    print(f"  ✓ {y_test_path.name}")
    print(f"  ✓ {groups_train_path.name}")
    print(f"  ✓ {groups_test_path.name}")
    print(f"  ✓ {summary_path.name}")

    return summary


# ============================================================
# PROCESS ALL DATASETS
# ============================================================

all_summaries = []

for name, config in DATASETS.items():

    summary = prepare_split(
        name=name,
        config=config
    )

    all_summaries.append(summary)


# ============================================================
# COMBINED SUMMARY
# ============================================================

combined_summary = pd.concat(
    all_summaries,
    ignore_index=True
)

combined_path = SPLIT_DIR / "all_split_summary.csv"

combined_summary.to_csv(
    combined_path,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ALL GAUGE-BASED SPLITS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCombined summary:")
print(combined_summary.to_string(index=False))

print(f"\nSaved combined summary:")
print(combined_path)

print("\nImportant:")
print("GaugeID was used only for grouping.")
print("GaugeID is NOT included as an ML predictor.")
print("No missing-value imputation was performed yet.")
print("No scaling or encoding was performed yet.")

print("\nNext stage:")
print("Preprocessing pipelines + model training.")