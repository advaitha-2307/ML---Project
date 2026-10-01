import pandas as pd
from pathlib import Path
import shutil

BASE = Path("data/splits")

TARGETS = [
    "peak_flood_level",
    "peak_discharge",
    "flood_volume"
]

REMOVE_COLUMNS = [
    "Warning Level",
    "Danger Level",
    "Level_Entries",
    "Streamflow_Entries",
    "Reliability"
]

for target in TARGETS:

    train_file = BASE / f"{target}_X_train.csv"
    test_file = BASE / f"{target}_X_test.csv"

    X_train = pd.read_csv(train_file)
    X_test = pd.read_csv(test_file)

    existing_remove = [
        col for col in REMOVE_COLUMNS
        if col in X_train.columns
    ]

    X_train_controlled = X_train.drop(columns=existing_remove)
    X_test_controlled = X_test.drop(columns=existing_remove)

    X_train_controlled.to_csv(
        BASE / f"controlled_{target}_X_train.csv",
        index=False
    )

    X_test_controlled.to_csv(
        BASE / f"controlled_{target}_X_test.csv",
        index=False
    )

    # Copy targets and groups unchanged
    shutil.copy(
        BASE / f"{target}_y_train.csv",
        BASE / f"controlled_{target}_y_train.csv"
    )

    shutil.copy(
        BASE / f"{target}_y_test.csv",
        BASE / f"controlled_{target}_y_test.csv"
    )

    shutil.copy(
        BASE / f"{target}_groups_train.csv",
        BASE / f"controlled_{target}_groups_train.csv"
    )

    shutil.copy(
        BASE / f"{target}_groups_test.csv",
        BASE / f"controlled_{target}_groups_test.csv"
    )

    print(f"\n{target}")
    print("Removed:", existing_remove)
    print("Original features:", X_train.shape[1])
    print("Controlled features:", X_train_controlled.shape[1])

print("\nControlled splits created successfully.")