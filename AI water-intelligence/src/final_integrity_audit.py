import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORT_DIR = os.path.join(BASE_DIR, "reports", "integrity_audit")

os.makedirs(REPORT_DIR, exist_ok=True)


TARGETS = {
    "peak_flood_level": "Peak Flood Level (m)",
    "peak_discharge": "Peak Discharge Q (cumec)",
    "flood_volume": "Flood Volume (cumec)",
}


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def check(condition, message):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    return {
        "Status": status,
        "Check": message
    }


results = []


print("=" * 75)
print("FINAL MODEL INTEGRITY AND LEAKAGE AUDIT")
print("=" * 75)


# =========================================================
# 1. CHECK TARGET DATASETS
# =========================================================

print("\n" + "=" * 75)
print("1. TARGET DATASET CHECK")
print("=" * 75)

target_files = {
    "peak_flood_level":
        "peak_flood_level_regression.csv",

    "peak_discharge":
        "peak_discharge_regression.csv",

    "flood_volume":
        "flood_volume_regression.csv",

    "classification":
        "flood_type_classification.csv"
}


for key, filename in target_files.items():

    path = os.path.join(PROCESSED_DIR, filename)

    exists = os.path.exists(path)

    results.append(
        check(
            exists,
            f"Target dataset exists: {filename}"
        )
    )

    if exists:
        df = pd.read_csv(path)

        print(
            f"   {filename}: "
            f"{df.shape[0]} rows × {df.shape[1]} columns"
        )


# =========================================================
# 2. TARGET LEAKAGE CHECK
# =========================================================

print("\n" + "=" * 75)
print("2. TARGET LEAKAGE CHECK")
print("=" * 75)


post_event_columns = {
    "Peak Flood Level (m)",
    "Peak FL Date",
    "Peak Discharge Q (cumec)",
    "Peak Discharge Date",
    "Flood Volume (cumec)",
    "Event Duration (days)",
    "Time to Peak (days)",
    "Recession Time (day)",
    "Num Peak FL",
    "Flood Type"
}


for target_key, target_name in TARGETS.items():

    filename = target_files[target_key]

    path = os.path.join(PROCESSED_DIR, filename)

    if not os.path.exists(path):
        continue

    df = pd.read_csv(path)

    leaked = []

    for column in post_event_columns:

        if column == target_name:
            continue

        if column in df.columns:
            leaked.append(column)

    results.append(
        check(
            len(leaked) == 0,
            f"{target_key}: no other post-event target/outcome columns"
        )
    )

    if leaked:
        print("   LEAKED COLUMNS:")
        for column in leaked:
            print(f"      - {column}")


# =========================================================
# 3. IDENTIFIER CHECK
# =========================================================

print("\n" + "=" * 75)
print("3. IDENTIFIER CHECK")
print("=" * 75)


for target_key, target_name in TARGETS.items():

    filename = target_files[target_key]

    path = os.path.join(PROCESSED_DIR, filename)

    if not os.path.exists(path):
        continue

    df = pd.read_csv(path)

    forbidden_identifiers = [
        "EventID",
        "Station",
        "River Name/ Tributory/ SubTributory"
    ]

    present = [
        column
        for column in forbidden_identifiers
        if column in df.columns
    ]

    results.append(
        check(
            len(present) == 0,
            f"{target_key}: event/station/river identifiers excluded"
        )
    )

    if present:
        print("   PRESENT:")
        for column in present:
            print(f"      - {column}")


# =========================================================
# 4. CHECK GAUGEID USAGE
# =========================================================

print("\n" + "=" * 75)
print("4. GAUGEID PREDICTOR CHECK")
print("=" * 75)


for target_key in TARGETS:

    x_train_file = os.path.join(
        SPLIT_DIR,
        f"controlled_{target_key}_X_train.csv"
    )

    x_test_file = os.path.join(
        SPLIT_DIR,
        f"controlled_{target_key}_X_test.csv"
    )

    if not os.path.exists(x_train_file):
        continue

    X_train = pd.read_csv(x_train_file)
    X_test = pd.read_csv(x_test_file)

    gauge_train = "GaugeID" in X_train.columns
    gauge_test = "GaugeID" in X_test.columns

    results.append(
        check(
            not gauge_train and not gauge_test,
            f"{target_key}: GaugeID excluded from predictors"
        )
    )


# =========================================================
# 5. TRAIN / TEST GAUGE OVERLAP
# =========================================================

print("\n" + "=" * 75)
print("5. TRAIN / TEST GAUGE SEPARATION")
print("=" * 75)


for target_key in TARGETS:

    train_group_file = os.path.join(
        SPLIT_DIR,
        f"controlled_{target_key}_groups_train.csv"
    )

    test_group_file = os.path.join(
        SPLIT_DIR,
        f"controlled_{target_key}_groups_test.csv"
    )

    if not os.path.exists(train_group_file):
        continue

    train_groups = pd.read_csv(train_group_file).iloc[:, 0]
    test_groups = pd.read_csv(test_group_file).iloc[:, 0]

    train_set = set(train_groups.dropna())
    test_set = set(test_groups.dropna())

    overlap = train_set.intersection(test_set)

    results.append(
        check(
            len(overlap) == 0,
            f"{target_key}: no GaugeID overlap between train and test"
        )
    )

    print(f"   Training gauges: {len(train_set)}")
    print(f"   Testing gauges:  {len(test_set)}")
    print(f"   Overlap:         {len(overlap)}")


# =========================================================
# 6. TRAIN / TEST ROW COUNT CONSISTENCY
# =========================================================

print("\n" + "=" * 75)
print("6. SPLIT CONSISTENCY CHECK")
print("=" * 75)


for target_key in TARGETS:

    files = {
        "X_train": f"controlled_{target_key}_X_train.csv",
        "X_test": f"controlled_{target_key}_X_test.csv",
        "y_train": f"controlled_{target_key}_y_train.csv",
        "y_test": f"controlled_{target_key}_y_test.csv",
        "groups_train": f"controlled_{target_key}_groups_train.csv",
        "groups_test": f"controlled_{target_key}_groups_test.csv",
    }

    missing = []

    for label, filename in files.items():

        path = os.path.join(SPLIT_DIR, filename)

        if not os.path.exists(path):
            missing.append(filename)

    if missing:

        results.append(
            check(
                False,
                f"{target_key}: all split files exist"
            )
        )

        continue

    X_train = pd.read_csv(
        os.path.join(SPLIT_DIR, files["X_train"])
    )

    X_test = pd.read_csv(
        os.path.join(SPLIT_DIR, files["X_test"])
    )

    y_train = pd.read_csv(
        os.path.join(SPLIT_DIR, files["y_train"])
    )

    y_test = pd.read_csv(
        os.path.join(SPLIT_DIR, files["y_test"])
    )

    groups_train = pd.read_csv(
        os.path.join(SPLIT_DIR, files["groups_train"])
    )

    groups_test = pd.read_csv(
        os.path.join(SPLIT_DIR, files["groups_test"])
    )

    consistent = (
        len(X_train) == len(y_train) == len(groups_train)
        and
        len(X_test) == len(y_test) == len(groups_test)
    )

    results.append(
        check(
            consistent,
            f"{target_key}: X/y/group row counts are consistent"
        )
    )

    print(
        f"   Train: X={len(X_train)}, "
        f"y={len(y_train)}, "
        f"groups={len(groups_train)}"
    )

    print(
        f"   Test:  X={len(X_test)}, "
        f"y={len(y_test)}, "
        f"groups={len(groups_test)}"
    )


# =========================================================
# 7. TARGET MISSING VALUE CHECK
# =========================================================

print("\n" + "=" * 75)
print("7. TARGET MISSING VALUE CHECK")
print("=" * 75)


for target_key, target_name in TARGETS.items():

    filename = target_files[target_key]

    path = os.path.join(PROCESSED_DIR, filename)

    if not os.path.exists(path):
        continue

    df = pd.read_csv(path)

    if target_name not in df.columns:
        results.append(
            check(
                False,
                f"{target_key}: target column exists"
            )
        )
        continue

    missing = df[target_name].isna().sum()

    results.append(
        check(
            missing == 0,
            f"{target_key}: target contains no missing values"
        )
    )

    print(
        f"   {target_name}: {missing} missing values"
    )


# =========================================================
# 8. DUPLICATE CHECK
# =========================================================

print("\n" + "=" * 75)
print("8. DUPLICATE EVENT CHECK")
print("=" * 75)


integrated_file = os.path.join(
    PROCESSED_DIR,
    "indofloods_integrated.csv"
)

if os.path.exists(integrated_file):

    df = pd.read_csv(integrated_file)

    if "EventID" in df.columns:

        duplicates = df["EventID"].duplicated().sum()

        results.append(
            check(
                duplicates == 0,
                "Integrated dataset contains no duplicate EventIDs"
            )
        )

        print(f"   Duplicate EventIDs: {duplicates}")


# =========================================================
# 9. CONTROLLED FEATURE REMOVAL CHECK
# =========================================================

print("\n" + "=" * 75)
print("9. CONTROLLED FEATURE CHECK")
print("=" * 75)


controlled_removed = [
    "Warning Level",
    "Danger Level",
    "Level_Entries",
    "Streamflow_Entries",
    "Reliability"
]


for target_key in TARGETS:

    path = os.path.join(
        SPLIT_DIR,
        f"controlled_{target_key}_X_train.csv"
    )

    if not os.path.exists(path):
        continue

    X = pd.read_csv(path)

    remaining = [
        column
        for column in controlled_removed
        if column in X.columns
    ]

    results.append(
        check(
            len(remaining) == 0,
            f"{target_key}: controlled quality/threshold fields removed"
        )
    )


# =========================================================
# 10. CLASSIFICATION LEAKAGE CHECK
# =========================================================

print("\n" + "=" * 75)
print("10. CLASSIFICATION LEAKAGE CHECK")
print("=" * 75)


classification_file = os.path.join(
    PROCESSED_DIR,
    "flood_type_classification.csv"
)

if os.path.exists(classification_file):

    df = pd.read_csv(classification_file)

    leakage_columns = [
        column
        for column in post_event_columns
        if column != "Flood Type"
        and column in df.columns
    ]

    results.append(
        check(
            len(leakage_columns) == 0,
            "Classification dataset excludes post-event outcome variables"
        )
    )

    if leakage_columns:

        print("   LEAKED COLUMNS:")

        for column in leakage_columns:
            print(f"      - {column}")


# =========================================================
# 11. SUMMARY
# =========================================================

print("\n" + "=" * 75)
print("AUDIT SUMMARY")
print("=" * 75)


audit_df = pd.DataFrame(results)

audit_file = os.path.join(
    REPORT_DIR,
    "final_integrity_audit.csv"
)

audit_df.to_csv(
    audit_file,
    index=False
)

print(audit_df.to_string(index=False))

pass_count = (audit_df["Status"] == "PASS").sum()
fail_count = (audit_df["Status"] == "FAIL").sum()

print("\n" + "-" * 75)
print(f"PASS: {pass_count}")
print(f"FAIL: {fail_count}")
print("-" * 75)

if fail_count == 0:

    print("\nFINAL AUDIT STATUS: ALL CHECKS PASSED")
    print("The current ML pipeline is ready for results freeze.")

else:

    print("\nFINAL AUDIT STATUS: ISSUES DETECTED")
    print("Review the failed checks before freezing the results.")


print("\nAudit report saved to:")
print(audit_file)

print("\n" + "=" * 75)
print("FINAL INTEGRITY AUDIT COMPLETED")
print("=" * 75)