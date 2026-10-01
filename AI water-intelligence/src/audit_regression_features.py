import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SPLIT_DIR = "data/splits"


TARGETS = {
    "peak_flood_level": "Peak Flood Level (m)",
    "peak_discharge": "Peak Discharge Q (cumec)",
    "flood_volume": "Flood Volume (cumec)"
}


# Features that should never be predictors for these
# target-specific regression experiments.
OUTCOME_COLUMNS = [
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
]


# ============================================================
# HELPER
# ============================================================

def classify_feature(feature):
    """
    Give a simple research-audit category to each feature.
    """

    feature_lower = feature.lower()

    # Identifiers
    if feature in ["GaugeID", "EventID"]:
        return "IDENTIFIER"

    # Outcome / post-event information
    if feature in OUTCOME_COLUMNS:
        return "OUTCOME / POST-EVENT"

    # Warning and danger thresholds
    if "warning level" in feature_lower:
        return "GAUGE THRESHOLD"

    if "danger level" in feature_lower:
        return "GAUGE THRESHOLD"

    # Dates generated during preprocessing
    if (
        "year" in feature_lower
        or "month" in feature_lower
        or "dayofyear" in feature_lower
        or "season" in feature_lower
    ):
        return "TEMPORAL FEATURE"

    # Precipitation
    if feature_lower.startswith("t") and feature_lower[1:].isdigit():
        return "PRECIPITATION"

    # Metadata
    metadata_keywords = [
        "station",
        "river",
        "tributory",
        "subtributory",
        "basin",
        "state",
        "latitude",
        "longitude",
        "reliability",
        "privacy"
    ]

    if any(keyword in feature_lower for keyword in metadata_keywords):
        return "METADATA"

    # Catchment / physical characteristics
    catchment_keywords = [
        "drainage",
        "catchment",
        "stream",
        "bifurcation",
        "relief",
        "slope",
        "length",
        "area",
        "magnitude",
        "density",
        "texture",
        "frequency",
        "shape",
        "elongation",
        "form",
        "lithology",
        "soil",
        "koppen",
        "land cover",
        "landcover"
    ]

    if any(keyword in feature_lower for keyword in catchment_keywords):
        return "CATCHMENT / PHYSICAL"

    return "OTHER"


# ============================================================
# MAIN AUDIT
# ============================================================

print("=" * 80)
print("REGRESSION FEATURE AUDIT")
print("AI-POWERED WATER INTELLIGENCE AND DISASTER RESILIENCE PLATFORM")
print("=" * 80)


for target_key, target_name in TARGETS.items():

    print("\n")
    print("=" * 80)
    print(f"TARGET: {target_name}")
    print("=" * 80)

    X_train_path = os.path.join(
        SPLIT_DIR,
        f"{target_key}_X_train.csv"
    )

    X_test_path = os.path.join(
        SPLIT_DIR,
        f"{target_key}_X_test.csv"
    )

    if not os.path.exists(X_train_path):
        print(f"ERROR: File not found: {X_train_path}")
        continue

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")

    features = X_train.columns.tolist()

    print(f"\nTotal predictor columns: {len(features)}")

    # --------------------------------------------------------
    # Categorize features
    # --------------------------------------------------------

    audit_rows = []

    for feature in features:

        category = classify_feature(feature)

        missing_train = X_train[feature].isna().sum()
        missing_test = X_test[feature].isna().sum()

        missing_train_pct = (
            missing_train / len(X_train) * 100
        )

        missing_test_pct = (
            missing_test / len(X_test) * 100
        )

        unique_values = X_train[feature].nunique(
            dropna=True
        )

        audit_rows.append(
            {
                "Feature": feature,
                "Category": category,
                "Data_Type": str(X_train[feature].dtype),
                "Unique_Values": unique_values,
                "Train_Missing_%": round(
                    missing_train_pct,
                    2
                ),
                "Test_Missing_%": round(
                    missing_test_pct,
                    2
                )
            }
        )

    audit_df = pd.DataFrame(audit_rows)

    # --------------------------------------------------------
    # Print category summary
    # --------------------------------------------------------

    print("\nFEATURE CATEGORIES")
    print("-" * 80)

    category_counts = (
        audit_df["Category"]
        .value_counts()
    )

    print(category_counts.to_string())

    # --------------------------------------------------------
    # Print potentially sensitive features
    # --------------------------------------------------------

    print("\n\nFEATURES REQUIRING SPECIAL REVIEW")
    print("-" * 80)

    review_categories = [
        "IDENTIFIER",
        "GAUGE THRESHOLD",
        "TEMPORAL FEATURE",
        "METADATA"
    ]

    review_df = audit_df[
        audit_df["Category"].isin(
            review_categories
        )
    ]

    if len(review_df) > 0:
        print(
            review_df[
                [
                    "Feature",
                    "Category",
                    "Data_Type",
                    "Unique_Values",
                    "Train_Missing_%",
                    "Test_Missing_%"
                ]
            ].to_string(index=False)
        )
    else:
        print("No special-review features found.")

    # --------------------------------------------------------
    # Check outcome columns accidentally present
    # --------------------------------------------------------

    outcome_present = [
        column
        for column in OUTCOME_COLUMNS
        if column in features
    ]

    print("\n\nOUTCOME / POST-EVENT COLUMNS PRESENT")
    print("-" * 80)

    if outcome_present:
        for column in outcome_present:
            print(f"WARNING: {column}")
    else:
        print("None")

    # --------------------------------------------------------
    # Check GaugeID
    # --------------------------------------------------------

    print("\n\nGAUGE GROUPING CHECK")
    print("-" * 80)

    if "GaugeID" in features:

        print(
            "GaugeID is present in X."
        )

        print(
            "IMPORTANT: GaugeID should be used only "
            "for group-based splitting, not as an ML predictor."
        )

    else:

        print(
            "GaugeID is not present in X."
        )

    # --------------------------------------------------------
    # Save audit
    # --------------------------------------------------------

    output_path = os.path.join(
        "reports",
        f"{target_key}_feature_audit.csv"
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    audit_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nFull audit saved to: {output_path}"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 80)
print("FEATURE AUDIT COMPLETED")
print("=" * 80)

print(
    "\nReview the three generated CSV files in the reports folder."
)

print(
    "\nDo NOT remove features yet. "
    "We will decide what to retain/remove after reviewing the audit."
)