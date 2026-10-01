import os
import joblib
import pandas as pd
import numpy as np
import shap


# ============================================================
# CONFIGURATION
# ============================================================

TARGETS = {
    "peak_flood_level": "Peak Flood Level (m)",
    "peak_discharge": "Peak Discharge Q (cumec)",
    "flood_volume": "Flood Volume (cumec)"
}

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "splits"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "regression_controlled"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "shap"
)

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# ANALYSIS
# ============================================================

for target_key, target_name in TARGETS.items():

    print("\n" + "=" * 75)
    print(f"SHAP DIRECTION ANALYSIS: {target_name}")
    print("=" * 75)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    x_test_path = os.path.join(
        SPLIT_DIR,
        f"controlled_{target_key}_X_test.csv"
    )

    x_test = pd.read_csv(x_test_path)

    # --------------------------------------------------------
    # Load Gradient Boosting model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"controlled_{target_key}_gradient_boosting.joblib"
    )

    model = joblib.load(model_path)

    print(f"Test samples: {len(x_test)}")

    # --------------------------------------------------------
    # Extract preprocessing and model
    # --------------------------------------------------------

    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]

    # --------------------------------------------------------
    # Transform features
    # --------------------------------------------------------

    X_transformed = preprocessor.transform(x_test)

    feature_names = preprocessor.get_feature_names_out()

    X_transformed = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(regressor)

    shap_values = explainer.shap_values(
        X_transformed
    )

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    records = []

    for i, feature in enumerate(feature_names):

        values = shap_values[:, i]

        mean_shap = np.mean(values)

        mean_abs_shap = np.mean(
            np.abs(values)
        )

        min_shap = np.min(values)

        max_shap = np.max(values)

        positive_count = np.sum(values > 0)

        negative_count = np.sum(values < 0)

        zero_count = np.sum(values == 0)

        total_count = len(values)

        positive_percentage = (
            positive_count /
            total_count *
            100
        )

        negative_percentage = (
            negative_count /
            total_count *
            100
        )

        records.append({
            "Feature": feature,
            "Mean_SHAP": mean_shap,
            "Mean_Absolute_SHAP": mean_abs_shap,
            "Min_SHAP": min_shap,
            "Max_SHAP": max_shap,
            "Positive_Contribution_%":
                positive_percentage,
            "Negative_Contribution_%":
                negative_percentage,
            "Zero_Contribution_%":
                zero_count /
                total_count *
                100
        })

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    direction_df = pd.DataFrame(records)

    direction_df = direction_df.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    # --------------------------------------------------------
    # Save complete analysis
    # --------------------------------------------------------

    output_path = os.path.join(
        REPORT_DIR,
        f"{target_key}_shap_direction.csv"
    )

    direction_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Display Top 20
    # --------------------------------------------------------

    print("\nTop 20 features with SHAP direction:")

    display_columns = [
        "Feature",
        "Mean_SHAP",
        "Mean_Absolute_SHAP",
        "Min_SHAP",
        "Max_SHAP",
        "Positive_Contribution_%",
        "Negative_Contribution_%"
    ]

    print(
        direction_df
        .head(20)[display_columns]
        .to_string(index=False)
    )

    print("\nSaved:")
    print(output_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("SHAP DIRECTION ANALYSIS COMPLETED")
print("=" * 75)