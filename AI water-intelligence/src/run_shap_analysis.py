import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

TARGETS = {
    "peak_flood_level": "Peak Flood Level (m)",
    "peak_discharge": "Peak Discharge Q (cumec)",
    "flood_volume": "Flood Volume (cumec)"
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")
MODEL_DIR = os.path.join(BASE_DIR, "models", "regression_controlled")

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "shap"
)

PLOT_DIR = os.path.join(
    REPORT_DIR,
    "plots"
)

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# SHAP ANALYSIS
# ============================================================

for target_key, target_name in TARGETS.items():

    print("\n" + "=" * 70)
    print(f"SHAP ANALYSIS: {target_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load controlled test data
    # --------------------------------------------------------

    x_test_path = os.path.join(
        SPLIT_DIR,
        f"controlled_{target_key}_X_test.csv"
    )

    if not os.path.exists(x_test_path):
        raise FileNotFoundError(
            f"Test file not found:\n{x_test_path}"
        )

    x_test = pd.read_csv(x_test_path)

    print(f"Test samples: {len(x_test)}")
    print(f"Original features: {x_test.shape[1]}")

    # --------------------------------------------------------
    # Load controlled Gradient Boosting model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"controlled_{target_key}_gradient_boosting.joblib"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Gradient Boosting model not found:\n{model_path}"
        )

    model = joblib.load(model_path)

    print(f"Model loaded: {model_path}")

    # --------------------------------------------------------
    # Extract preprocessing and model
    # --------------------------------------------------------

    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]

    # --------------------------------------------------------
    # Transform test data
    # --------------------------------------------------------

    X_transformed = preprocessor.transform(x_test)

    feature_names = preprocessor.get_feature_names_out()

    X_transformed = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=x_test.index
    )

    print(
        f"Transformed features: {X_transformed.shape[1]}"
    )

    # --------------------------------------------------------
    # SHAP TreeExplainer
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(regressor)

    shap_values = explainer.shap_values(
        X_transformed
    )

    # --------------------------------------------------------
    # Mean absolute SHAP importance
    # --------------------------------------------------------

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_abs_shap
    })

    importance_df = importance_df.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    # --------------------------------------------------------
    # Save importance table
    # --------------------------------------------------------

    importance_path = os.path.join(
        REPORT_DIR,
        f"{target_key}_shap_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    # --------------------------------------------------------
    # Display Top 20
    # --------------------------------------------------------

    print("\nTop 20 SHAP features:")

    print(
        importance_df
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # SHAP Summary Plot
    # --------------------------------------------------------

    shap.summary_plot(
        shap_values,
        X_transformed,
        show=False,
        max_display=20
    )

    plt.title(
        f"SHAP Feature Importance - {target_name}"
    )

    plt.tight_layout()

    summary_path = os.path.join(
        PLOT_DIR,
        f"{target_key}_shap_summary.png"
    )

    plt.savefig(
        summary_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # SHAP Bar Plot
    # --------------------------------------------------------

    shap.summary_plot(
        shap_values,
        X_transformed,
        plot_type="bar",
        show=False,
        max_display=20
    )

    plt.title(
        f"Mean Absolute SHAP Values - {target_name}"
    )

    plt.tight_layout()

    bar_path = os.path.join(
        PLOT_DIR,
        f"{target_key}_shap_bar.png"
    )

    plt.savefig(
        bar_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSaved:")
    print(f"  Importance: {importance_path}")
    print(f"  Summary:    {summary_path}")
    print(f"  Bar plot:   {bar_path}")


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETED")
print("=" * 70)