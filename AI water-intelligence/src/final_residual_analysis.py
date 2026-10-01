# ============================================================
# FINAL RESIDUAL ANALYSIS
# AI-Powered Water Intelligence and Disaster Resilience Platform
# INDOFLOODS Dataset
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")
MODEL_DIR = os.path.join(BASE_DIR, "models", "regression_controlled")
REPORT_DIR = os.path.join(BASE_DIR, "reports", "residuals")

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# TARGET CONFIGURATION
# ============================================================

TARGETS = {
    "peak_flood_level": {
        "display_name": "Peak Flood Level (m)"
    },
    "peak_discharge": {
        "display_name": "Peak Discharge Q (cumec)"
    },
    "flood_volume": {
        "display_name": "Flood Volume (cumec)"
    }
}


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "gradient_boosting"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate basic regression metrics.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = y_true - y_pred

    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - (ss_res / ss_tot)

    mean_error = np.mean(residuals)
    median_error = np.median(residuals)

    std_residual = np.std(residuals)

    max_positive_error = np.max(residuals)
    max_negative_error = np.min(residuals)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Mean_Error": mean_error,
        "Median_Error": median_error,
        "Residual_STD": std_residual,
        "Max_Positive_Error": max_positive_error,
        "Max_Negative_Error": max_negative_error
    }


def save_actual_vs_predicted_plot(
    y_true,
    y_pred,
    target_name,
    output_path
):
    """
    Generate Actual vs Predicted plot.
    """

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.6
    )

    # Perfect prediction line
    min_value = min(np.min(y_true), np.min(y_pred))
    max_value = max(np.max(y_true), np.max(y_pred))

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.title(
        f"Actual vs Predicted - {target_name}"
    )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_residual_vs_predicted_plot(
    y_pred,
    residuals,
    target_name,
    output_path
):
    """
    Generate Residual vs Predicted plot.
    """

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.6
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Residual")

    plt.title(
        f"Residuals vs Predicted - {target_name}"
    )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_residual_distribution_plot(
    residuals,
    target_name,
    output_path
):
    """
    Generate residual distribution histogram.
    """

    plt.figure(figsize=(8, 6))

    plt.hist(
        residuals,
        bins=30,
        alpha=0.75,
        edgecolor="black"
    )

    plt.axvline(
        x=0,
        linestyle="--"
    )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")

    plt.title(
        f"Residual Distribution - {target_name}"
    )

    plt.grid(
        True,
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 70)
    print("FINAL RESIDUAL ANALYSIS")
    print("Controlled Gradient Boosting Regression Models")
    print("=" * 70)

    all_results = []

    for target_key, config in TARGETS.items():

        target_name = config["display_name"]

        print("\n" + "=" * 70)
        print(f"TARGET: {target_name}")
        print("=" * 70)

        # ----------------------------------------------------
        # File paths
        # ----------------------------------------------------

        x_test_path = os.path.join(
            SPLIT_DIR,
            f"controlled_{target_key}_X_test.csv"
        )

        y_test_path = os.path.join(
            SPLIT_DIR,
            f"controlled_{target_key}_y_test.csv"
        )

        model_path = os.path.join(
            MODEL_DIR,
            f"controlled_{target_key}_{MODEL_NAME}.joblib"
        )

        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        required_files = [
            x_test_path,
            y_test_path,
            model_path
        ]

        for file_path in required_files:

            if not os.path.exists(file_path):

                raise FileNotFoundError(
                    f"\nRequired file not found:\n{file_path}"
                )

        # ----------------------------------------------------
        # Load test data
        # ----------------------------------------------------

        X_test = pd.read_csv(
            x_test_path
        )

        y_test_df = pd.read_csv(
            y_test_path
        )

        # y_test contains one target column
        y_true = y_test_df.iloc[:, 0].values

        print(f"Test samples: {len(y_true)}")
        print(f"Test features: {X_test.shape[1]}")

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model = joblib.load(
            model_path
        )

        print(
            f"Model loaded: {MODEL_NAME}"
        )

        # ----------------------------------------------------
        # Generate predictions
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # Calculate residuals
        #
        # Residual = Actual - Predicted
        # ----------------------------------------------------

        residuals = y_true - y_pred

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            y_true,
            y_pred
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("\nPerformance:")
        print(
            f"MAE  : {metrics['MAE']:.4f}"
        )

        print(
            f"RMSE : {metrics['RMSE']:.4f}"
        )

        print(
            f"R²   : {metrics['R2']:.4f}"
        )

        print("\nResidual statistics:")

        print(
            f"Mean Error   : {metrics['Mean_Error']:.4f}"
        )

        print(
            f"Median Error : {metrics['Median_Error']:.4f}"
        )

        print(
            f"Residual STD : {metrics['Residual_STD']:.4f}"
        )

        print(
            f"Maximum Positive Error : "
            f"{metrics['Max_Positive_Error']:.4f}"
        )

        print(
            f"Maximum Negative Error : "
            f"{metrics['Max_Negative_Error']:.4f}"
        )

        # ----------------------------------------------------
        # Save prediction table
        # ----------------------------------------------------

        prediction_df = pd.DataFrame({
            "Actual": y_true,
            "Predicted": y_pred,
            "Residual": residuals,
            "Absolute_Error": np.abs(residuals)
        })

        prediction_output = os.path.join(
            REPORT_DIR,
            f"{target_key}_final_predictions.csv"
        )

        prediction_df.to_csv(
            prediction_output,
            index=False
        )

        print(
            f"\nSaved predictions:\n{prediction_output}"
        )

        # ----------------------------------------------------
        # Actual vs Predicted plot
        # ----------------------------------------------------

        actual_predicted_path = os.path.join(
            REPORT_DIR,
            f"{target_key}_actual_vs_predicted.png"
        )

        save_actual_vs_predicted_plot(
            y_true,
            y_pred,
            target_name,
            actual_predicted_path
        )

        print(
            f"Saved:\n{actual_predicted_path}"
        )

        # ----------------------------------------------------
        # Residual vs Predicted plot
        # ----------------------------------------------------

        residual_predicted_path = os.path.join(
            REPORT_DIR,
            f"{target_key}_residuals_vs_predicted.png"
        )

        save_residual_vs_predicted_plot(
            y_pred,
            residuals,
            target_name,
            residual_predicted_path
        )

        print(
            f"Saved:\n{residual_predicted_path}"
        )

        # ----------------------------------------------------
        # Residual distribution
        # ----------------------------------------------------

        residual_distribution_path = os.path.join(
            REPORT_DIR,
            f"{target_key}_residual_distribution.png"
        )

        save_residual_distribution_plot(
            residuals,
            target_name,
            residual_distribution_path
        )

        print(
            f"Saved:\n{residual_distribution_path}"
        )

        # ----------------------------------------------------
        # Store final result
        # ----------------------------------------------------

        result_row = {
            "Target": target_name,
            "Target_Key": target_key,
            "Model": "Controlled Gradient Boosting",
            "Test_Samples": len(y_true),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "Mean_Error": metrics["Mean_Error"],
            "Median_Error": metrics["Median_Error"],
            "Residual_STD": metrics["Residual_STD"],
            "Max_Positive_Error": metrics["Max_Positive_Error"],
            "Max_Negative_Error": metrics["Max_Negative_Error"]
        }

        all_results.append(
            result_row
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        all_results
    )

    summary_path = os.path.join(
        REPORT_DIR,
        "final_residual_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL RESIDUAL ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        f"\nAll results saved in:\n{REPORT_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()