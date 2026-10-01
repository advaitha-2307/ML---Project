import os
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "splits"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "residuals"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "error_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


TARGETS = {
    "peak_flood_level": "Peak Flood Level (m)",
    "peak_discharge": "Peak Discharge Q (cumec)",
    "flood_volume": "Flood Volume (cumec)"
}


def analyze_target(target_key, target_name):

    print("\n" + "=" * 70)
    print(f"EXTREME ERROR ANALYSIS: {target_name}")
    print("=" * 70)

    prediction_file = os.path.join(
        REPORT_DIR,
        f"{target_key}_final_predictions.csv"
    )

    X_test_file = os.path.join(
        SPLIT_DIR,
        f"controlled_{target_key}_X_test.csv"
    )

    groups_file = os.path.join(
        SPLIT_DIR,
        f"controlled_{target_key}_groups_test.csv"
    )

    y_test_file = os.path.join(
        SPLIT_DIR,
        f"controlled_{target_key}_y_test.csv"
    )

    for file_path in [
        prediction_file,
        X_test_file,
        groups_file,
        y_test_file
    ]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File not found:\n{file_path}"
            )

    predictions = pd.read_csv(
        prediction_file
    )

    X_test = pd.read_csv(
        X_test_file
    )

    groups = pd.read_csv(
        groups_file
    )

    y_test = pd.read_csv(
        y_test_file
    )

    # --------------------------------------------------------
    # Combine information
    # --------------------------------------------------------

    result = predictions.copy()

    result["GaugeID"] = groups.iloc[:, 0].values

    result["Absolute_Error"] = (
        result["Residual"].abs()
    )

    # Add target
    result["Target"] = y_test.iloc[:, 0].values

    # Add available test features
    for column in X_test.columns:

        if column not in result.columns:

            result[column] = X_test[column].values

    # --------------------------------------------------------
    # Sort by absolute error
    # --------------------------------------------------------

    result = result.sort_values(
        "Absolute_Error",
        ascending=False
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Top 20 errors
    # --------------------------------------------------------

    top20 = result.head(20)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{target_key}_top20_errors.csv"
    )

    top20.to_csv(
        output_file,
        index=False
    )

    print("\nTop 20 largest errors:")

    columns_to_show = [
        "Actual",
        "Predicted",
        "Residual",
        "Absolute_Error",
        "GaugeID"
    ]

    print(
        top20[columns_to_show].to_string(
            index=False
        )
    )

    print(
        f"\nSaved:\n{output_file}"
    )

    # --------------------------------------------------------
    # Gauge-level error analysis
    # --------------------------------------------------------

    gauge_summary = (
        result
        .groupby("GaugeID")
        .agg(
            Events=("Absolute_Error", "count"),
            Mean_Absolute_Error=("Absolute_Error", "mean"),
            Median_Absolute_Error=("Absolute_Error", "median"),
            Max_Absolute_Error=("Absolute_Error", "max"),
            Mean_Residual=("Residual", "mean")
        )
        .sort_values(
            "Mean_Absolute_Error",
            ascending=False
        )
    )

    gauge_output = os.path.join(
        OUTPUT_DIR,
        f"{target_key}_gauge_error_summary.csv"
    )

    gauge_summary.to_csv(
        gauge_output
    )

    print("\nTop 10 gauges by mean absolute error:")

    print(
        gauge_summary.head(10).to_string()
    )

    print(
        f"\nSaved:\n{gauge_output}"
    )

    # --------------------------------------------------------
    # Error quantiles
    # --------------------------------------------------------

    quantiles = result["Absolute_Error"].quantile(
        [0.50, 0.75, 0.90, 0.95, 0.99]
    )

    print("\nAbsolute error quantiles:")

    print(
        quantiles.to_string()
    )

    # --------------------------------------------------------
    # Save full enriched error dataset
    # --------------------------------------------------------

    full_output = os.path.join(
        OUTPUT_DIR,
        f"{target_key}_full_error_analysis.csv"
    )

    result.to_csv(
        full_output,
        index=False
    )

    print(
        f"\nSaved full analysis:\n{full_output}"
    )


def main():

    print("=" * 70)
    print("EXTREME FLOOD MODEL ERROR ANALYSIS")
    print("=" * 70)

    for target_key, target_name in TARGETS.items():

        analyze_target(
            target_key,
            target_name
        )

    print("\n" + "=" * 70)
    print("EXTREME ERROR ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        f"\nResults saved in:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()