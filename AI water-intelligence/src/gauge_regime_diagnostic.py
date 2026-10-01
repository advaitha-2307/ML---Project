import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ERROR_DIR = os.path.join(BASE_DIR, "reports", "error_analysis")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "gauge_diagnostics")

os.makedirs(OUTPUT_DIR, exist_ok=True)


TARGETS = {
    "peak_flood_level": "Peak Flood Level (m)",
    "peak_discharge": "Peak Discharge Q (cumec)",
    "flood_volume": "Flood Volume (cumec)"
}


def analyze_target(target_key, target_name):

    print("\n" + "=" * 75)
    print(f"GAUGE / REGIME DIAGNOSTIC: {target_name}")
    print("=" * 75)

    input_file = os.path.join(
        ERROR_DIR,
        f"{target_key}_full_error_analysis.csv"
    )

    if not os.path.exists(input_file):
        raise FileNotFoundError(input_file)

    df = pd.read_csv(input_file)

    print(f"\nLoaded: {input_file}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # 1. Error direction
    # ---------------------------------------------------------

    df["Error_Type"] = np.where(
        df["Residual"] > 0,
        "Underprediction",
        "Overprediction"
    )

    direction_summary = (
        df["Error_Type"]
        .value_counts()
        .rename_axis("Error_Type")
        .reset_index(name="Count")
    )

    direction_summary["Percentage"] = (
        direction_summary["Count"] / len(df) * 100
    )

    direction_file = os.path.join(
        OUTPUT_DIR,
        f"{target_key}_error_direction.csv"
    )

    direction_summary.to_csv(direction_file, index=False)

    print("\nError direction:")
    print(direction_summary.to_string(index=False))

    # ---------------------------------------------------------
    # 2. Gauge-level diagnostic
    # ---------------------------------------------------------

    gauge_summary = (
        df.groupby("GaugeID")
        .agg(
            Events=("Absolute_Error", "count"),
            Mean_Absolute_Error=("Absolute_Error", "mean"),
            Median_Absolute_Error=("Absolute_Error", "median"),
            Max_Absolute_Error=("Absolute_Error", "max"),
            Mean_Residual=("Residual", "mean"),
            Mean_Target=("Target", "mean"),
            Max_Target=("Target", "max"),
            Min_Target=("Target", "min")
        )
    )

    gauge_summary["Relative_Error_Percent"] = (
        gauge_summary["Mean_Absolute_Error"]
        / gauge_summary["Mean_Target"].replace(0, np.nan)
        * 100
    )

    gauge_summary = gauge_summary.sort_values(
        "Mean_Absolute_Error",
        ascending=False
    )

    gauge_file = os.path.join(
        OUTPUT_DIR,
        f"{target_key}_gauge_diagnostic.csv"
    )

    gauge_summary.to_csv(gauge_file)

    print("\nTop 15 gauges by mean absolute error:")
    print(gauge_summary.head(15).to_string())

    # ---------------------------------------------------------
    # 3. Extreme-event analysis
    # ---------------------------------------------------------

    target_q90 = df["Target"].quantile(0.90)
    target_q95 = df["Target"].quantile(0.95)

    df["Extreme_Event_90"] = df["Target"] >= target_q90
    df["Extreme_Event_95"] = df["Target"] >= target_q95

    extreme_90 = df[df["Extreme_Event_90"]]
    extreme_95 = df[df["Extreme_Event_95"]]

    print("\nTarget thresholds:")
    print(f"90th percentile: {target_q90:.4f}")
    print(f"95th percentile: {target_q95:.4f}")

    print("\n90th percentile events:")
    print(f"Count: {len(extreme_90)}")
    print(
        f"Mean absolute error: "
        f"{extreme_90['Absolute_Error'].mean():.4f}"
    )

    print("\n95th percentile events:")
    print(f"Count: {len(extreme_95)}")
    print(
        f"Mean absolute error: "
        f"{extreme_95['Absolute_Error'].mean():.4f}"
    )

    # ---------------------------------------------------------
    # 4. Compare normal vs extreme events
    # ---------------------------------------------------------

    regime_summary = pd.DataFrame({
        "Regime": [
            "All Events",
            "Below 90th Percentile",
            "90th Percentile and Above",
            "95th Percentile and Above"
        ],
        "Events": [
            len(df),
            len(df[df["Target"] < target_q90]),
            len(extreme_90),
            len(extreme_95)
        ],
        "Mean_Absolute_Error": [
            df["Absolute_Error"].mean(),
            df[df["Target"] < target_q90]["Absolute_Error"].mean(),
            extreme_90["Absolute_Error"].mean(),
            extreme_95["Absolute_Error"].mean()
        ],
        "Median_Absolute_Error": [
            df["Absolute_Error"].median(),
            df[df["Target"] < target_q90]["Absolute_Error"].median(),
            extreme_90["Absolute_Error"].median(),
            extreme_95["Absolute_Error"].median()
        ]
    })

    regime_file = os.path.join(
        OUTPUT_DIR,
        f"{target_key}_regime_comparison.csv"
    )

    regime_summary.to_csv(regime_file, index=False)

    print("\nRegime comparison:")
    print(regime_summary.to_string(index=False))

    # ---------------------------------------------------------
    # 5. Extreme errors
    # ---------------------------------------------------------

    extreme_errors = df[
        df["Absolute_Error"] >= df["Absolute_Error"].quantile(0.95)
    ].copy()

    extreme_error_file = os.path.join(
        OUTPUT_DIR,
        f"{target_key}_extreme_error_events.csv"
    )

    extreme_errors.to_csv(
        extreme_error_file,
        index=False
    )

    # ---------------------------------------------------------
    # 6. Gauge concentration
    # ---------------------------------------------------------

    top_error_gauges = (
        df.groupby("GaugeID")["Absolute_Error"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    concentration_file = os.path.join(
        OUTPUT_DIR,
        f"{target_key}_error_concentration.csv"
    )

    top_error_gauges.rename(
        "Total_Absolute_Error"
    ).to_csv(concentration_file)

    print("\nTop gauges contributing to total absolute error:")
    print(top_error_gauges.to_string())

    # ---------------------------------------------------------
    # 7. Correlation between target magnitude and error
    # ---------------------------------------------------------

    correlation = df[
        ["Target", "Absolute_Error"]
    ].corr().iloc[0, 1]

    correlation_file = os.path.join(
        OUTPUT_DIR,
        f"{target_key}_target_error_correlation.txt"
    )

    with open(correlation_file, "w") as f:
        f.write(
            f"Target vs Absolute Error correlation: "
            f"{correlation:.6f}\n"
        )

    print(
        f"\nTarget vs Absolute Error correlation: "
        f"{correlation:.6f}"
    )

    # ---------------------------------------------------------
    # 8. Save complete diagnostic dataset
    # ---------------------------------------------------------

    full_file = os.path.join(
        OUTPUT_DIR,
        f"{target_key}_diagnostic_full.csv"
    )

    df.to_csv(full_file, index=False)

    print(f"\nSaved diagnostic files in:")
    print(OUTPUT_DIR)


def main():

    print("=" * 75)
    print("GAUGE AND REGIME DIAGNOSTIC ANALYSIS")
    print("=" * 75)

    for target_key, target_name in TARGETS.items():
        analyze_target(target_key, target_name)

    print("\n" + "=" * 75)
    print("GAUGE / REGIME DIAGNOSTIC ANALYSIS COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()