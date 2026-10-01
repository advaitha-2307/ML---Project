import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


SPLIT_DIR = Path("data/splits")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models/regression_controlled")

REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


TARGETS = {
    "peak_flood_level": "Peak Flood Level (m)",
    "peak_discharge": "Peak Discharge Q (cumec)",
    "flood_volume": "Flood Volume (cumec)"
}


MODELS = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=None
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3
    )
}


all_results = []


for target_key, target_name in TARGETS.items():

    print("\n" + "=" * 70)
    print(f"CONTROLLED REGRESSION: {target_name}")
    print("=" * 70)

    X_train = pd.read_csv(
        SPLIT_DIR / f"controlled_{target_key}_X_train.csv"
    )

    X_test = pd.read_csv(
        SPLIT_DIR / f"controlled_{target_key}_X_test.csv"
    )

    y_train = pd.read_csv(
        SPLIT_DIR / f"controlled_{target_key}_y_train.csv"
    ).squeeze()

    y_test = pd.read_csv(
        SPLIT_DIR / f"controlled_{target_key}_y_test.csv"
    ).squeeze()

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    # Identify numerical and categorical features
    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("Numerical features:", len(numerical_features))
    print("Categorical features:", len(categorical_features))

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    for model_name, model in MODELS.items():

        print(f"\nTraining: {model_name}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R²   : {r2:.4f}")

        all_results.append({
            "Target": target_name,
            "Target_Key": target_key,
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        # Save model
        model_filename = (
            f"controlled_{target_key}_"
            f"{model_name.lower().replace(' ', '_')}.joblib"
        )

        joblib.dump(
            pipeline,
            MODEL_DIR / model_filename
        )

        # Save predictions
        prediction_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": predictions
        })

        prediction_filename = (
            f"controlled_{target_key}_"
            f"{model_name.lower().replace(' ', '_')}_predictions.csv"
        )

        prediction_df.to_csv(
            REPORT_DIR / prediction_filename,
            index=False
        )


# Save combined results
results_df = pd.DataFrame(all_results)

results_df.to_csv(
    REPORT_DIR / "controlled_regression_model_results.csv",
    index=False
)

print("\n" + "=" * 70)
print("CONTROLLED REGRESSION COMPLETE")
print("=" * 70)

print(results_df.to_string(index=False))

print(
    "\nSaved:",
    REPORT_DIR / "controlled_regression_model_results.csv"
)