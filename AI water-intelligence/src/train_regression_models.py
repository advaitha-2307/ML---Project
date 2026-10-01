import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

SPLIT_DIR = "data/splits"
REPORT_DIR = "reports"
MODEL_DIR = "models/regression"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# TARGETS
# ============================================================

TARGETS = {
    "peak_flood_level": "Peak Flood Level (m)",
    "peak_discharge": "Peak Discharge Q (cumec)",
    "flood_volume": "Flood Volume (cumec)"
}


# ============================================================
# REGRESSION MODELS
# ============================================================

def create_models(numeric_features, categorical_features):

    numeric_scaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    numeric_tree = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ("num", numeric_scaled, numeric_features),
            ("cat", categorical, categorical_features)
        ],
        sparse_threshold=0
    )

    preprocessor_tree = ColumnTransformer(
        transformers=[
            ("num", numeric_tree, numeric_features),
            ("cat", categorical, categorical_features)
        ],
        sparse_threshold=0
    )

    models = {

        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor_scaled),
                ("model", LinearRegression())
            ]
        ),

        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", preprocessor_tree),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=10,
                        random_state=42
                    )
                )
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor_tree),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=15,
                        random_state=42,
                        n_jobs=-1
                    )
                )
            ]
        ),

        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor_tree),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42
                    )
                )
            ]
        )
    }

    return models


# ============================================================
# MAIN
# ============================================================

print("=" * 75)
print("AI WATER INTELLIGENCE - REGRESSION MODEL TRAINING")
print("=" * 75)


all_results = []


for target_key, target_column in TARGETS.items():

    print("\n")
    print("=" * 75)
    print(f"TARGET: {target_column}")
    print("=" * 75)

    # --------------------------------------------------------
    # Load split files
    # --------------------------------------------------------

    X_train_path = os.path.join(
        SPLIT_DIR,
        f"{target_key}_X_train.csv"
    )

    X_test_path = os.path.join(
        SPLIT_DIR,
        f"{target_key}_X_test.csv"
    )

    y_train_path = os.path.join(
        SPLIT_DIR,
        f"{target_key}_y_train.csv"
    )

    y_test_path = os.path.join(
        SPLIT_DIR,
        f"{target_key}_y_test.csv"
    )

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    y_train = pd.read_csv(y_train_path).squeeze("columns")
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")

    # --------------------------------------------------------
    # Feature types
    # --------------------------------------------------------

    numeric_features = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print(f"Numerical features   : {len(numeric_features)}")
    print(f"Categorical features : {len(categorical_features)}")

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    models = create_models(
        numeric_features,
        categorical_features
    )

    # --------------------------------------------------------
    # Train each model
    # --------------------------------------------------------

    for model_name, pipeline in models.items():

        print("\n" + "-" * 75)
        print(f"Training: {model_name}")

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        print(f"MAE  : {mae:.6f}")
        print(f"RMSE : {rmse:.6f}")
        print(f"R²   : {r2:.6f}")

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        predictions = pd.DataFrame(
            {
                "Actual": y_test.values,
                "Predicted": y_pred
            }
        )

        prediction_path = os.path.join(
            REPORT_DIR,
            f"{target_key}_{model_name.lower().replace(' ', '_')}_predictions.csv"
        )

        predictions.to_csv(
            prediction_path,
            index=False
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        target_model_dir = os.path.join(
            MODEL_DIR,
            target_key
        )

        os.makedirs(
            target_model_dir,
            exist_ok=True
        )

        model_path = os.path.join(
            target_model_dir,
            f"{model_name.lower().replace(' ', '_')}.joblib"
        )

        joblib.dump(
            pipeline,
            model_path
        )

        print(f"Model saved: {model_path}")

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        all_results.append(
            {
                "Target": target_column,
                "Target_Key": target_key,
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            }
        )


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_path = os.path.join(
    REPORT_DIR,
    "regression_model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("FINAL REGRESSION MODEL RESULTS")
print("=" * 75)

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved to:")
print(results_path)

print("\nRegression training completed successfully.")