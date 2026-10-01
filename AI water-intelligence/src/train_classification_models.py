import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

TRAIN_X = "data/splits/classification_X_train.csv"
TEST_X = "data/splits/classification_X_test.csv"
TRAIN_Y = "data/splits/classification_y_train.csv"
TEST_Y = "data/splits/classification_y_test.csv"

REPORT_DIR = "reports"
MODEL_DIR = "models/classification"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AI WATER INTELLIGENCE - FLOOD TYPE CLASSIFICATION")
print("=" * 70)

print("\nLoading training and testing data...")

X_train = pd.read_csv(TRAIN_X)
X_test = pd.read_csv(TEST_X)

y_train = pd.read_csv(TRAIN_Y).squeeze("columns")
y_test = pd.read_csv(TEST_Y).squeeze("columns")

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape : {y_test.shape}")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ============================================================
# CHECK TARGET
# ============================================================

if y_train.nunique() < 2:
    raise ValueError("Training data contains only one class.")

if y_test.nunique() < 2:
    print("\nWARNING: Test data contains only one class.")
    print("ROC-AUC will not be calculated.")


# ============================================================
# DETECT FEATURE TYPES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nFeature information:")
print(f"Numerical features   : {len(numeric_features)}")
print(f"Categorical features : {len(categorical_features)}")

print("\nCategorical columns:")
print(categorical_features)


# ============================================================
# PREPROCESSOR FOR LOGISTIC REGRESSION
# ============================================================

numeric_pipeline_scaled = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
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
        ("num", numeric_pipeline_scaled, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ],
    sparse_threshold=0
)


# ============================================================
# PREPROCESSOR FOR TREE MODELS
# ============================================================

numeric_pipeline_tree = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline_tree = Pipeline(
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

preprocessor_tree = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline_tree, numeric_features),
        ("cat", categorical_pipeline_tree, categorical_features)
    ],
    sparse_threshold=0
)


# ============================================================
# DEFINE MODELS
# ============================================================

models = {
    "Logistic Regression": Pipeline(
        steps=[
            ("preprocessor", preprocessor_scaled),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]
    ),

    "Decision Tree": Pipeline(
        steps=[
            ("preprocessor", preprocessor_tree),
            (
                "model",
                DecisionTreeClassifier(
                    random_state=42,
                    max_depth=10
                )
            )
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            ("preprocessor", preprocessor_tree),
            (
                "model",
                RandomForestClassifier(
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
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            )
        ]
    )
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = []

for model_name, pipeline in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    print("Fitting model...")

    pipeline.fit(X_train, y_train)

    print("Prediction...")

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # --------------------------------------------------------
    # Find probability corresponding to Severe Flood
    # --------------------------------------------------------

    model_classes = pipeline.named_steps["model"].classes_

    if "Severe Flood" in model_classes:
        severe_index = list(model_classes).index("Severe Flood")
        severe_probability = y_prob[:, severe_index]
    else:
        severe_probability = None

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="Severe Flood",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="Severe Flood",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="Severe Flood",
        zero_division=0
    )

    if y_test.nunique() == 2 and severe_probability is not None:
        roc_auc = roc_auc_score(
            (y_test == "Severe Flood").astype(int),
            severe_probability
        )
    else:
        roc_auc = np.nan

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print("\nResults:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}" if not np.isnan(roc_auc)
          else "ROC-AUC  : Not available")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Flood", "Severe Flood"]
    )

    print("Confusion Matrix:")
    print(cm)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Flood", "Actual Severe Flood"],
        columns=["Predicted Flood", "Predicted Severe Flood"]
    )

    safe_name = model_name.lower().replace(" ", "_")

    cm_path = os.path.join(
        REPORT_DIR,
        f"{safe_name}_confusion_matrix.csv"
    )

    cm_df.to_csv(cm_path)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"{safe_name}.joblib"
    )

    joblib.dump(pipeline, model_path)

    print(f"\nModel saved: {model_path}")
    print(f"Confusion matrix saved: {cm_path}")

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision_Severe_Flood": precision,
            "Recall_Severe_Flood": recall,
            "F1_Severe_Flood": f1,
            "ROC_AUC": roc_auc
        }
    )


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_path = os.path.join(
    REPORT_DIR,
    "classification_model_results.csv"
)

results_df.to_csv(results_path, index=False)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))

print("\nResults saved to:")
print(results_path)

print("\nAll classification models trained successfully.")