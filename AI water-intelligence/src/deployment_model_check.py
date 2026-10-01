import joblib
from pathlib import Path

MODEL_DIR = Path("models/regression_controlled")

MODELS = {
    "Peak Flood Level": MODEL_DIR / "controlled_peak_flood_level_gradient_boosting.joblib",
    "Peak Discharge": MODEL_DIR / "controlled_peak_discharge_gradient_boosting.joblib",
    "Flood Volume": MODEL_DIR / "controlled_flood_volume_gradient_boosting.joblib",
}


def inspect_pipeline(name, path):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    if not path.exists():
        print(f"[ERROR] Model not found: {path}")
        return

    pipeline = joblib.load(path)

    print(f"Model file: {path}")
    print(f"Pipeline type: {type(pipeline).__name__}")

    print("\nPipeline steps:")

    if hasattr(pipeline, "named_steps"):
        for step_name, step in pipeline.named_steps.items():
            print(f"  - {step_name}: {type(step).__name__}")

    # Look for preprocessing transformer
    preprocessor = None

    if hasattr(pipeline, "named_steps"):
        for name, step in pipeline.named_steps.items():
            if hasattr(step, "transformers_"):
                preprocessor = step
                break

    if preprocessor is None:
        print("\n[WARNING] Could not automatically locate ColumnTransformer.")
        return

    print("\nInput feature groups:")

    for transformer_name, transformer, columns in preprocessor.transformers_:

        if transformer_name == "remainder":
            continue

        print(f"\n[{transformer_name}]")

        print(f"Transformer: {type(transformer).__name__}")

        try:
            print(f"Number of input columns: {len(columns)}")

            if len(columns) <= 150:
                for column in columns:
                    print(f"  - {column}")
            else:
                print("Too many columns to display individually.")

        except TypeError:
            print("Columns could not be enumerated.")


def main():

    print("=" * 70)
    print("DEPLOYMENT MODEL CHECK")
    print("=" * 70)

    for name, path in MODELS.items():
        inspect_pipeline(name, path)

    print("\n" + "=" * 70)
    print("MODEL CHECK COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()