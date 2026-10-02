"""
Train and evaluate the Random Forest disease prediction model.

Dataset:
    dataset/Training.csv
    dataset/Testing.csv

The model artifact contains:
    - trained RandomForestClassifier
    - exact ordered feature_names (132 symptom columns)
    - target_name
    - class_names
"""
from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "dataset" / "Training.csv"
TEST_PATH = BASE_DIR / "dataset" / "Testing.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "disease_model.pkl"

TARGET = "prognosis"
EXCLUDED_TRAINING_COLUMNS = {"Unnamed: 133"}


def load_dataset(path: Path):
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' was not found in {path}.")

    # The Kaggle Training.csv contains an extra index-like column.
    feature_names = [
        col for col in df.columns
        if col != TARGET and col not in EXCLUDED_TRAINING_COLUMNS
    ]
    if len(feature_names) != 132:
        raise ValueError(
            f"Expected exactly 132 symptom features, found {len(feature_names)} "
            f"in {path.name}."
        )

    return df, feature_names


def main():
    train_df, feature_names = load_dataset(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if TARGET not in test_df.columns:
        raise ValueError(f"Required target column '{TARGET}' was not found in Testing.csv.")

    test_features = [col for col in test_df.columns if col != TARGET]
    if test_features != feature_names:
        raise ValueError(
            "Training and Testing feature columns/order do not match. "
            "The model must use exactly the same feature order."
        )

    X_train = train_df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    y_train = train_df[TARGET].astype(str).str.strip()

    X_test = test_df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    y_test = test_df[TARGET].astype(str).str.strip()

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("=" * 72)
    print("Random Forest Disease Prediction Model")
    print("=" * 72)
    print(f"Training rows : {len(train_df)}")
    print(f"Testing rows  : {len(test_df)}")
    print(f"Features      : {len(feature_names)}")
    print(f"Disease classes: {len(model.classes_)}")
    print(f"Accuracy      : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": feature_names,
        "target_name": TARGET,
        "class_names": list(model.classes_),
        "accuracy": float(accuracy),
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
    }

    with MODEL_PATH.open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
