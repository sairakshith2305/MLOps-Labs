import os
import json
import joblib
from typing import Dict, Any

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Loading the  Dataset
def load_data():
    data = load_iris()
    X, y = data.data, data.target
    print("[INFO] Dataset loaded successfully.")
    return X, y, data.feature_names, data.target_names


# Preprocessing the dataset
def preprocess_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[INFO] Data preprocessed")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# Training Multiple Models
def train_models(X_train, y_train):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": SVC(kernel="rbf", probability=True, random_state=42),
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"[INFO] Trained model: {name}")
    return trained_models


# Evaluate the trained Models
def evaluate_models(models, X_test, y_test) -> Dict[str, Any]:
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="macro")),
            "recall": float(recall_score(y_test, y_pred, average="macro")),
            "f1_score": float(f1_score(y_test, y_pred, average="macro")),
        }
        print(f"[INFO] Evaluation for {name}: {results[name]}")
    return results


# Save Models with their respective metrics
def save_artifacts(models, metrics, scaler, output_dir="model"):
    os.makedirs(output_dir, exist_ok=True)

    for name, model in models.items():
        path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, path)

    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Saved models, scaler, and metrics to {output_dir}")
    return metrics_path


if __name__ == "__main__":
    run_pipeline()