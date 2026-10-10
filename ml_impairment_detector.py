"""
ml_impairment_detector.py
ML model to detect RF impairments from extracted signal features.

Features used:
  evm_percent, amp_std, phase_std, kurtosis, mean_amp_error,
  snr_db, freq_offset, interference_power

Target:
  label  (1 = PASS, 0 = FAIL)

Model: RandomForestClassifier (scikit-learn)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "evm_percent",
    "amp_std",
    "phase_std",
    "kurtosis",
    "mean_amp_error",
    "snr_db",
    "freq_offset",
    "interference_power",
]
TARGET_COL = "label"
MODEL_PATH = "models/impairment_detector.joblib"
ENCODER_PATH = "models/mod_encoder.joblib"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(data_path: str = "data/signal_dataset.csv") -> RandomForestClassifier:
    df = pd.read_csv(data_path)

    # Encode modulation as a numeric feature
    le = LabelEncoder()
    df["modulation_enc"] = le.fit_transform(df["modulation"])
    feature_cols = FEATURE_COLS + ["modulation_enc"]

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    print("=== Impairment Detector — Test Set ===")
    print(classification_report(y_test, y_pred, target_names=["FAIL", "PASS"]))

    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="f1_weighted")
    print(f"5-Fold CV F1 (weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Confusion matrix plot
    os.makedirs("outputs", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAIL", "PASS"])
    disp.plot(cmap="Blues")
    plt.title("Impairment Detector — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/impairment_cm.png", dpi=150)
    plt.close()

    # Feature importance plot
    importances = clf.feature_importances_
    feat_names = feature_cols
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(feat_names)), importances[indices])
    plt.xticks(range(len(feat_names)), [feat_names[i] for i in indices], rotation=45, ha="right")
    plt.title("Feature Importances — Impairment Detector")
    plt.tight_layout()
    plt.savefig("outputs/impairment_feature_importance.png", dpi=150)
    plt.close()

    # Persist model
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return clf


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model():
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    return clf, le


def predict_impairment(features: dict) -> dict:
    """
    Given a dict of signal features, return predicted label and probability.

    Expected keys: evm_percent, amp_std, phase_std, kurtosis,
                   mean_amp_error, snr_db, freq_offset,
                   interference_power, modulation
    """
    clf, le = load_model()
    mod_enc = le.transform([features["modulation"]])[0]

    row = np.array(
        [
            features["evm_percent"],
            features["amp_std"],
            features["phase_std"],
            features["kurtosis"],
            features["mean_amp_error"],
            features["snr_db"],
            features["freq_offset"],
            features["interference_power"],
            mod_enc,
        ]
    ).reshape(1, -1)

    label = int(clf.predict(row)[0])
    proba = clf.predict_proba(row)[0]

    return {
        "predicted_label": label,
        "label_name": "PASS" if label == 1 else "FAIL",
        "pass_probability": float(proba[1]),
        "fail_probability": float(proba[0]),
    }


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    # Generate dataset if not present
    if not os.path.exists("data/signal_dataset.csv"):
        from signal_generator import generate_dataset
        df = generate_dataset(n_samples=500)
        df.to_csv("data/signal_dataset.csv", index=False)

    train()
