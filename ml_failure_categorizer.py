"""
ml_failure_categorizer.py
Categorizes failures from test log messages using TF-IDF + Logistic Regression.

Input:  test log CSV (log_message column)
Target: failure_reason column (multi-label encoded as primary category)

Categories (primary):
  HIGH_EVM, HIGH_BER, PHASE_NOISE, MULTIPLE, NONE (pass)
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = "models/failure_categorizer.joblib"


# ---------------------------------------------------------------------------
# Label engineering
# ---------------------------------------------------------------------------

def encode_failure_category(reason: str) -> str:
    """Map raw failure_reason string to a single primary category."""
    if reason == "NONE":
        return "PASS"
    parts = [r.strip() for r in reason.split(";")]
    if len(parts) > 1:
        return "MULTIPLE"
    return parts[0]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(log_path: str = "data/test_log.csv") -> Pipeline:
    df = pd.read_csv(log_path)
    df["category"] = df["failure_reason"].apply(encode_failure_category)

    print("Category distribution:")
    print(df["category"].value_counts(), "\n")

    X = df["log_message"].values
    y = df["category"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("=== Failure Categorizer — Test Set ===")
    print(classification_report(y_test, y_pred))

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1_weighted")
    print(f"5-Fold CV F1 (weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Confusion matrix
    os.makedirs("outputs", exist_ok=True)
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Oranges", colorbar=False)
    plt.title("Failure Categorizer — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/failure_cm.png", dpi=150)
    plt.close()

    # Top TF-IDF tokens per class
    tfidf: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf: LogisticRegression = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()

    fig, axes = plt.subplots(1, len(clf.classes_), figsize=(4 * len(clf.classes_), 4))
    for ax, cls, coef in zip(axes, clf.classes_, clf.coef_):
        top_idx = coef.argsort()[-10:][::-1]
        ax.barh(feature_names[top_idx], coef[top_idx])
        ax.set_title(cls)
        ax.invert_yaxis()
    plt.suptitle("Top TF-IDF Features per Class")
    plt.tight_layout()
    plt.savefig("outputs/failure_tfidf_features.png", dpi=150)
    plt.close()

    # Persist
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return pipeline


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model() -> Pipeline:
    return joblib.load(MODEL_PATH)


def categorize_failure(log_message: str) -> dict:
    """Classify a single log message string into a failure category."""
    pipeline = load_model()
    category = pipeline.predict([log_message])[0]
    proba = pipeline.predict_proba([log_message])[0]
    classes = pipeline.classes_

    return {
        "category": category,
        "probabilities": {cls: float(p) for cls, p in zip(classes, proba)},
    }


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    if not os.path.exists("data/test_log.csv"):
        from test_cases import generate_test_log
        df = generate_test_log(n_cases=300)
        df.to_csv("data/test_log.csv", index=False)

    train()
