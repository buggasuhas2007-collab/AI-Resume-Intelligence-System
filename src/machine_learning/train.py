"""
Train, compare, and register the shortlisting models.
======================================================

    python -m src.ml.train

Trains Random Forest, Logistic Regression and Gradient Boosting on the
same split, logs every run to MLflow (params, metrics, model, and the
preprocessors as artifacts), registers each in the MLflow Model Registry,
then saves the winner plus its metadata to models/.

Selection metric: ROC-AUC, not accuracy. The target is ~70% positive, so a
model that predicts "Yes" for everyone scores 0.70 accuracy while being
useless. AUC measures ranking quality across every threshold, which is what
a screening tool actually needs — recruiters work down a ranked list, they
do not consume a hard yes/no.
"""

import json
import logging
import sys
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.ml.config import (
    CV_FOLDS,
    ENCODER_FILE,
    EXPERIMENT_NAME,
    METADATA_FILE,
    MLRUNS_DIR,
    MODEL_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    SCALER_FILE,
    TARGET_COLUMN,
    TEST_SIZE,
    TRACKING_URI,
)
from src.ml.dataset import describe, encode_features, load_dataset
from src.models.feature_vector import EDUCATION_MAP, FEATURE_NAMES

logger = logging.getLogger(__name__)

# Hyperparameters kept close to the reference project so results are
# comparable; tuning is a documented V3 step (RandomizedSearchCV).
MODEL_SPECS = {
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=30, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "LogisticRegression": lambda: LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    "GradientBoosting": lambda: GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
    ),
}

REGISTRY_PREFIX = "ResumeScreening"


def _setup_mlflow() -> None:
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=MLRUNS_DIR.resolve().as_uri()
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def _evaluate(model, X_test, y_test) -> dict:
    """Metrics on the held-out test set.

    Both precision and recall are reported because they trade off in
    opposite directions for a screening tool: low precision wastes
    interviewer time, low recall silently discards good candidates. The
    second failure is invisible to the business and matters more.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_proba)),
    }


def train() -> dict:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _setup_mlflow()

    # ── Data ──────────────────────────────────────────────────────────
    df = load_dataset()
    stats = describe(df)
    logger.info(
        "Dataset: %s rows, %.1f%% positive (%s)",
        f"{stats['rows']:,}", stats["positive_rate"] * 100, stats["class_counts"],
    )

    X = encode_features(df).values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COLUMN])
    # Guard the class order: index 1 must be "Yes", since predict_proba[:, 1]
    # is treated as P(shortlisted) everywhere downstream.
    assert list(label_encoder.classes_) == ["No", "Yes"], label_encoder.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scaler is fitted on TRAIN ONLY. Fitting on all of X first — as many
    # tutorials do — leaks test-set statistics into training and inflates
    # every metric below.
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    joblib.dump(scaler, MODELS_DIR / SCALER_FILE)
    joblib.dump(label_encoder, MODELS_DIR / ENCODER_FILE)
    logger.info("Train: %d   Test: %d", len(X_train), len(X_test))

    # Training-set medians, used to impute fields a resume cannot supply.
    imputation_defaults = {
        name: float(np.median(X_train[:, i]))
        for i, name in enumerate(FEATURE_NAMES)
    }

    # ── Train every candidate model ───────────────────────────────────
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, build in MODEL_SPECS.items():
        with mlflow.start_run(run_name=name):
            model = build()
            model.fit(X_train_s, y_train)

            metrics = _evaluate(model, X_test_s, y_test)

            # Cross-validated AUC on the training folds: a single test split
            # can flatter one model by luck, and the three models here score
            # within ~0.01 of each other. CV makes the comparison honest.
            cv_scores = cross_val_score(
                build(), X_train_s, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
            )
            metrics["cv_auc_mean"] = float(cv_scores.mean())
            metrics["cv_auc_std"] = float(cv_scores.std())

            params = {**model.get_params(), "model_type": name}
            mlflow.log_params({k: v for k, v in params.items() if v is not None})
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"features": ",".join(FEATURE_NAMES), "stage": "training"})

            mlflow.sklearn.log_model(
                model, name="model", registered_model_name=f"{REGISTRY_PREFIX}_{name}"
            )
            mlflow.log_artifact(str(MODELS_DIR / SCALER_FILE), "preprocessors")
            mlflow.log_artifact(str(MODELS_DIR / ENCODER_FILE), "preprocessors")

            results[name] = metrics
            fitted[name] = model
            logger.info(
                "%-20s acc=%.4f  f1=%.4f  auc=%.4f  cv_auc=%.4f±%.4f",
                name, metrics["accuracy"], metrics["f1"], metrics["auc"],
                metrics["cv_auc_mean"], metrics["cv_auc_std"],
            )

    # ── Select and persist the winner ─────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["auc"])
    best_model = fitted[best_name]
    joblib.dump(best_model, MODELS_DIR / MODEL_FILE)

    importances = _feature_importances(best_model)

    metadata = {
        "model": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_NAMES,
        "education_map": EDUCATION_MAP,
        "classes": list(label_encoder.classes_),
        "selection_metric": "auc",
        "metrics": results[best_name],
        "all_results": results,
        "dataset": stats,
        "imputation_defaults": imputation_defaults,
        "feature_importances": importances,
    }
    (MODELS_DIR / METADATA_FILE).write_text(json.dumps(metadata, indent=2))

    logger.info("Best model: %s (AUC %.4f) → %s",
                best_name, results[best_name]["auc"], MODELS_DIR / MODEL_FILE)
    logger.info("Browse runs: mlflow ui --backend-store-uri %s", TRACKING_URI)
    return metadata


def _feature_importances(model) -> dict[str, float]:
    """Tree importances, or |coefficients| for the linear model.

    These are not the same quantity and must not be compared across model
    types — coefficients are per-standard-deviation effects on the log-odds,
    tree importances are impurity reductions. Reported only for the winner.
    """
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return {}
    return {name: float(v) for name, v in zip(FEATURE_NAMES, values)}


if __name__ == "__main__":
    try:
        train()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)
