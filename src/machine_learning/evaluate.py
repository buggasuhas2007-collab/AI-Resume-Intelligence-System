"""
Evaluate the saved model and write a report.
=============================================

    python -m src.ml.evaluate

Regenerates the held-out split with the same seed as training, then writes
to artifacts/:
    confusion_matrix.png
    roc_curve.png
    feature_importance.png
    classification_report.txt
    threshold_analysis.txt

The threshold analysis is the part worth reading. Training reports metrics
at the default 0.5 cutoff, but a screening tool is a ranking device — the
operating point is a business decision (how many interview slots exist),
not a modelling one.
"""

import json
import logging
import sys

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: no display in a container or CI
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from src.ml.config import (  # noqa: E402
    ARTIFACTS_DIR,
    ENCODER_FILE,
    METADATA_FILE,
    MODEL_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    SCALER_FILE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.ml.dataset import encode_features, load_dataset  # noqa: E402
from src.models.feature_vector import FEATURE_NAMES  # noqa: E402

logger = logging.getLogger(__name__)


def evaluate() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not (MODELS_DIR / MODEL_FILE).exists():
        raise FileNotFoundError(
            f"No trained model at {MODELS_DIR / MODEL_FILE}. Run: python -m src.ml.train"
        )

    model = joblib.load(MODELS_DIR / MODEL_FILE)
    scaler = joblib.load(MODELS_DIR / SCALER_FILE)
    encoder = joblib.load(MODELS_DIR / ENCODER_FILE)
    metadata = json.loads((MODELS_DIR / METADATA_FILE).read_text())

    df = load_dataset()
    X = encode_features(df).values
    y = encoder.transform(df[TARGET_COLUMN])

    # Same seed and stratification as training → the same held-out rows.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_test_s = scaler.transform(X_test)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    # ── Classification report ─────────────────────────────────────────
    report = classification_report(y_test, y_pred, target_names=list(encoder.classes_))
    (ARTIFACTS_DIR / "classification_report.txt").write_text(
        f"Model: {metadata['model']}\n\n{report}\n"
    )
    print(f"\nModel: {metadata['model']}\n{report}")

    # ── Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=list(encoder.classes_)).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion Matrix — {metadata['model']}")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ── ROC curve ─────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {metadata['metrics']['auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — {metadata['model']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    # ── Feature importance ────────────────────────────────────────────
    importances = metadata.get("feature_importances", {})
    if importances:
        names = list(importances)
        values = [importances[n] for n in names]
        order = np.argsort(values)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.barh([names[i] for i in order], [values[i] for i in order], color="#4C72B0")
        ax.set_title(f"Feature importance — {metadata['model']}")
        fig.tight_layout()
        fig.savefig(ARTIFACTS_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)

    # ── Threshold analysis ────────────────────────────────────────────
    lines = [
        "Operating-point analysis (positive class = shortlisted)",
        "",
        "The default 0.5 cutoff is a modelling convention, not a business",
        "decision. Pick the row matching your interview capacity.",
        "",
        f"{'thresh':>7} {'precision':>10} {'recall':>8} {'f1':>7} {'flagged%':>9}",
    ]
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred_t = (y_proba >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_test, pred_t, average="binary", zero_division=0
        )
        lines.append(f"{t:>7.2f} {p:>10.4f} {r:>8.4f} {f:>7.4f} {pred_t.mean():>9.1%}")
    text = "\n".join(lines)
    (ARTIFACTS_DIR / "threshold_analysis.txt").write_text(text + "\n")
    print("\n" + text)

    logger.info("Artifacts written to %s", ARTIFACTS_DIR)
    return {"confusion_matrix": cm.tolist(), "report": report}


if __name__ == "__main__":
    try:
        evaluate()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)
