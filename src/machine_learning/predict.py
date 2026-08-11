"""
ScreeningModel — the inference wrapper.
========================================

Loads the trained artifacts once and turns a FeatureVector into a decision.
Used by the CLI (src/main.py) and the API (api/main.py), so the loading and
prediction logic exists in exactly one place.

Two loading strategies:
    ScreeningModel.from_files()    models/*.pkl  — fast, no MLflow needed
    ScreeningModel.from_mlflow()   Model Registry — picks the best AUC run

from_files is the default: the Docker image copies models/ and must not
depend on an MLflow store being present at runtime. from_mlflow exists
because "which model is in production?" should be answerable from the
registry rather than from whichever .pkl someone last copied.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from src.ml.config import (
    ENCODER_FILE,
    METADATA_FILE,
    MODEL_FILE,
    MODELS_DIR,
    SCALER_FILE,
    TRACKING_URI,
)
from src.models.feature_vector import FeatureVector

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A screening decision, with everything needed to judge how much to
    trust it — including which inputs were assumed rather than measured."""

    shortlisted: str                       # "Yes" | "No"
    confidence: float                      # % for the predicted class
    probability_shortlisted: float         # % for the positive class
    probabilities: dict[str, float]
    model_name: str
    model_source: str
    imputed_fields: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "shortlisted": self.shortlisted,
            "confidence": self.confidence,
            "probability_shortlisted": self.probability_shortlisted,
            "probabilities": self.probabilities,
            "model_name": self.model_name,
            "model_source": self.model_source,
            "imputed_fields": self.imputed_fields,
            "features": self.features,
        }


class ScreeningModel:
    """Wraps model + scaler + label encoder + metadata."""

    def __init__(self, model, scaler, encoder, metadata: dict, source: str):
        self.model = model
        self.scaler = scaler
        self.encoder = encoder
        self.metadata = metadata
        self.source = source

        # Fail loudly at load time if the artifacts disagree about columns.
        # A column-order mismatch produces confident wrong answers forever,
        # with no exception anywhere, which is the worst failure available.
        expected = metadata.get("features")
        if expected and list(expected) != FeatureVector.feature_names():
            raise ValueError(
                "Feature contract mismatch between the trained model and the "
                f"current code.\n  model_metadata.json: {expected}\n"
                f"  FeatureVector:       {FeatureVector.feature_names()}\n"
                "Retrain after changing features — do not edit one side only."
            )

    # ── constructors ──────────────────────────────────────────────────

    @classmethod
    def from_files(cls, models_dir: Path | None = None) -> "ScreeningModel":
        models_dir = Path(models_dir or MODELS_DIR)
        missing = [
            f for f in (MODEL_FILE, SCALER_FILE, ENCODER_FILE, METADATA_FILE)
            if not (models_dir / f).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing artifacts in {models_dir}: {missing}. "
                "Run: python -m src.ml.train"
            )
        metadata = json.loads((models_dir / METADATA_FILE).read_text())
        return cls(
            model=joblib.load(models_dir / MODEL_FILE),
            scaler=joblib.load(models_dir / SCALER_FILE),
            encoder=joblib.load(models_dir / ENCODER_FILE),
            metadata=metadata,
            source=f"local files ({metadata.get('model')})",
        )

    @classmethod
    def from_mlflow(cls) -> "ScreeningModel":
        """Load the highest-AUC version from the MLflow Model Registry."""
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(TRACKING_URI)
        client = MlflowClient()

        best = None
        for rm in client.search_registered_models():
            for v in client.search_model_versions(f"name='{rm.name}'"):
                try:
                    auc = client.get_run(v.run_id).data.metrics.get("auc", -1)
                except Exception:  # a deleted or corrupt run must not abort the scan
                    continue
                if best is None or auc > best[0]:
                    best = (auc, rm.name, v.version, v.run_id)

        if best is None:
            raise RuntimeError(
                "No registered models found in the MLflow registry. "
                "Run: python -m src.ml.train"
            )

        auc, name, version, run_id = best
        model = mlflow.sklearn.load_model(f"models:/{name}/{version}")

        # Preprocessors travel with the run as artifacts — loading them from
        # models/ instead would risk pairing a registry model with a scaler
        # fitted during a different run.
        artifacts = Path(
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessors")
        )
        # Merge, don't replace: the local metadata carries the full metric
        # set, feature list and imputation defaults. Overwriting `metrics`
        # with just the registry AUC would silently hollow out /model/info.
        metadata = json.loads((MODELS_DIR / METADATA_FILE).read_text())
        model_type = name.split("_", 1)[-1]
        metadata = {
            **metadata,
            "model": model_type,
            "metrics": {**metadata.get("all_results", {}).get(model_type, {}), "auc": auc},
        }

        logger.info("Loaded %s v%s from MLflow registry (AUC %.4f)", name, version, auc)
        return cls(
            model=model,
            scaler=joblib.load(artifacts / SCALER_FILE),
            encoder=joblib.load(artifacts / ENCODER_FILE),
            metadata=metadata,
            source=f"MLflow Registry: {name} v{version}",
        )

    # ── properties ────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self.metadata.get("model", type(self.model).__name__)

    @property
    def imputation_defaults(self) -> dict[str, float]:
        """Training-set medians — handed to FeatureEngineer so an imputed
        field is filled with the training distribution's centre rather than
        an arbitrary constant."""
        return self.metadata.get("imputation_defaults", {})

    # ── inference ─────────────────────────────────────────────────────

    def predict(self, features: FeatureVector) -> Prediction:
        row = np.array([features.to_list()], dtype=float)
        scaled = self.scaler.transform(row)

        index = int(self.model.predict(scaled)[0])
        proba = self.model.predict_proba(scaled)[0]
        label = str(self.encoder.inverse_transform([index])[0])

        probabilities = {
            str(cls): round(float(p) * 100, 2)
            for cls, p in zip(self.encoder.classes_, proba)
        }
        return Prediction(
            shortlisted=label,
            confidence=round(float(np.max(proba)) * 100, 2),
            probability_shortlisted=probabilities.get("Yes", 0.0),
            probabilities=probabilities,
            model_name=self.model_name,
            model_source=self.source,
            imputed_fields=list(features.imputed),
            features=features.to_dict(),
        )
