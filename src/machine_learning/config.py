"""Single source of truth for ML paths and constants.

Everything downstream (training, evaluation, inference, the API, the
notebook) imports from here, so changing a path is a one-line edit rather
than a search-and-replace across seven files.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_FILE = Path(os.getenv("DATA_FILE", PROJECT_ROOT / "data/raw/ai_resume_screening.csv"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
MLRUNS_DIR = Path(os.getenv("MLRUNS_DIR", PROJECT_ROOT / "mlruns"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))

# ── MLflow backend ────────────────────────────────────────────────────
# MLflow 3.x put the filesystem tracking store ("./mlruns") into
# maintenance mode: set_tracking_uri("file://.../mlruns") now RAISES on a
# fresh install. So the backend is SQLite (what MLflow recommends) while
# artifacts still land in mlruns/.
#
# Browse runs with:
#     mlflow ui --backend-store-uri sqlite:///mlflow.db
#
# To force the legacy file store instead, export both:
#     MLFLOW_ALLOW_FILE_STORE=true
#     MLFLOW_TRACKING_URI=file:///abs/path/to/mlruns
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB}")

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "Resume_Screening_Experiment")

# Artifact filenames — the API's load contract.
MODEL_FILE = "best_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "label_encoder.pkl"
METADATA_FILE = "model_metadata.json"

TARGET_COLUMN = "shortlisted"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
