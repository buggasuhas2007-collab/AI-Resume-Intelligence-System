"""Dataset loading with an explicit schema contract.

The most expensive ML bug is a silent one: a renamed column, a unit change,
or an unexpected category that trains fine and predicts nonsense. So the
dataset is validated on load and the process stops on any mismatch.
"""

import logging
from pathlib import Path

import pandas as pd

from src.models.feature_vector import EDUCATION_MAP, FEATURE_NAMES
from src.ml.config import DATA_FILE, TARGET_COLUMN

logger = logging.getLogger(__name__)

# Raw CSV columns: the model features, but with education still categorical.
RAW_COLUMNS = [
    "years_experience",
    "skills_match_score",
    "education_level",
    "project_count",
    "resume_length",
    "github_activity",
    TARGET_COLUMN,
]

VALID_TARGETS = {"Yes", "No"}


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load and validate the screening dataset.

    Raises:
        FileNotFoundError: with DVC-aware guidance.
        ValueError: on any schema violation — missing columns, unknown
            education categories, unexpected target labels, or nulls.
    """
    path = Path(path or DATA_FILE)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "  • With DVC remote access:  dvc pull\n"
            "  • Without it (synthetic):  python scripts/generate_dataset.py"
        )

    df = pd.read_csv(path)

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset schema mismatch. Missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    nulls = int(df[RAW_COLUMNS].isnull().sum().sum())
    if nulls:
        raise ValueError(
            f"Dataset contains {nulls} null values in required columns. "
            "V1 has no imputation strategy for training data — clean the "
            "source rather than letting nulls become silent zeros."
        )

    unknown_edu = set(df["education_level"].unique()) - set(EDUCATION_MAP)
    if unknown_edu:
        raise ValueError(
            f"Unknown education categories: {sorted(unknown_edu)}. "
            f"Known: {sorted(EDUCATION_MAP)}. Add a mapping in "
            "src/models/feature_vector.py::EDUCATION_MAP and retrain."
        )

    unknown_target = set(df[TARGET_COLUMN].unique()) - VALID_TARGETS
    if unknown_target:
        raise ValueError(f"Unexpected target labels: {sorted(unknown_target)}")

    logger.info("Loaded %s rows from %s", f"{len(df):,}", path)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map education to its ordinal and return columns in FEATURE_NAMES order.

    Ordinal, not one-hot: education has a natural order, and one integer
    column lets a tree split on 'at least a Masters' in a single node
    instead of needing several. It also keeps the feature count at 6 rather
    than 9, which matters more the smaller the dataset gets.
    """
    out = df.copy()
    out["education_level"] = out["education_level"].map(EDUCATION_MAP)
    return out[FEATURE_NAMES]


def describe(df: pd.DataFrame) -> dict:
    """Summary used for logging and the model card."""
    counts = df[TARGET_COLUMN].value_counts()
    return {
        "rows": int(len(df)),
        "positive_rate": float(counts.get("Yes", 0) / len(df)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
    }
