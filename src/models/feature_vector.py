from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────
# THE FEATURE CONTRACT
# ─────────────────────────────────────────────────────────────────────
# These six names, in this order, ARE the training data's columns
# (ai_resume_screening.csv). The dataset defines the contract; the
# extraction pipeline's job is to satisfy it.
#
# V1 of this project had a different, self-invented feature set
# (skill_count, education_count, experience_count, ...). Those features
# were dropped in V2: a model cannot consume a feature it was never
# trained on, so inventing features upstream of a fixed dataset is wasted
# work. Adding one back means adding a column to the dataset and
# retraining — not editing this file alone.
#
# NEVER reorder these. A trained model's columns are positional; a silent
# reorder produces confident, wrong predictions with no error anywhere.
# ─────────────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    "years_experience",
    "skills_match_score",
    "education_level",
    "project_count",
    "resume_length",
    "github_activity",
]

# Training-data encoding of education_level (from the reference notebook).
EDUCATION_MAP: dict[str, int] = {
    "High School": 1,
    "Bachelors": 2,
    "Masters": 3,
    "PhD": 4,
}


class FeatureVector(BaseModel):
    """Numeric representation of a candidate, matching the training schema.

    `imputed` is provenance, NOT a feature: it lists which fields could not
    be measured from the resume and were filled with a training-set default.
    It never reaches the model — it reaches the human reading the prediction,
    so a decision that leaned on an assumption is visibly distinguishable
    from one grounded entirely in the document.
    """

    years_experience: float = Field(ge=0.0, le=60.0)
    skills_match_score: float = Field(ge=0.0, le=100.0)
    education_level: int = Field(ge=1, le=4)
    project_count: int = Field(ge=0)
    resume_length: int = Field(ge=0)
    github_activity: int = Field(ge=0)

    imputed: list[str] = Field(default_factory=list)

    @classmethod
    def feature_names(cls) -> list[str]:
        """Ordered model input columns (excludes `imputed`)."""
        return list(FEATURE_NAMES)

    def to_list(self) -> list[float]:
        """Ordered numeric row, ready for scaler.transform([row])."""
        return [float(getattr(self, name)) for name in FEATURE_NAMES]

    def to_dict(self) -> dict:
        """Model inputs only — `imputed` is deliberately excluded."""
        return {name: getattr(self, name) for name in FEATURE_NAMES}

    @property
    def is_fully_measured(self) -> bool:
        """True when every feature came from the resume, none from a default."""
        return not self.imputed
