from pydantic import BaseModel, Field


class JobRequirement(BaseModel):
    """What a job asks for.

    `required_skills` drives `skills_match_score`, which the trained model
    weights most heavily. A prediction without a job is meaningless —
    "will this candidate be shortlisted?" is only answerable relative to
    something they are being shortlisted FOR. FeatureEngineer therefore
    treats a missing/empty job as an error, not as a zero.

    Weighted skills (must-have vs nice-to-have) are a V3 upgrade; it would
    change how skills_match_score is computed, so it requires regenerating
    the training column and retraining.
    """

    title: str | None = None
    required_skills: list[str] = Field(default_factory=list)
