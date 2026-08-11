from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    """Our application's representation of a candidate.

    The LLM merely fills this object; swapping Gemini for another model must
    not change this class. Deliberately flat for V1 (lists of plain strings).
    Nested models (Skill, Education, ...) are a V2 concern — see README.

    Note: Pydantic v2 ignores unknown keys by default, so extra fields the
    LLM invents (e.g. "certifications") are dropped silently rather than
    crashing the pipeline. That is the behavior we want in V1.
    """

    name: str | None = None
    education: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
