import re
from datetime import datetime

from src.models.candidate_profile import CandidateProfile
from src.models.document import Document
from src.models.feature_vector import FeatureVector
from src.models.job_requirement import JobRequirement

# Fallbacks used only when the caller supplies no training-set defaults.
# Production callers pass `imputation_defaults` from model_metadata.json so
# the fill value is the actual training median — see ScreeningModel.
_FALLBACK_DEFAULTS: dict[str, float] = {
    "education_level": 2.0,   # Bachelors — the modal class
    "github_activity": 225.0,
}


class FeatureEngineer:
    """Translates a CandidateProfile into the model's FeatureVector.

    Knows nothing about PDFs, Gemini, or sklearn — it maps structured
    candidate data onto the six columns the training data defines
    (Single Responsibility).

    All derived features are heuristics, and they are deliberately
    conservative: when a pattern is ambiguous, prefer missing a signal
    (false negative) over fabricating one (false positive) — the same
    "never guess" rule the LLM prompt enforces.

    Two things this class will NOT do:
      1. Invent a skills_match_score when there is no job (raises instead).
      2. Silently fill a missing field with a plausible number — every
         imputed field is named in FeatureVector.imputed.
    """

    # ── education ────────────────────────────────────────────────────
    # Ordered highest-first; first match wins per entry.
    # Bare "ME"/"MS"/"BE" (no dots) are NOT matched: in Indian resumes
    # "ME" usually means Mechanical Engineering, not Master of Engineering.
    _EDUCATION_LEVELS: list[tuple[re.Pattern, int]] = [
        (re.compile(r"ph\.?\s?d|doctor(al|ate)", re.I), 4),          # → PhD
        (
            re.compile(
                r"\bm\.?\s?tech\b|\bm\.e\.?\b|\bm\.s\.?\b|\bm\.?\s?sc\b"
                r"|\bmasters?\b|\bmba\b|\bpgdm\b|post[\s-]?graduat"
                r"|\bdual degree\b",
                re.I,
            ),
            3,                                                        # → Masters
        ),
        (
            re.compile(
                r"\bb\.?\s?tech\b|\bb\.e\.?\b|\bb\.s\.?\b|\bb\.?\s?sc\b"
                r"|\bbachelors?\b|\bbca\b|\bundergraduate\b",
                re.I,
            ),
            2,                                                        # → Bachelors
        ),
        # The training schema has no "Diploma" class. Mapping it to High
        # School (the nearest level below) is a lossy choice made explicit
        # here rather than hidden: a diploma holder is scored as school-level.
        (re.compile(r"\bdiploma\b", re.I), 1),
        (
            re.compile(
                r"class\s*(xii|xi|x|12|11|10)\b|\b1[02]th\b|higher secondary"
                r"|senior secondary|secondary school|\bmatriculation\b"
                r"|\bintermediate\b|\bhsc\b|\bssc\b",
                re.I,
            ),
            1,                                                        # → High School
        ),
    ]

    # ── experience ───────────────────────────────────────────────────
    # Word-bounded so "presentation" doesn't false-trigger. "currently" is
    # listed explicitly because \bcurrent\b will not match inside it.
    _PRESENT = re.compile(
        r"\b(present|presently|current|currently|ongoing|till date|to date)\b",
        re.I,
    )
    _YEAR = re.compile(r"\b(19[5-9]\d|20\d{2})\b")

    # ── skills ───────────────────────────────────────────────────────
    # Extension point: grow as real resumes reveal new spellings.
    _SKILL_ALIASES = {
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "np": "numpy",
        "sklearn": "scikit-learn",
        "tf": "tensorflow",
        "k8s": "kubernetes",
        "postgres": "postgresql",
        "node": "node.js",
        "cpp": "c++",
    }

    def __init__(
        self,
        reference_year: int | None = None,
        imputation_defaults: dict[str, float] | None = None,
    ):
        # reference_year is injectable so tests are deterministic; it
        # resolves "2019 - Present" style ranges.
        self.reference_year = reference_year or datetime.now().year
        self.defaults = {**_FALLBACK_DEFAULTS, **(imputation_defaults or {})}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(
        self,
        profile: CandidateProfile,
        job: JobRequirement,
        *,
        document: Document | None = None,
        resume_length: int | None = None,
        github_activity: int | None = None,
    ) -> FeatureVector:
        """Map a CandidateProfile onto the six training features.

        Args:
            profile: extracted candidate data.
            job: the role being screened for. REQUIRED — see below.
            document: source document; used to derive resume_length.
            resume_length: explicit override (word count).
            github_activity: contribution score. Not derivable from a PDF;
                supply it if you have it, otherwise it is imputed and flagged.

        Raises:
            ValueError: if `job` has no required_skills. skills_match_score
                is the model's dominant feature; imputing it would produce a
                confident number with no basis. Refusing is the honest failure.
        """
        if not job or not job.required_skills:
            raise ValueError(
                "A JobRequirement with at least one required skill is needed: "
                "skills_match_score cannot be computed without a job, and "
                "imputing the model's strongest feature would invent a result."
            )

        imputed: list[str] = []

        education_level = self._education_level(profile.education)
        if education_level is None:
            education_level = int(self.defaults["education_level"])
            imputed.append("education_level")

        if github_activity is None:
            github_activity = int(self.defaults["github_activity"])
            imputed.append("github_activity")

        length = self._resume_length(document, resume_length)
        if length is None:
            length = 0
            imputed.append("resume_length")

        return FeatureVector(
            years_experience=self._experience_years(profile.experience),
            skills_match_score=self._skills_match_score(profile.skills, job),
            education_level=education_level,
            project_count=len(profile.projects),
            resume_length=length,
            github_activity=max(0, int(github_activity)),
            imputed=imputed,
        )

    # ------------------------------------------------------------------
    # Individual feature computations
    # ------------------------------------------------------------------

    def _education_level(self, education: list[str]) -> int | None:
        """Highest qualification, on the training scale 1-4.

        1 = High School, 2 = Bachelors, 3 = Masters, 4 = PhD.
        Returns None when nothing matched — the caller imputes and flags,
        rather than this method returning a 0 the model never saw.
        """
        best: int | None = None
        for entry in education:
            for pattern, level in self._EDUCATION_LEVELS:
                if pattern.search(entry):
                    best = level if best is None else max(best, level)
                    break  # patterns are ordered high→low; first hit wins
        return best

    def _experience_years(self, experience: list[str]) -> float:
        """Best-effort total years, at YEAR granularity.

        Per entry: span = max(year) - min(year), with "present" counting as
        the reference year. Entries with fewer than two year signals
        contribute 0.

        Two documented consequences:
          - A 3-month internship ("May 2025 - July 2025") scores 0.0.
          - Overlapping roles double-count, because spans are summed.
        Month-level parsing fixes both and is the obvious V3 upgrade.
        """
        total = 0.0
        for entry in experience:
            years = [int(y) for y in self._YEAR.findall(entry)]
            if self._PRESENT.search(entry):
                years.append(self.reference_year)
            years = [y for y in years if y <= self.reference_year]
            if len(years) >= 2:
                total += max(years) - min(years)
        # Cap absurd totals from bad parses (a stray "1995" in an address).
        return float(min(total, 40.0))

    def _skills_match_score(self, skills: list[str], job: JobRequirement) -> float:
        """Percentage of required skills the candidate has, 0-100.

        Scale matters: the training column is 0-100, not 0-1. A 0-1 value
        fed to a scaler fitted on 0-100 lands ~4 standard deviations below
        the training mean — the model would read every candidate as having
        essentially no relevant skills, and would still return a confident
        answer. Unit mismatches are the quietest bug in an ML pipeline.
        """
        have = {self._canonical(s) for s in skills}
        required = {self._canonical(s) for s in job.required_skills}
        required.discard("")
        if not required:
            raise ValueError("JobRequirement.required_skills contained no usable skills.")
        return 100.0 * len(have & required) / len(required)

    def _resume_length(
        self,
        document: Document | None,
        override: int | None,
    ) -> int | None:
        """Resume length in WORDS.

        Inference, not documented fact: the training column ranges roughly
        180-800. As characters that would be a two-line resume, so the unit
        is almost certainly words. If the real dataset turns out to be
        characters, change this method to len(text) and retrain — the fix is
        one line plus a retrain, but only if you notice, which is why the
        assumption is written down here.
        """
        if override is not None:
            return max(0, int(override))
        if document is not None and document.text:
            return len(document.text.split())
        return None

    def _canonical(self, skill: str) -> str:
        """lowercase, collapse whitespace, resolve known aliases."""
        s = re.sub(r"\s+", " ", skill.strip().lower())
        return self._SKILL_ALIASES.get(s, s)
