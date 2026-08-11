"""
Run the full pipeline on one resume.

    PDF -> Document -> Gemini -> CandidateProfile -> FeatureVector -> Prediction

Usage (from the project root):

    python -m src.main data/sample_resume.pdf \\
        --job-skills "python, fastapi, docker, scikit-learn" \\
        --job-title "ML Intern"

    # Skip the LLM and score six features directly (no API key needed):
    python -m src.main --features 3 82 Bachelors 6 480 210

Requires a trained model (`python -m src.ml.train`), and GEMINI_API_KEY in
.env for the PDF path.
"""

import argparse
import json
import logging
import sys

from src.ml.predict import ScreeningModel
from src.models.feature_vector import EDUCATION_MAP, FeatureVector
from src.models.job_requirement import JobRequirement
from src.pipeline import ScreeningPipeline


def _build_job(args: argparse.Namespace) -> JobRequirement:
    skills = [s.strip() for s in (args.job_skills or "").split(",") if s.strip()]
    return JobRequirement(title=args.job_title, required_skills=skills)


def _print_prediction(prediction) -> None:
    print(f"\n  Shortlisted: {prediction.shortlisted}")
    print(f"  Confidence:  {prediction.confidence:.2f}%")
    print(f"  P(shortlist): {prediction.probability_shortlisted:.2f}%")
    print(f"  Model:       {prediction.model_name}  [{prediction.model_source}]")
    if prediction.imputed_fields:
        print(
            f"  ⚠ Imputed (not measured from the resume): "
            f"{', '.join(prediction.imputed_fields)}"
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="AI Resume Intelligence — screening pipeline")
    parser.add_argument("pdf_path", nargs="?", help="Path to the resume PDF")
    parser.add_argument(
        "--job-skills", default=None,
        help='Comma-separated required skills, e.g. "python, docker, fastapi"',
    )
    parser.add_argument("--job-title", default=None, help="Optional job title")
    parser.add_argument(
        "--github-activity", type=int, default=None,
        help="GitHub contribution score. Omit and it is imputed (and flagged).",
    )
    parser.add_argument(
        "--features", nargs=6, default=None,
        metavar=("YEARS", "MATCH", "EDUCATION", "PROJECTS", "LENGTH", "GITHUB"),
        help="Score six features directly, skipping the PDF and the LLM.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    try:
        # ── Direct feature mode: no PDF, no LLM, no API key ───────────
        if args.features:
            years, match, education, projects, length, github = args.features
            if education not in EDUCATION_MAP:
                raise ValueError(
                    f"education must be one of {list(EDUCATION_MAP)}, got '{education}'"
                )
            features = FeatureVector(
                years_experience=float(years),
                skills_match_score=float(match),
                education_level=EDUCATION_MAP[education],
                project_count=int(projects),
                resume_length=int(length),
                github_activity=int(github),
            )
            prediction = ScreeningModel.from_files().predict(features)
            if args.json:
                print(json.dumps(prediction.to_dict(), indent=2))
            else:
                _print_prediction(prediction)
            return 0

        # ── Full pipeline mode ────────────────────────────────────────
        if not args.pdf_path:
            parser.error("provide a pdf_path, or use --features")
        job = _build_job(args)
        if not job.required_skills:
            parser.error(
                "--job-skills is required: the model's strongest feature is "
                "skills_match_score, which has no meaning without a job."
            )

        pipeline = ScreeningPipeline()
        result = pipeline.run(args.pdf_path, job, github_activity=args.github_activity)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
            return 0

        print("\n[1/3] CandidateProfile")
        print(result.profile.model_dump_json(indent=2))
        print("\n[2/3] FeatureVector")
        for name, value in result.features.to_dict().items():
            flag = "  ← imputed" if name in result.features.imputed else ""
            print(f"    {name}: {value}{flag}")
        print("\n[3/3] Prediction")
        _print_prediction(result.prediction)
        return 0

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
