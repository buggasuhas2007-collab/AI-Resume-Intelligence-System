import json
import os

from dotenv import load_dotenv
from google import genai
from pydantic import ValidationError

from src.models.candidate_profile import CandidateProfile
from src.models.document import Document


class ResumeParser:
    """Converts a Document into a structured CandidateProfile using Gemini.

    V1 philosophy for LLM output handling:
        Repair syntax. Don't repair meaning.
    We fix formatting problems (markdown fences, string-instead-of-list),
    but we never invent or guess content the model did not return.

    V1 keeps construction simple: the client is built here from the
    GEMINI_API_KEY environment variable. Dependency injection, retries,
    and Gemini's native structured output are planned V2 refactors.
    """

    MODEL_NAME = "gemini-2.5-flash"

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please add it to your .env file."
            )
        self.client = genai.Client(api_key=api_key)

    def parse(self, document: Document) -> CandidateProfile:
        """Parse a resume document into a CandidateProfile.

        Raises ValueError for every failure mode at this boundary: empty
        response, invalid JSON, JSON that is not an object, or data that
        fails CandidateProfile validation.
        """
        prompt = self._build_prompt(document)

        response = self.client.models.generate_content(
            model=self.MODEL_NAME,
            contents=prompt,
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        cleaned = self._clean_response(response.text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError("Gemini returned invalid JSON.") from e

        # json.loads happily returns lists, strings, and numbers too.
        # CandidateProfile(**data) needs a dict, so guard here with a
        # clearer error than the TypeError we would otherwise get.
        if not isinstance(data, dict):
            raise ValueError("Gemini returned valid JSON, but not a JSON object.")

        data = self._normalize_data(data)

        try:
            return CandidateProfile(**data)
        except ValidationError as e:
            raise ValueError("CandidateProfile validation failed.") from e

    def _build_prompt(self, document: Document) -> str:
        """Build the prompt sent to Gemini."""
        role = """
You are an expert AI Resume Parser.
"""
        task = """
Extract structured information from the resume below.
"""
        rules = """
Rules:
1. Return ONLY valid JSON.
2. Do NOT explain anything.
3. Do NOT include markdown.
4. Do NOT wrap JSON inside ```json.
5. Never guess or invent information that is not in the resume.
6. Every array element must be a plain string — never an object.
7. If information is missing:
   - name -> null
   - education -> []
   - skills -> []
   - projects -> []
   - experience -> []
"""
        # Types are described instead of showing literal empty lists, so the
        # model does not copy the empties verbatim for a filled resume.
        schema = """
Expected JSON format:
{
    "name": "full name as a string, or null",
    "education": ["one string per degree/qualification, including institute and years"],
    "skills": ["one string per skill"],
    "projects": ["one string per project, including a short description"],
    "experience": ["one string per role, including organization and dates"]
}
"""
        resume = f"""
Resume:
{document.text}
"""
        return "\n".join([role, task, rules, schema, resume])

    def _clean_response(self, response: str) -> str:
        """Remove markdown code fences if present.

        LLMs frequently wrap JSON in ```json fences despite instructions —
        this is syntax repair, so it is allowed.
        """
        response = response.strip()
        if response.startswith("```json"):
            response = response.replace("```json", "", 1)
        if response.startswith("```"):
            response = response.replace("```", "", 1)
        if response.endswith("```"):
            response = response[:-3]
        return response.strip()

    def _normalize_data(self, data: dict) -> dict:
        """Perform safe, purely syntactic corrections before validation.

        - missing/null list field  -> []
        - single string where a list belongs -> [string]
        - name whitespace stripped; empty name -> null
        Anything else (e.g. a list of objects) is left untouched and will
        fail validation loudly — that is meaning, not syntax.
        """
        list_fields = ["education", "skills", "projects", "experience"]

        for field in list_fields:
            if field not in data or data[field] is None:
                data[field] = []
            elif isinstance(data[field], str):
                data[field] = [data[field]]

        if "name" not in data or data["name"] is None:
            data["name"] = None
        elif isinstance(data["name"], str):
            data["name"] = data["name"].strip() or None

        return data
