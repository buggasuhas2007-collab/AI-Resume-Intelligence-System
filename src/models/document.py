from pydantic import BaseModel, Field


class Document(BaseModel):
    """A plain-text document extracted from a source file (V1: PDFs only).

    Downstream components depend only on `text`, so the extraction source
    (PDF today; DOCX, images+OCR later) can change without touching them.
    """

    text: str = Field(min_length=1)
    source_name: str | None = None
    page_count: int | None = Field(default=None, ge=1)
