from typing import Optional

from pydantic import BaseModel, Field

class Document(BaseModel):
    """
    Standard representaion of any extracted document

    This Model is independent of the source document type
    (PDF, DOCX, Image, HTMl, etc..)
    """
    text : str
    
    metadata :dict = Field(default_factory=dict)
    
    filename : Optional[str] = None
    
    file_type : Optional[str] = None
    
    page_count: Optional[int] = None
    
    warnings: list[str] = Field(default_factory=list)
    