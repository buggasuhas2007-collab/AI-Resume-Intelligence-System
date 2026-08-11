from pathlib import Path
import fitz  #PyMuPDF

from src.models.document import Document

class PDFReader:
    """
    Responsible for extracting text from PDF documents.

    Reads a PDF document and converts it into a standardized Document object.
    """
    
    def read(self, file_path: str) -> Document :
        
        """
        Reads a PDF file and returns a Document object.
        
        Args: 
            file_path: Path to PDF file
        
        Returns:
            Document 
        """

        # ----------------------------
        # Step 1: Validate file
        # ----------------------------

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError("Input File must be a PDF.")
        
        page_texts =[]
        warnings =[]
        
        
        # ----------------------------
        # Step 2: Open PDF safely
        # ----------------------------

        try:
            with fitz.open(path) as pdf:

                metadata = pdf.metadata or {}

                page_count = pdf.page_count

                # ----------------------------
                # Step 3: Extract text
                # ----------------------------

                for page_number, page in enumerate(pdf):

                    text = page.get_text().strip()

                    if not text:
                        warnings.append(
                            f"No extractable text found on page {page_number + 1}."
                        )

                    page_texts.append(text)

        except Exception as e:
            raise RuntimeError(
                f"Unable to read PDF '{file_path}'."
            ) from e

        # ----------------------------
        # Step 4: Merge page texts
        # ----------------------------

        complete_text = "\n\n".join(page_texts)

        # ----------------------------
        # Step 5: Build Document
        # ----------------------------

        return Document(
            text=complete_text,
            metadata=metadata,
            filename=path.name,
            file_type="pdf",
            page_count=page_count,
            warnings=warnings,
        )