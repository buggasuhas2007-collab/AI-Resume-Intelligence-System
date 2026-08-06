from pathlib import Path
import fitz

class PDFReader:
    """
    Responsible for extracting text from PDF documents.

    This class has only one responsibility:
    Read a PDF and return its extracted content.
    """
    
    def read(self, file_path: str) -> dict :
        
        """
        Responsible for extracting text from PDF documents.

        This class has only one responsibility:
        Read a PDF and return its extracted content.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        document = fitz.open(path)

        extracted_text = ""

        for page in document:
            extracted_text += page.get_text()

        return {
            "text": extracted_text,
            "page_count": len(document),
            "metadata": document.metadata
        }

