import fitz  # PyMuPDF
from pathlib import Path


def extract_text_from_pdf(file_path: str) -> str:
    path = Path(file_path)

    if path.suffix.lower() != ".pdf":
        return ""

    text_parts = []

    with fitz.open(file_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())

    return "\n".join(text_parts).strip()