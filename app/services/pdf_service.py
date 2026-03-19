import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageFilter, ImageEnhance

from app.services.arabic_cleanup_service import cleanup_arabic_ocr_text
from app.services.ocr_service import extract_text_from_image


def _preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    image = image.convert("L")

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.8)

    image = image.filter(ImageFilter.SHARPEN)

    image = image.point(lambda x: 0 if x < 160 else 255, "L")

    return image


def render_first_page_to_image(pdf_path: str, output_path: str) -> str:
    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        matrix = fitz.Matrix(2.5, 2.5)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(output_file))

        return str(output_file)
    finally:
        doc.close()


def extract_text_from_pdf(pdf_path: str) -> str:
    text_content = ""

    doc = fitz.open(pdf_path)

    try:
        for page_number in range(len(doc)):
            page = doc[page_number]

            text = page.get_text()

            if text and text.strip():
                text_content += text + "\n"
                continue

            matrix = fitz.Matrix(2.5, 2.5)
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))

            image = _preprocess_image_for_ocr(image)

            ocr_text = extract_text_from_image(image)

            text_content += ocr_text + "\n"

    finally:
        doc.close()

    text_content = cleanup_arabic_ocr_text(text_content)

    return text_content.strip()