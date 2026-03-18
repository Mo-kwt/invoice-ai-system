import fitz  # PyMuPDF
from pathlib import Path
import pytesseract
from PIL import Image, ImageOps, ImageFilter
import io

from app.services.arabic_cleanup_service import cleanup_arabic_ocr_text

# عدل هذا المسار إذا كان Tesseract مثبتًا عندك في مكان مختلف
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    image = ImageOps.autocontrast(image)

    width, height = image.size
    image = image.resize((width * 2, height * 2))

    image = image.filter(ImageFilter.SHARPEN)

    threshold = 180
    image = image.point(lambda p: 255 if p > threshold else 0)

    return image


def extract_text_from_pdf(file_path: str) -> str:
    path = Path(file_path)

    if path.suffix.lower() != ".pdf":
        return ""

    text_parts = []

    with fitz.open(file_path) as doc:
        for page in doc:
            direct_text = page.get_text().strip()

            if len(direct_text) > 50:
                cleaned_direct_text = cleanup_arabic_ocr_text(direct_text)
                text_parts.append(cleaned_direct_text)
                continue

            matrix = fitz.Matrix(3, 3)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_bytes = pix.tobytes("png")

            image = Image.open(io.BytesIO(img_bytes))
            image = preprocess_image_for_ocr(image)

            ocr_text = pytesseract.image_to_string(
                image,
                lang="ara+eng",
                config="--oem 3 --psm 6"
            )

            cleaned_ocr_text = cleanup_arabic_ocr_text(ocr_text)
            text_parts.append(cleaned_ocr_text)

    return "\n".join(text_parts).strip()