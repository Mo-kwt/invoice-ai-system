import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import io

from app.services.arabic_cleanup_service import cleanup_arabic_ocr_text


def _preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    # تحويل إلى grayscale
    image = image.convert("L")

    # تحسين التباين
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # تحسين الحدة
    image = image.filter(ImageFilter.SHARPEN)

    # threshold (تحويل إلى أبيض وأسود)
    image = image.point(lambda x: 0 if x < 140 else 255, "1")

    return image


def extract_text_from_pdf(pdf_path: str) -> str:
    text_content = ""

    doc = fitz.open(pdf_path)

    for page_number in range(len(doc)):
        page = doc[page_number]

        # محاولة استخراج النص مباشرة
        text = page.get_text()

        if text and text.strip():
            text_content += text + "\n"
        else:
            # fallback إلى OCR
            pix = page.get_pixmap()

            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))

            # 🔥 preprocessing هنا
            image = _preprocess_image_for_ocr(image)

            ocr_text = pytesseract.image_to_string(
                image,
                lang="ara+eng",
                config="--oem 3 --psm 6"
            )

            text_content += ocr_text + "\n"

    doc.close()

    # تنظيف عربي إضافي
    text_content = cleanup_arabic_ocr_text(text_content)

    return text_content.strip()