from pathlib import Path
import pytesseract

from app.config import settings


class OCRConfigurationError(RuntimeError):
    pass


def configure_tesseract():
    if not settings.tesseract_cmd:
        raise OCRConfigurationError("Tesseract path is not configured.")

    tesseract_path = Path(settings.tesseract_cmd)

    if not tesseract_path.exists():
        raise OCRConfigurationError(
            f"Tesseract executable was not found at: {settings.tesseract_cmd}"
        )

    pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)


def _score_text(text: str) -> int:
    """
    نقيم جودة النص:
    - عدد الحروف العربية والإنجليزية
    - وجود أرقام
    - طول النص
    """
    if not text:
        return 0

    score = 0

    # طول النص
    score += len(text)

    # حروف عربية
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    score += arabic_chars * 2

    # أرقام
    digits = sum(1 for c in text if c.isdigit())
    score += digits * 2

    return score


def extract_text_from_image(image) -> str:
    configure_tesseract()

    configs = [
        "--oem 3 --psm 6",   # فقرة
        "--oem 3 --psm 4",   # أعمدة
        "--oem 3 --psm 11",  # نص مبعثر
        "--oem 3 --psm 3",   # تلقائي كامل
    ]

    best_text = ""
    best_score = -1

    for cfg in configs:
        try:
            text = pytesseract.image_to_string(
                image,
                lang=settings.tesseract_lang,
                config=cfg
            )

            score = _score_text(text)

            if score > best_score:
                best_score = score
                best_text = text

        except Exception:
            continue

    return best_text