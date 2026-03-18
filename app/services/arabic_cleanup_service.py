import re


def cleanup_arabic_ocr_text(text: str | None) -> str | None:
    if not text:
        return text

    text = str(text)

    corrections = {
        "موسسة": "مؤسسة",
        "مؤسسة": "مؤسسة",
        "الخل يج": "الخليج",
        "التجاربة": "التجارية",
        "وزاره": "وزارة",
        "الماليه": "المالية",
        "شركه": "شركة",
        "الكويتيه": "الكويتية",
        "اإلجمالي": "الإجمالي",
    }

    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)

    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _extract_arabic_part(text: str) -> str:
    if not text:
        return text

    arabic_parts = re.findall(r"[\u0600-\u06FF\s]+", text)
    arabic_parts = [p.strip() for p in arabic_parts if len(p.strip()) > 2]

    if not arabic_parts:
        return text.strip()

    return max(arabic_parts, key=len).strip()


def cleanup_vendor_name(name: str | None) -> str | None:
    if not name:
        return name

    name = cleanup_arabic_ocr_text(name)
    name = _extract_arabic_part(name)
    name = cleanup_arabic_ocr_text(name)

    return name.strip() if name else name