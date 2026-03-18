import re


COMMON_ARABIC_OCR_FIXES = {
    "موسسة": "مؤسسة",
    "موسسه": "مؤسسة",
    "نجل": "نجد",
    "الخليع": "الخليج",
    "الخليح": "الخليج",
    "الصتامية": "الصناعية",
    "الصتاعية": "الصناعية",
    "الصتاعيه": "الصناعية",
    "الشويخ الصتامية": "الشويخ الصناعية",
    "الشويخ الصتاعية": "الشويخ الصناعية",
    "نلادوات": "للأدوات",
    "الادوات": "الأدوات",
    "يرجاء": "يرجى",
    "اليك": "الشيك",
    "الصحيه": "الصحية",
}

COMMON_VENDOR_FIXES = {
    "Najd Al-Khaleej Est, sanitary ware": "Najd Al-Khaleej Est. for Sanitary Ware",
    "Najd Al-Khaleej Est sanitary ware": "Najd Al-Khaleej Est. for Sanitary Ware",
    "Najd Al Khaleej Est, sanitary ware": "Najd Al-Khaleej Est. for Sanitary Ware",
    "Najd Al Khaleej Est sanitary ware": "Najd Al-Khaleej Est. for Sanitary Ware",
    "مؤسسة نجد الخليج للأدوات الصحية": "مؤسسة نجد الخليج للأدوات الصحية",
}


def normalize_arabic_letters(text: str) -> str:
    if not text:
        return ""

    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    return text


def cleanup_arabic_ocr_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text

    # إزالة العلامات المخفية واتجاه النص
    cleaned = cleaned.replace("\u200f", " ").replace("\u200e", " ")
    cleaned = cleaned.replace("‎", " ").replace("‏", " ")

    # توحيد المسافات
    cleaned = re.sub(r"\s+", " ", cleaned)

    # تصحيح مباشر للكلمات الشائعة
    for wrong, correct in COMMON_ARABIC_OCR_FIXES.items():
        cleaned = cleaned.replace(wrong, correct)

    normalized_fixes = {
        normalize_arabic_letters("مؤسسة"): "مؤسسة",
        normalize_arabic_letters("نجد"): "نجد",
        normalize_arabic_letters("الخليج"): "الخليج",
        normalize_arabic_letters("الشويخ"): "الشويخ",
        normalize_arabic_letters("الصناعية"): "الصناعية",
        normalize_arabic_letters("للأدوات"): "للأدوات",
        normalize_arabic_letters("الصحية"): "الصحية",
    }

    words = cleaned.split()
    corrected_words = []

    for word in words:
        normalized_word = normalize_arabic_letters(word)
        if normalized_word in normalized_fixes:
            corrected_words.append(normalized_fixes[normalized_word])
        else:
            corrected_words.append(word)

    cleaned = " ".join(corrected_words)

    # إصلاح عبارات كاملة
    phrase_fixes = {
        "مؤسسة نجد الخليع": "مؤسسة نجد الخليج",
        "مؤسسة نجد الخليح": "مؤسسة نجد الخليج",
        "الشويخ الصتامية": "الشويخ الصناعية",
        "الشويخ الصتاعية": "الشويخ الصناعية",
        "مؤسسة نجد Callie للأدوات الصحية": "مؤسسة نجد الخليج للأدوات الصحية",
        "مؤسسة نجد Cailie للأدوات الصحية": "مؤسسة نجد الخليج للأدوات الصحية",
        "مؤسسة نجد الخليج للادوات الصحية": "مؤسسة نجد الخليج للأدوات الصحية",
    }

    for wrong, correct in phrase_fixes.items():
        cleaned = cleaned.replace(wrong, correct)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def cleanup_vendor_name(vendor_name: str | None) -> str | None:
    if not vendor_name:
        return vendor_name

    cleaned = vendor_name.strip()

    cleaned = cleaned.replace("‎", " ").replace("‏", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)

    # تنظيف بعض الرموز
    cleaned = cleaned.replace(")", "").replace("(", "")
    cleaned = cleaned.replace(" ,", ",").replace(" .", ".")

    # تنظيف عربي
    cleaned = cleanup_arabic_ocr_text(cleaned)

    # إصلاحات مباشرة لأسماء الموردين
    for wrong, correct in COMMON_VENDOR_FIXES.items():
        if cleaned.lower() == wrong.lower():
            cleaned = correct

    # تحسينات إنجليزية شائعة
    cleaned = cleaned.replace("Est,", "Est.")
    cleaned = cleaned.replace("Est ,", "Est.")
    cleaned = cleaned.replace("sanitary ware", "Sanitary Ware")
    cleaned = cleaned.replace("Sanitary ware", "Sanitary Ware")

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_,.;")

    return cleaned or None