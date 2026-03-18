import re
from copy import deepcopy

from app.services.normalization_service import (
    normalize_currency,
    normalize_date,
    normalize_digits,
    normalize_amount,
)
from app.services.arabic_cleanup_service import cleanup_vendor_name


INVALID_INVOICE_NUMBER_WORDS = {
    "description",
    "qty",
    "quantity",
    "price",
    "amount",
    "total",
    "item",
    "items",
    "fits",
    "unit",
    "kd",
    "kwd",
    "no",
}


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_digits(text)
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_currency_in_text(text: str):
    text = _clean_text(text)

    currency_patterns = [
        r"\bKWD\b",
        r"\bKD\b",
        r"\bK\.D\.?\b",
        r"د\.?\s*ك",
        r"دينار\s+كويتي",
        r"kuwaiti\s+dinar",
    ]

    for pattern in currency_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_currency(match.group(0))

    return None


def find_date_in_text(text: str):
    text = _clean_text(text)

    labeled_patterns = [
        r"(?:invoice\s*date|credit\s*invoice\s*date|credit\s*inverce\s*date|date|التاريخ|تاريخ)\s*[:\-]?\s*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})",
        r"(?:invoice\s*date|credit\s*invoice\s*date|credit\s*inverce\s*date|date|التاريخ|تاريخ)\s*[:\-]?\s*([0-9]{4}[\/\-][0-9]{1,2}[\/\-][0-9]{1,2})",
    ]

    for pattern in labeled_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_date(match.group(1))

    generic_patterns = [
        r"\b([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{4})\b",
        r"\b([0-9]{4}[\/\-][0-9]{1,2}[\/\-][0-9]{1,2})\b",
    ]

    for pattern in generic_patterns:
        match = re.search(pattern, text)
        if match:
            return normalize_date(match.group(1))

    return None


def _is_valid_invoice_number(candidate: str) -> bool:
    if not candidate:
        return False

    candidate = candidate.strip().strip(":#-/")
    candidate_upper = candidate.upper()

    if candidate.lower() in INVALID_INVOICE_NUMBER_WORDS:
        return False

    if re.fullmatch(r"[A-Z]+", candidate_upper):
        return False

    if len(candidate) < 2:
        return False

    if re.fullmatch(r"\d{8,}", candidate):
        return False

    has_digit = any(ch.isdigit() for ch in candidate)
    has_alpha = any(ch.isalpha() for ch in candidate)

    if not has_digit and not has_alpha:
        return False

    if has_alpha and not has_digit and len(candidate) < 6:
        return False

    return True


def find_invoice_number_in_text(text: str):
    text = _clean_text(text)

    labeled_patterns = [
        r"(?:invoice\s*no|invoice\s*number|inv\s*no|رقم\s*الفاتورة)\s*[:\-#]?\s*([A-Z0-9\-\/]{2,})",
        r"(?:credit\s*invoice\s*no|credit\s*note\s*no|credit\s*inverce\s*no)\s*[:\-#]?\s*([A-Z0-9\-\/]{2,})",
    ]

    for pattern in labeled_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for candidate in matches:
            candidate = candidate.strip()
            if _is_valid_invoice_number(candidate):
                return candidate

    fallback_pattern = r"\bNo\s*[:\-#]?\s*([A-Z]*\d+[A-Z0-9\-\/]*)\b"
    matches = re.findall(fallback_pattern, text, flags=re.IGNORECASE)
    for candidate in matches:
        candidate = candidate.strip()
        if _is_valid_invoice_number(candidate):
            return candidate

    return None


def find_total_in_text(text: str):
    text = _clean_text(text)

    labeled_patterns = [
        r"(?:grand\s*total|total|net|amount|الإجمالي|المجموع|الصافي)\s*[:\-]?\s*([0-9][0-9,\.\s]*)",
    ]

    for pattern in labeled_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for candidate in matches:
            amount = normalize_amount(candidate)
            if amount is not None:
                return amount

    currency_amount_patterns = [
        r"([0-9][0-9,\.\s]*)\s*(?:KWD|KD|K\.D\.?)",
        r"(?:KWD|KD|K\.D\.?)\s*([0-9][0-9,\.\s]*)",
    ]

    for pattern in currency_amount_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for candidate in matches:
            amount = normalize_amount(candidate)
            if amount is not None:
                return amount

    return None


def remove_empty_items(data: dict):
    items = data.get("items", [])
    cleaned_items = []

    for item in items:
        if not item:
            continue

        has_real_value = any(
            item.get(field) not in (None, "", 0, 0.0)
            for field in ["description", "quantity", "unit_price", "total_price"]
        )

        if has_real_value:
            cleaned_items.append(item)

    data["items"] = cleaned_items
    return data


def enrich_normalized_invoice_data(extracted_text: str, raw_invoice_data: dict, normalized_invoice_data: dict):
    result = deepcopy(normalized_invoice_data)

    # تنظيف اسم المورد أولًا
    result["vendor_name"] = cleanup_vendor_name(result.get("vendor_name"))

    raw_currency = raw_invoice_data.get("currency")
    if raw_currency:
        result["currency"] = normalize_currency(raw_currency)

    if not result.get("currency"):
        result["currency"] = find_currency_in_text(extracted_text)

    raw_date = raw_invoice_data.get("invoice_date")
    if raw_date:
        result["invoice_date"] = normalize_date(raw_date)

    if not result.get("invoice_date"):
        result["invoice_date"] = find_date_in_text(extracted_text)

    if not result.get("invoice_number"):
        result["invoice_number"] = find_invoice_number_in_text(extracted_text)

    if result.get("total") is None:
        result["total"] = find_total_in_text(extracted_text)

    result = remove_empty_items(result)

    return result